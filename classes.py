import numpy as np


class PartType:
    def __init__(self, name: str, cost: float, value: float = 0.0):
        """
        Repräsentiert einen Typ von Teil.
        cost: Verarbeitungskosten für diesen Typ.
        value: Verkaufswert bzw. hergeleiteter Wert (wird bei finalen Produkten explizit gesetzt).
        """
        self.name = name
        self.cost = cost
        self.value = value


class Transformation:
    def __init__(self, input_types, output_type: PartType, duration: int):
        """
        Definiert, wie Teile eines oder mehrerer Input-Typen in einen Output-Typ transformiert werden.
        input_types: Liste von PartTypes, die als Input benötigt werden. Falls ein einzelner PartType
                     übergeben wird, wird dieser intern in eine Liste umgewandelt.
        output_type: Der resultierende PartType.
        duration: Anzahl der Zeitschritte, die diese Transformation benötigt.
        """
        if not isinstance(input_types, list):
            input_types = [input_types]
        self.input_types = input_types
        self.output_type = output_type
        self.duration = duration


class MachineType:
    def __init__(self, name: str, slots: int, transformations: list):
        """
        Repräsentiert den Typ einer Maschine.
        slots: Maximale Anzahl simultan laufender Transformationen (Jobs).
        transformations: Liste der Transformationen, die diese Maschine durchführen kann.
        """
        self.name = name
        self.slots = slots
        self.transformations = transformations


class Part:
    def __init__(self, part_id: int, part_type: PartType):
        """
        Repräsentiert ein konkretes Teil-Exemplar.
        part_id: Eindeutige Identifikation.
        part_type: Der zugehörige Typ (enthält Kosten und Wert).
        """
        self.id = part_id
        self.type = part_type


class Product:
    def __init__(self, name: str, part_type: PartType, sale_value: float):
        """
        Ein finales Produkt, das zu einem bestimmten PartType gehört.
        sale_value: Der Verkaufswert des finalen Produkts.
        """
        self.name = name
        self.part_type = part_type
        self.sale_value = sale_value


class Machine:
    def __init__(self, machine_type: MachineType, machine_id: str):
        """
        Repräsentiert eine konkrete Maschine.
        machine_type: Der Maschinentyp, der z.B. Slots und unterstützte Transformationen definiert.
        machine_id: Eindeutige Kennung der Maschine.
        """
        self.machine_type = machine_type
        self.machine_id = machine_id
        self.input_buffer = []  # Liste von Parts, die auf Transformation warten.
        self.output_buffer = []  # Ergebnisse abgeschlossener Transformationen, die noch nicht in den globalen Puffer übertragen wurden.
        self.current_jobs = []  # Liste aktiver Jobs; jeder Job ist ein Dict mit "transformation", "input_parts" und "remaining_time".
        self.transformation_priority = list(machine_type.transformations)
        self.connected_machines = []  # Liste anderer Maschinen – wird extern gesetzt.

    def can_start_transformation(self, transformation: Transformation):
        """
        Prüft, ob ausreichend passende Parts im Input-Puffer vorhanden sind, um diese Transformation zu starten.
        """
        required = {}
        for pt in transformation.input_types:
            required[pt.name] = required.get(pt.name, 0) + 1
        available = {}
        for part in self.input_buffer:
            available[part.type.name] = available.get(part.type.name, 0) + 1
        for pt_name, count in required.items():
            if available.get(pt_name, 0) < count:
                return False
        return True

    def start_transformation(self, transformation: Transformation, part_id_counter: int):
        """
        Entfernt die notwendigen Parts aus dem Input-Puffer und startet einen neuen Job.
        """
        required = {}
        for pt in transformation.input_types:
            required[pt.name] = required.get(pt.name, 0) + 1
        input_parts = []
        new_input_buffer = []
        for part in self.input_buffer:
            if part.type.name in required and required[part.type.name] > 0:
                input_parts.append(part)
                required[part.type.name] -= 1
            else:
                new_input_buffer.append(part)
        self.input_buffer = new_input_buffer
        job = {
            "transformation": transformation,
            "input_parts": input_parts,
            "remaining_time": transformation.duration
        }
        self.current_jobs.append(job)
        return part_id_counter

    def progress_jobs(self, part_id_counter: int, final_product_mapping: dict):
        """
        Reduziert die verbleibende Dauer der laufenden Jobs.
        Ist ein Job abgeschlossen, wird ein neuer Part gemäß der Output-Definition erzeugt und in den Output-Puffer gelegt.
        final_product_mapping: Dictionary, das angibt, ob ein PartType als final gilt.
        """
        completed = []
        for job in self.current_jobs.copy():
            job["remaining_time"] -= 1
            if job["remaining_time"] <= 0:
                transformation = job["transformation"]
                new_part = Part(part_id_counter, transformation.output_type)
                part_id_counter += 1
                self.output_buffer.append(new_part)
                self.current_jobs.remove(job)
                completed.append(new_part)
        return part_id_counter, completed

    def reset(self):
        """
        Setzt die Maschine auf den Anfangszustand zurück.
        """
        self.input_buffer = []
        self.output_buffer = []
        self.current_jobs = []
        self.transformation_priority = list(self.machine_type.transformations)


class Anlage:
    def __init__(self, machines: list, timestep: float, input_parts: list, all_part_types: list):
        """
        Die Fertigungsanlage, bestehend aus einer Menge von Maschinen, einem globalen Puffer und weiteren Parametern.
        machines: Liste von Machine-Objekten.
        timestep: Startzeit bzw. aktueller Zeitschritt.
        input_parts: Liste von externen Input-Parts.
        all_part_types: Liste aller im System vorkommenden PartTypes.
        """
        self.machines = machines
        self.timestep = timestep
        self.input_parts = input_parts
        self.all_part_types = all_part_types
        self.global_buffer = []  # Zunächst leer.
        self.cost = 0
        self.current_value = 0

        # Berechne elementare PartTypes: jene, die nie als Output einer Transformation erscheinen (als Rohmaterial).
        self.elementary_part_types = self.compute_elementary_part_types()
        self.part_id_counter = 0

    def compute_elementary_part_types(self):
        """
        Identifiziert alle PartTypes, die niemals als Output einer Transformation auftreten.
        Diese gelten als elementare (Roh-) Teiltypen.
        """
        output_types = set()
        for machine in self.machines:
            for transformation in machine.machine_type.transformations:
                output_types.add(transformation.output_type)
        elementary = [pt for pt in self.all_part_types if pt not in output_types]
        return elementary

    def next_part_id(self):
        """
        Liefert eine eindeutige Part-ID.
        """
        pid = self.part_id_counter
        self.part_id_counter += 1
        return pid

    def refill_global_buffer(self, capacity: int):
        """
        Füllt den globalen Puffer bis zur angegebenen Kapazität auf.

        1. Zuerst werden die Output-Puffer aller Maschinen geleert und in den globalen Puffer übertragen,
           solange noch Platz vorhanden ist.
        2. Anschließend, falls noch freie Plätze existieren, werden diese gleichmäßig mit elementaren Parts
           (Rohprodukten) befüllt.
        """
        free_slots = capacity - len(self.global_buffer)
        # Schritt 1: Transfer aus den Output-Puffern.
        while free_slots > 0:
            any_transfer = False
            for machine in self.machines:
                while machine.output_buffer and free_slots > 0:
                    part = machine.output_buffer.pop(0)
                    self.global_buffer.append(part)
                    free_slots -= 1
                    any_transfer = True
            if not any_transfer:
                break

        # Schritt 2: Falls freie Plätze bleiben, gleichmäßig mit elementaren PartTypes befüllen.
        if free_slots > 0:
            if not self.elementary_part_types:
                self.elementary_part_types = [self.all_part_types[0]]
            num_elem = len(self.elementary_part_types)
            for i in range(free_slots):
                pt = self.elementary_part_types[i % num_elem]
                new_part = Part(self.next_part_id(), pt)
                self.global_buffer.append(new_part)

    def reset(self):
        """
        Setzt die Anlage inklusive aller Maschinen und globaler Variablen zurück.
        """
        self.global_buffer = []
        for machine in self.machines:
            machine.reset()
        self.timestep = 0
        self.cost = 0
        self.current_value = 0
        self.part_id_counter = 0


if __name__ == "__main__":
    # Beispiel: Erstelle hier ggf. eine komplette Fertigungsstruktur.
    print("Fertigungsanlage erfolgreich erstellt.")
