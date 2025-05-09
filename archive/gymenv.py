import gym
from gym import spaces
import numpy as np


# -------------------------------
# SIMULATIONSKLASSEN
# -------------------------------

class PartType:
    def __init__(self, name: str, cost: float, value: float = 0.0):
        """
        Repräsentiert einen Typ von Teil.
        cost: Verarbeitungskosten dieses Typs.
        value: Verkaufswert bzw. hergeleiteter Wert. Für finale Produkte wird dieser explizit gesetzt.
        """
        self.name = name
        self.cost = cost
        self.value = value  # Wird per Propagation berechnet (außer bei fertigen Produkten)


class Transformation:
    def __init__(self, input_types: [PartType], output_type: PartType, duration: int):
        """
        Definiert, wie Teile eines oder mehrerer Input-Typen in einen Output-Typ transformiert werden.
        duration: Anzahl der Zeitschritte für diese Transformation.
        """
        self.input_types = input_types  # Liste der für diese Transformation benötigten Teiltypen.
        self.output_type = output_type  # Ergebnis-Typ.
        self.duration = duration


class MachineType:
    def __init__(self, name: str, slots: int, transformations: [Transformation]):
        """
        Typ einer Maschine mit einer Menge möglicher Transformationen und vorgegebener Slot-Anzahl.
        """
        self.name = name
        self.slots = slots
        self.transformations = transformations


class Part:
    def __init__(self, part_id: int, part_type: PartType):
        """
        Repräsentiert ein konkretes Teil, das zu einem bestimmten Typ gehört.
        Es werden keine redundanten Werte (wie Kosten oder Wert) gespeichert – diese entstammen
        dem zugehörigen PartType.
        """
        self.id = part_id
        self.type = part_type


class Machine:
    def __init__(self, machine_type: MachineType, machine_id: str):
        self.machine_type = machine_type
        self.machine_id = machine_id
        self.input_buffer = []  # Liste von Part-Objekten, die auf Transformation warten.
        self.output_buffer = []  # Produzierte Teile, die noch nicht in den globalen Puffer transferiert wurden.
        self.current_jobs = []  # Aktive Transformationen (jedes Element ist ein Dict mit 'transformation', 'input_parts' und 'remaining_time').
        # Zur generischen Steuerung kann eine Prioritätsreihenfolge für Transformationen gesetzt werden.
        self.transformation_priority = list(machine_type.transformations)

    def can_start_transformation(self, transformation: Transformation):
        """Prüft, ob genügend passende Teile im Input-Puffer für diese Transformation vorhanden sind."""
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

    def start_transformation(self, transformation: Transformation, part_id_counter):
        """
        Entfernt die benötigten Teile aus dem Input-Puffer und startet einen Transformation-Job.
        """
        required = {}
        for pt in transformation.input_types:
            required[pt.name] = required.get(pt.name, 0) + 1
        input_parts = []
        remaining_buffer = []
        for part in self.input_buffer:
            if part.type.name in required and required[part.type.name] > 0:
                input_parts.append(part)
                required[part.type.name] -= 1
            else:
                remaining_buffer.append(part)
        self.input_buffer = remaining_buffer
        job = {
            'transformation': transformation,
            'input_parts': input_parts,
            'remaining_time': transformation.duration
        }
        self.current_jobs.append(job)
        return part_id_counter

    def progress_jobs(self, part_id_counter, final_product_mapping):
        """
        Verringert die verbleibende Dauer der aktiven Jobs.
        Nach Abschluss eines Jobs wird ein neues Teil (vom entsprechenden Typ) erzeugt und in den Output-Puffer gelegt.
        """
        completed_parts = []
        for job in self.current_jobs[:]:
            job['remaining_time'] -= 1
            if job['remaining_time'] <= 0:
                transformation = job['transformation']
                new_part = Part(part_id_counter, transformation.output_type)
                part_id_counter += 1
                self.output_buffer.append(new_part)
                self.current_jobs.remove(job)
                completed_parts.append(new_part)
        return part_id_counter, completed_parts


# -------------------------------
# RUTINE ZUR PROPAGATION DER TEILWERTE
# -------------------------------

def propagate_part_values(transformations):
    """
    Berechnet die Werte aller PartTypes rückwärts (von finalen Produkten zu den Rohteilen).
    Regel:
      - Ein fertiges Teil hat den explizit gesetzten Verkaufswert.
      - Bei einer Transformation erhalten die Input-Teile gemeinsam 50 % des Wertes des Output-Teils,
        gleichverteilt. Falls ein Teil mehrfach verwendet wird, wird stets der maximale Wert übernommen.
    Die Funktion iteriert, bis keine Änderungen mehr auftreten.
    """
    updated = True
    while updated:
        updated = False
        for trans in transformations:
            if trans.output_type.value > 0 and len(trans.input_types) > 0:
                candidate = (0.5 * trans.output_type.value) / len(trans.input_types)
                for inp in trans.input_types:
                    if candidate > inp.value:
                        inp.value = candidate
                        updated = True


# -------------------------------
# GYM ENVIRONMENT IMPLEMENTIERUNG
# -------------------------------

class JobShopEnv(gym.Env):
    """
    Gym‑Environment für ein Job Shop Scheduling Problem.

    Beobachtung:
      Der Beobachtungsvektor besteht aus:
        - einem festen Array (Größe max_buffer) zur Repräsentation des globalen Puffers:
          Jeder Slot wird als Integer kodiert (0: raw, 1: intermediate, 2: finished, 3: empty).
        - Für jede Maschine werden drei Werte geliefert:
          Anzahl der Teile im Input-Puffer, Output-Puffer und aktiver Jobs.

    Aktion:
      Der Agent liefert einen Vektor der Länge max_buffer, wobei für jeden Slot im globalen Puffer
      (beziehungsweise für jedes vorhandene Produkt, bei weniger als max_buffer werden die übrigen
      Slots ignoriert) eine diskrete Entscheidung getroffen wird.

      Die möglichen Entscheidungen pro Produkt sind:
        0: Keine Aktion – das Produkt verbleibt im globalen Puffer
        1 bis (1 + num_machines * num_transformations):
            Kodiert als Kombination (machine_index, transformation_choice)
            wobei machine_index = (action-1) // num_transformations und
                  transformation_choice = (action-1) % num_transformations

    Reward:
      Differenz des „System-Profits“ (Summe der Teilwerte minus Kosten) zwischen aufeinanderfolgenden Zeitschritten.
      Da Transformationen mehrere Zeitschritte in Anspruch nehmen, tritt der Reward verzögert ein.

    Die Episode endet nach einer fix vorgegebenen Anzahl an Schritten.
    """

    def __init__(self, max_steps=50, max_buffer=10, num_machines=2, initial_global=10):
        super(JobShopEnv, self).__init__()
        self.max_steps = max_steps
        self.current_step = 0
        self.max_buffer = max_buffer
        self.initial_global = initial_global

        # --- Definition der PartTypes ---
        # Nur die Kosten werden initial festgelegt; der Wert wird per Propagation berechnet.
        self.raw_part = PartType("raw", cost=10, value=0.0)
        self.intermediate = PartType("intermediate", cost=0, value=0.0)
        self.finished = PartType("finished", cost=0, value=100.0)  # Fertiges Produkt mit vorgegebenem Verkaufswert.

        # Teiltypenliste (für Mapping und Beobachtungsdarstellung)
        self.part_types = [self.raw_part, self.intermediate, self.finished]
        # Kodierung: raw -> 0, intermediate -> 1, finished -> 2; für leere Slots verwenden wir den Wert 3.
        self.empty_marker = 3

        # --- Definition der Transformationen ---
        # Transformation 1: raw -> intermediate (Dauer: 2 Zeitschritte)
        self.transformation_1 = Transformation([self.raw_part], self.intermediate, duration=2)
        # Transformation 2: intermediate -> finished (Dauer: 3 Zeitschritte)
        self.transformation_2 = Transformation([self.intermediate], self.finished, duration=3)
        self.transformations = [self.transformation_1, self.transformation_2]

        # Propagiere die Werte aller PartTypes (von hinten nach vorne)
        propagate_part_values(self.transformations)
        # Nun gilt bspw.: intermediate.value = 50 und raw.value = 25

        # --- Maschinenaufbau ---
        # Erzeuge num_machines Maschinen, die alle dieselben Transformationen (über einen gemeinsamen MachineType) unterstützen.
        self.machine_type = MachineType("machine_type_A", slots=1, transformations=self.transformations)
        self.machines = []
        for i in range(num_machines):
            self.machines.append(Machine(self.machine_type, machine_id=f"machine_{i}"))

        # --- Globaler Puffer ---
        # Produkte, die noch nicht zugeordnet wurden. (In diesem Modell wird nur mit raw Produkten gestartet.)
        self.global_buffer = []
        self.part_id_counter = 0
        self._init_global_buffer(self.initial_global)

        # Mapping, ob ein PartType final ist (bei finalen Produkten wird das Teil verkauft).
        self.final_product_mapping = {
            self.finished.name: True,
            self.raw_part.name: False,
            self.intermediate.name: False
        }

        # --- Beobachtungsraum ---
        # Beobachtung besteht aus zwei Teilen:
        # 1. Globaler Puffer: fester Vektor der Länge max_buffer, Werte in {0,1,2,3} (3: leer)
        # 2. Für jede Maschine: [input_count, output_count, active_jobs]
        obs_dim = self.max_buffer + 3 * len(self.machines)
        # Da alle Werte nicht negativ und klein sind, kann hier ein Box mit int32-Werten verwendet werden.
        self.observation_space = spaces.Box(low=0, high=100, shape=(obs_dim,), dtype=np.int32)

        # --- Aktionsraum ---
        # Pro Produkt im globalen Puffer entscheidet der Agent über eine Aktion.
        # Möglichkeiten pro Produkt: 0 = do nothing, und für jede Kombination aus Maschine und Transformation:
        # Gesamtzahl = 1 + (num_machines * num_transformations)
        self.n_machines = len(self.machines)
        self.n_transformations = len(self.transformations)
        self.n_actions_per_product = 1 + (self.n_machines * self.n_transformations)
        self.action_space = spaces.MultiDiscrete([self.n_actions_per_product] * self.max_buffer)
        # Die Aktion ist ein Vektor der Länge max_buffer mit diskreten Werten.

        self.last_profit = self._calculate_profit()

    def _init_global_buffer(self, n):
        """Füllt den globalen Puffer initial mit n Rohteilen."""
        for _ in range(n):
            part = Part(self.part_id_counter, self.raw_part)
            self.part_id_counter += 1
            self.global_buffer.append(part)

    def reset(self):
        self.current_step = 0
        self.global_buffer = []
        self.part_id_counter = 0
        self._init_global_buffer(self.initial_global)
        for machine in self.machines:
            machine.input_buffer = []
            machine.output_buffer = []
            machine.current_jobs = []
            # Setze die Priorität auf die Standardreihenfolge (wie im MachineType definiert)
            machine.transformation_priority = list(machine.machine_type.transformations)
        self.last_profit = self._calculate_profit()
        return self._get_observation()

    def _get_observation(self):
        """
        Erzeugt einen Beobachtungsvektor bestehend aus:
          1. Globaler Puffer als fester Vektor der Länge max_buffer (produktweise kodiert)
             — Falls weniger als max_buffer Produkte vorhanden sind, werden leere Slots (empty_marker) angehängt.
          2. Für jede Maschine: [Anzahl der Teile im Input-Puffer, Output-Puffer, aktive Jobs]
        """
        # Globaler Puffer: Produkte in der Reihenfolge ihres Eintreffens
        global_obs = []
        for part in self.global_buffer[:self.max_buffer]:
            # Kodierung entsprechend: raw->0, intermediate->1, finished->2
            for i, pt in enumerate(self.part_types):
                if part.type.name == pt.name:
                    global_obs.append(i)
                    break
        # Falls weniger Produkte als max_buffer vorhanden sind, mit empty_marker auffüllen.
        while len(global_obs) < self.max_buffer:
            global_obs.append(self.empty_marker)

        machine_obs = []
        for machine in self.machines:
            machine_obs.append(len(machine.input_buffer))
            machine_obs.append(len(machine.output_buffer))
            machine_obs.append(len(machine.current_jobs))
        observation = np.array(global_obs + machine_obs, dtype=np.int32)
        return observation

    def step(self, action):
        """
        Führt den Zeitschritt aus. Der Agent liefert einen Vektor der Länge max_buffer,
        in dem für jeden Slot im globalen Puffer eine Aktion kodiert ist.
        """
        done = False
        info = {}
        prev_profit = self._calculate_profit()

        # Kopiere den aktuellen globalen Puffer, um die Reihenfolge beizubehalten.
        current_products = self.global_buffer.copy()
        products_to_remove = set()  # IDs der Produkte, die aus dem globalen Puffer entfernt werden

        # Verarbeite die Aktionen für die ersten min(len(global_buffer), max_buffer) Produkte.
        n_to_process = min(len(current_products), self.max_buffer)
        for i in range(n_to_process):
            prod = current_products[i]
            act = action[i]
            # 0 bedeutet "do nothing"
            if act != 0:
                # Kodierung: act-1 entspricht einer Kombination aus (machine_index, transformation_choice)
                decision = act - 1
                machine_index = decision // self.n_transformations
                transformation_choice = decision % self.n_transformations
                chosen_transformation = self.transformations[transformation_choice]
                # Optional: Nur transferieren, wenn das Produkt zum Input dieses Transformationsschritts passt.
                required_type = chosen_transformation.input_types[0]
                if prod.type.name == required_type.name:
                    # Transferiere das Produkt in den Input-Puffer der gewählten Maschine.
                    self.machines[machine_index].input_buffer.append(prod)
                    products_to_remove.add(prod.id)
                # Andernfalls wird nichts unternommen.

        # Aktualisiere den globalen Puffer: Entferne diejenigen, die transferiert wurden.
        new_global = []
        for prod in self.global_buffer:
            if prod.id not in products_to_remove:
                new_global.append(prod)
        self.global_buffer = new_global

        # --- Simulationsfortschritt in den Maschinen ---
        for machine in self.machines:
            # Versuche, einen neuen Job zu starten, falls ein freier Slot vorhanden ist.
            if len(machine.current_jobs) < machine.machine_type.slots:
                for t in machine.transformation_priority:
                    if machine.can_start_transformation(t):
                        self.part_id_counter = machine.start_transformation(t, self.part_id_counter)
                        # Pro Maschine wird maximal ein Job pro Zeitschritt gestartet.
                        break
            # Fortschreiten der aktiven Jobs.
            self.part_id_counter, _ = machine.progress_jobs(self.part_id_counter, self.final_product_mapping)
            # Transfer fertiger Teile:
            # Finalisierte (fertige) Produkte werden verkauft (d.h. nicht in den globalen Puffer aufgenommen),
            # während Zwischenprodukte wieder in den globalen Puffer gelangen.
            while machine.output_buffer:
                part = machine.output_buffer.pop(0)
                if self.final_product_mapping.get(part.type.name, False):
                    # Verkauft – tue hier ggf. Logik zum Erfassen von Umsätzen.
                    pass
                else:
                    self.global_buffer.append(part)

        # --- Reward-Berechnung ---
        current_profit = self._calculate_profit()
        reward = current_profit - prev_profit

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        observation = self._get_observation()
        self.last_profit = current_profit
        return observation, reward, done, info

    def _calculate_profit(self):
        """
        Berechnet den aktuellen Profit als Summe aller (Werte – Kosten) der Teile, die sich
        im globalen Puffer, in den Maschinen (Input-/Output-Puffer) und in aktiven Jobs befinden.
        """
        total_value = 0
        total_cost = 0
        for part in self.global_buffer:
            total_value += part.type.value
            total_cost += part.type.cost
        for machine in self.machines:
            for part in machine.input_buffer:
                total_value += part.type.value
                total_cost += part.type.cost
            for part in machine.output_buffer:
                total_value += part.type.value
                total_cost += part.type.cost
            for job in machine.current_jobs:
                for part in job['input_parts']:
                    total_value += part.type.value
                    total_cost += part.type.cost
        return total_value - total_cost

    def render(self, mode='human'):
        print(f"Step: {self.current_step}")
        # Ausgabe des globalen Puffers als Liste der Codes (0:raw, 1:intermediate, 2:finished, 3:leer)
        global_codes = []
        for part in self.global_buffer:
            for i, pt in enumerate(self.part_types):
                if part.type.name == pt.name:
                    global_codes.append(i)
                    break
        print("Global Buffer (codes):", global_codes)
        for machine in self.machines:
            print(f"Machine {machine.machine_id}:")
            print(f"  Input Buffer: {len(machine.input_buffer)} parts")
            print(f"  Current Jobs: {len(machine.current_jobs)}")
            print(f"  Output Buffer: {len(machine.output_buffer)}")
        print(f"Current Profit: {self._calculate_profit()}\n")

    def close(self):
        pass


# -------------------------------
# Testlauf des Environments
# -------------------------------

if __name__ == "__main__":
    env = JobShopEnv(max_steps=20, max_buffer=10, num_machines=2, initial_global=10)
    obs = env.reset()
    print("Initial Observation:", obs)

    done = False
    total_reward = 0
    while not done:
        # Beispiel: Wähle eine zufällige Aktion für jedes globale Pufferelement.
        action = env.action_space.sample()  # Z.B. ein Vektor der Länge max_buffer
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
    print("Total Reward:", total_reward)
