# manufacturing_structure.py
import classes

# --- Erstelle PartTypes ---
# Für die "a"-Teile:
a1 = classes.PartType("a1", cost=10)
a2 = classes.PartType("a2", cost=10)
a3 = classes.PartType("a3", cost=0)
a4 = classes.PartType("a4", cost=10)
a5 = classes.PartType("a5", cost=10)
a6 = classes.PartType("a6", cost=10)
a7 = classes.PartType("a7", cost=0)
a8 = classes.PartType("a8", cost=10)
a9 = classes.PartType("a9", cost=0)
a0 = classes.PartType("a0", cost=10)

# Für die "b"-Teile:
b1 = classes.PartType("b1", cost=0)
b2 = classes.PartType("b2", cost=0)
b3 = classes.PartType("b3", cost=0)
b4 = classes.PartType("b4", cost=0)
b5 = classes.PartType("b5", cost=0)
b6 = classes.PartType("b6", cost=0)
b7 = classes.PartType("b7", cost=0)
b8 = classes.PartType("b8", cost=0)
b9 = classes.PartType("b9", cost=0)
b0 = classes.PartType("b0", cost=0)

# Finalprodukt-PartTypes: Hier wird zusätzlich der Verkaufswert (value) explizit gesetzt.
fp1_type = classes.PartType("fp1", cost=0, value=20)
fp2_type = classes.PartType("fp2", cost=0, value=30)

# --- Erstelle Transformationen ---
# Beachte: Als Eingabeparameter für Transformationen werden Listen von PartTypes benötigt.
tr1 = classes.Transformation([a1, a2], a3, 3)
tr2 = classes.Transformation([a4, a5, a6], a7, 6)
tr3 = classes.Transformation([a8], a9, 2)           # vormals: transformation(a8,a9,2)
tr4 = classes.Transformation([a8, a0], b1, 2)
tr5 = classes.Transformation([a3, a0], b2, 3)
tr6 = classes.Transformation([b2, a9], b3, 5)
tr7 = classes.Transformation([b2, a5], b4, 5)
tr8 = classes.Transformation([a2, a9], b5, 5)
tr9 = classes.Transformation([b2, a5], b6, 5)
tr10 = classes.Transformation([b3, b5, b1], b7, 5)
tr11 = classes.Transformation([b1, a5, a7], b8, 5)
tr12 = classes.Transformation([b8], b9, 5)
tr13 = classes.Transformation([b7], b0, 5)

# Transformationen zu finalen Produkten:
ftran1 = classes.Transformation([b4, b5, b6, b7], fp1_type, 10)
ftran2 = classes.Transformation([b1, b2, b3, b9, b8, b0], fp2_type, 15)

# --- Erstelle finale Produkte (optional, falls benötigt) ---
finalproduct1 = classes.Product("fp1", fp1_type, 20)
finalproduct2 = classes.Product("fp2", fp2_type, 30)

# --- Erstelle MachineTypes ---
# Jede Maschine erhält eine maximale Slot-Anzahl und unterstützt eine bestimmte Liste von Transformationen.
m1_type = classes.MachineType("m1", 4, [tr1, tr6])
m2_type = classes.MachineType("m2", 3, [tr2, tr9])
m3_type = classes.MachineType("m3", 2, [tr2, tr5, tr11])
m4_type = classes.MachineType("m4", 6, [tr12, tr3, tr7])
m5_type = classes.MachineType("m5", 5, [tr4, tr8, ftran1])
m6_type = classes.MachineType("m6", 1, [tr4, ftran2])

# --- Erstelle Maschinen ---
# Achte darauf, dass Maschinen eindeutige Bezeichner besitzen.
m1 = classes.Machine(m1_type, "m1")
m2 = classes.Machine(m2_type, "m2")
m3 = classes.Machine(m3_type, "m3")
m4 = classes.Machine(m4_type, "m4")
m5 = classes.Machine(m5_type, "m5")
m6 = classes.Machine(m6_type, "m6")
m7 = classes.Machine(m1_type, "m7")
m8 = classes.Machine(m3_type, "m8")
m9 = classes.Machine(m4_type, "m9")
m0 = classes.Machine(m6_type, "m0")

# Alle Maschinen in ein Array zusammenfassen.
machine_array = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m0]

# Verbinde jede Maschine mit allen anderen (jedoch nicht mit sich selbst).
for m in machine_array:
    m.connected_machines = [x for x in machine_array if x != m]

# --- Sammle alle PartTypes ---
all_part_types = [
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a0,
    b1, b2, b3, b4, b5, b6, b7, b8, b9, b0,
    fp1_type, fp2_type
]

# --- Erstelle ggf. globale Input-Teile (hier als leere Liste, falls nicht benötigt) ---
input_parts = []

# --- Erstelle die Fertigungsanlage ---
# Die Anlage wird über eine Liste von Maschinen, den Startzeitpunkt und die Inputteile sowie alle PartTypes initialisiert.
anlage = classes.Anlage(machine_array, 0, input_parts, all_part_types)

if __name__ == "__main__":
    print("Fertigungsstruktur erfolgreich erstellt.")
