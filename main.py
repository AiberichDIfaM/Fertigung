import subprocess
import sys

steps = [
    ("Low-Level Training", "train_low_level.py"),
    ("High-Level Training", "train_high_level.py"),
    #("Joint Training", "train_joint.py"),
    ("Hierarchical Test", "test_hierarchical.py"),
    ("Produktions-Simulation", "production_process_with_rl.py")
]

for name, script in steps:
    print(f"=== {name} ===")
    result = subprocess.run([sys.executable, script], check=False)
    if result.returncode != 0:
        print(f"Fehler in Schritt '{name}', Script {script} mit Code {result.returncode}")
        break
    print()

print("Workflow abgeschlossen.")