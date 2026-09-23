import subprocess
import sys

STEPS = [
    ("Low-Level Training", "fertigung.rl.train_low_level"),
    ("High-Level Training", "fertigung.rl.train_high_level"),
    ("Hierarchical Test", "fertigung.rl.evaluate"),
    ("Produktions-Simulation", "fertigung.rl.simulate"),
]


def main():
    for name, module in STEPS:
        print(f"=== {name} ===")
        result = subprocess.run([sys.executable, "-m", module], check=False)
        if result.returncode != 0:
            print(f"Fehler in Schritt '{name}', Modul {module} mit Code {result.returncode}")
            sys.exit(result.returncode)
        print()
    print("Workflow abgeschlossen.")


if __name__ == "__main__":
    main()
