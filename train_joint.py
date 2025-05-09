import subprocess

# Zuerst Low-Level, dann High-Level
subprocess.run(["python", "train_low_level.py"], check=True)
subprocess.run(["python", "train_high_level.py"], check=True)
print("Joint-Training (Low+High) abgeschlossen.")