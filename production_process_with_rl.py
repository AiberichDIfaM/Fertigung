# File: production_process_with_rl.py
# Simulation des hierarchischen Produktionsprozesses mit High-Level & Low-Level Agent
import os
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from hierarchical_env import HighLevelEnv
from manufacturing_structure import anlage

# Modell-Pfade
MODEL_LL = "lowlevel_ppo_model.zip"
MODEL_HL = "highlevel_ppo_model.zip"
LOG_FILE = "production_rl_event_log.txt"

# Sicherstellen, dass beide Modelle existieren
if not os.path.exists(MODEL_LL) or not os.path.exists(MODEL_HL):
    print("Fehlende Modelle. Bitte zuerst Low- und High-Level trainieren.")
    exit(1)

# Modelle laden
model_ll = MaskablePPO.load(MODEL_LL)
model_hl = MaskablePPO.load(MODEL_HL)

# Subgoals definieren (außer elementare Rohteile)
elementary = {'a1','a2','a4','a5','a6','a8','a0'}
SUBGOALS = [pt.name for pt in anlage.all_part_types if pt.name not in elementary]

# High-Level Env erstellen und maskieren
def make_hl_env():
    env = HighLevelEnv(anlage, SUBGOALS)
    return ActionMasker(env, lambda e: e.reset()[1]['action_mask'])

vec_env = DummyVecEnv([make_hl_env])

# Hilfsfunktion zum Loggen des Puffer- und Maschinenstatus

def log_status():
    lines = []
    # Globaler Puffer
    buf = anlage.global_buffer
    lines.append("Global Buffer: " + ", ".join(f"{p.id}:{p.type.name}" for p in buf))
    # Maschinenstatus
    for m in anlage.machines:
        inp = ", ".join(f"{p.id}:{p.type.name}" for p in m.input_buffer)
        out = ", ".join(f"{p.id}:{p.type.name}" for p in m.output_buffer)
        jobs = "; ".join("[" + ", ".join(f"{p.id}:{p.type.name}" for p in job['input_parts']) + "]" for job in m.current_jobs)
        lines.append(f"Machine {m.machine_id}: Input [{inp}] | Output [{out}] | Jobs [{jobs}]")
    return "\n".join(lines)

# Simulation starten
obs = vec_env.reset()
logs = ["=== Hierarchical Produktionssimulation mit RL gestartet ===",
        f"Initial HL-Observation: {obs}",
        "Initialer Anlagenstatus:",
        log_status()]

step = 0
terminated = False
truncated = False

while not (terminated or truncated):
    # High-Level Agent wählt Subziel
    action_hl, _ = model_hl.predict(obs, deterministic=True)
    # Ausführen
    ret = vec_env.step(action_hl)
    # Entpacken je nach Gym-Version
    if len(ret) == 5:
        obs, reward_hl, terminated, truncated, info = ret
    else:
        obs, reward_hl, terminated, info = ret
        truncated = False

    # Log-Eintrag
    logs.append(f"\n--- HL Schritt {step} ---")
    logs.append(f"Subgoal-Action: {action_hl} ({'noop' if action_hl==0 else SUBGOALS[action_hl-1]})")
    logs.append(f"HL-Reward: {reward_hl}")
    logs.append("Anlagenstatus nach Schritt:")
    logs.append(log_status())
    step += 1

logs.append("=== Hierarchical Produktionssimulation beendet ===")

# Logfile schreiben
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(logs))

print(f"Simulation abgeschlossen. Log in '{LOG_FILE}' gespeichert.")

