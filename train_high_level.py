import os
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from hierarchical_env import HighLevelEnv
from manufacturing_structure import anlage

MODEL_HL = "highlevel_ppo_model.zip"
TOTAL_TIMESTEPS = 100_000
MAX_HL_STEPS = 10
# Definiere Subziele (alle PartTypes außer Rohmaterialien)
elementary = {'a1','a2','a4','a5','a6','a8','a0'}
SUBGOALS = [pt.name for pt in anlage.all_part_types if pt.name not in elementary]

# Factory

def make_hl_env():
    env = HighLevelEnv(anlage, SUBGOALS, max_steps=MAX_HL_STEPS)
    return ActionMasker(env, lambda e: e.reset()[1]["action_mask"] )

vec_hl = DummyVecEnv([make_hl_env])

# Modell laden oder initialisieren
if os.path.exists(MODEL_HL):
    try:
        model_hl = MaskablePPO.load(MODEL_HL, env=vec_hl)
    except ValueError:
        model_hl = MaskablePPO('MlpPolicy', vec_hl, verbose=1)
else:
    model_hl = MaskablePPO('MlpPolicy', vec_hl, verbose=1)

# Training
model_hl.learn(total_timesteps=TOTAL_TIMESTEPS)
model_hl.save(MODEL_HL)
print("High-Level Training abgeschlossen.")