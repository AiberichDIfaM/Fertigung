# File: train_low_level.py
# Goal-conditioned Low-Level PPO Training für flexible_jobshop_env
import os
import random
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from flexible_jobshop_env import FlexibleJobShopEnv
from manufacturing_structure import anlage

# Definiere Subgoals: alle PartTypes außer elementaren Rohmaterialien
elementary = {'a1','a2','a4','a5','a6','a8','a0'}
SUBGOALS = [pt.name for pt in anlage.all_part_types if pt.name not in elementary]

MODEL_LL = "lowlevel_ppo_model.zip"
MAX_BUFFER = 10
MAX_STEPS = 50
TOTAL_TIMESTEPS = 200_000

class GoalSamplerEnv(FlexibleJobShopEnv):
    """
    FlexibleJobShopEnv, das bei jedem reset() ein zufälliges Subgoal setzt.
    """
    def __init__(self, anlage, subgoals, max_buffer, max_steps):
        super().__init__(anlage, max_buffer=max_buffer, max_steps=max_steps, goal=None)
        self.subgoals = subgoals

    def reset(self, seed=None, options=None):
        # Wähle zufälliges Subgoal
        self.goal = random.choice(self.subgoals)
        # Basis-Reset aufrufen
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

# Factory-Funktion: erstellt einen maskierbaren, goal-conditioned Env

def make_env():
    env = GoalSamplerEnv(anlage, SUBGOALS, MAX_BUFFER, MAX_STEPS)
    return ActionMasker(env, lambda e: e.get_action_mask())

# Vektor-Umgebung
vec_env = DummyVecEnv([make_env])

# Modell laden oder neu initialisieren
if os.path.exists(MODEL_LL):
    try:
        model = MaskablePPO.load(MODEL_LL, env=vec_env)
    except ValueError:
        model = MaskablePPO("MlpPolicy", vec_env, verbose=1)
else:
    model = MaskablePPO("MlpPolicy", vec_env, verbose=1)

# Training
model.learn(total_timesteps=TOTAL_TIMESTEPS)
model.save(MODEL_LL)
print("Low-Level goal-conditioned Training abgeschlossen.")
