# File: ppo_hierarchical_training.py
import os
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from hierarchical_env import HighLevelEnv
from manufacturing_structure import anlage

SUBGOALS = [pt.name for pt in anlage.all_part_types if pt.name not in ["a1","a2","a4","a5","a6","a8","a0"]]
MODEL_H_PATH = "hl_ppo_model.zip"

def make_hl_env():
    env = HighLevelEnv(anlage, SUBGOALS)
    # direkt aus env zurückliefern
    return ActionMasker(env, lambda e: e.reset()[1]["action_mask"])

vec_h = DummyVecEnv([make_hl_env])
if os.path.exists(MODEL_H_PATH):
    try:
        m_h = MaskablePPO.load(MODEL_H_PATH, env=vec_h)
    except Exception:
        m_h = MaskablePPO('MlpPolicy', vec_h, verbose=1)
else:
    m_h = MaskablePPO('MlpPolicy', vec_h, verbose=1)

m_h.learn(total_timesteps=200000)
m_h.save(MODEL_H_PATH)
