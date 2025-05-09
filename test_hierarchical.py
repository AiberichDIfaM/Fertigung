import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from hierarchical_env import HighLevelEnv
from manufacturing_structure import anlage

# Lade Modelle
model_ll = MaskablePPO.load("archive/lowlevel_ppo_model.zip")
model_hl = MaskablePPO.load("archive/highlevel_ppo_model.zip")

# High-Level Env
SUBGOALS = [pt.name for pt in anlage.all_part_types if pt.name not in ['a1','a2','a4','a5','a6','a8','a0']]
env_hl = HighLevelEnv(anlage, SUBGOALS)
env_hl = ActionMasker(env_hl, lambda e: e.reset()[1]['action_mask'])

# Test-Episode
episode_reward = 0.0
obs_hl, info_hl = env_hl.reset()
done = False
while not done:
    action_hl, _ = model_hl.predict(obs_hl, deterministic=True)
    obs_hl, r, done, _, _ = env_hl.step(action_hl)
    episode_reward += r

print(f"Hierarchical Test Reward: {episode_reward}")