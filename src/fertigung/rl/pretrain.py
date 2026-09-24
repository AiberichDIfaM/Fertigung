import numpy as np
import torch
from sb3_contrib import MaskablePPO

from fertigung.core.config import PlantConfig
from fertigung.heuristics import make_policy
from fertigung.rl.env import JobShopEnv


def expert_rollouts(
    config: PlantConfig, horizon: int, expert: str, episodes: int, gamma: float, seed: int = 0
) -> dict[str, np.ndarray]:
    """Label every visited state with the expert's action; actions are randomized with rising epsilon
    across episodes so the data also covers states off the expert's path."""
    env = JobShopEnv(config, horizon)
    rng = np.random.default_rng(seed)
    policy = make_policy(expert, seed)
    obs_l, act_l, mask_l, ret_l = [], [], [], []
    for episode in range(episodes):
        epsilon = 0.3 * episode / max(episodes - 1, 1)
        obs, _ = env.reset()
        rewards = []
        while True:
            mask = env.action_masks()
            choice = policy(env.sim)
            label = 0 if choice is None else env.observer.pairs.index(choice) + 1
            obs_l.append(obs)
            act_l.append(label)
            mask_l.append(mask)
            action = rng.choice(np.flatnonzero(mask)) if rng.random() < epsilon else label
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break
        ret, returns = 0.0, []
        for r in reversed(rewards):
            ret = r + gamma * ret
            returns.append(ret)
        ret_l += reversed(returns)
    return {
        "obs": np.asarray(obs_l, dtype=np.float32),
        "actions": np.asarray(act_l),
        "masks": np.asarray(mask_l),
        "returns": np.asarray(ret_l, dtype=np.float32),
    }


def behavior_cloning(
    model: MaskablePPO, data: dict, epochs: int = 30, batch_size: int = 256, lr: float = 1e-3
):
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    n = len(data["actions"])
    tensors = {k: torch.as_tensor(v, device=policy.device) for k, v in data.items()}
    policy.set_training_mode(True)
    for _ in range(epochs):
        for idx in torch.randperm(n).split(batch_size):
            obs = tensors["obs"][idx]
            dist = policy.get_distribution(obs, action_masks=data["masks"][idx.numpy()])
            log_prob = dist.log_prob(tensors["actions"][idx])
            values = policy.predict_values(obs).flatten()
            loss = -log_prob.mean() + 0.5 * torch.nn.functional.mse_loss(values, tensors["returns"][idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    policy.set_training_mode(False)
