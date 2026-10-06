import multiprocessing

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from fertigung.core.config import PlantConfig
from fertigung.heuristics import make_policy
from fertigung.rl.env import JobShopEnv


def _episode(task: tuple) -> dict[str, list]:
    config, env_kwargs, expert, epsilon, seed, gamma = task
    env = JobShopEnv(PlantConfig.model_validate(config), **env_kwargs)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    policy = make_policy(expert, seed)
    data = {"obs": [], "actions": [], "masks": [], "rewards": []}
    while True:
        mask = env.action_masks()
        label = env.observer.action_for(policy(env.sim), env.sim)
        data["obs"].append(obs)
        data["actions"].append(label)
        data["masks"].append(mask)
        action = rng.choice(np.flatnonzero(mask)) if rng.random() < epsilon else label
        obs, reward, terminated, truncated, _ = env.step(action)
        data["rewards"].append(reward)
        if terminated or truncated:
            break
    ret, returns = 0.0, []
    for r in reversed(data.pop("rewards")):
        ret = r + gamma * ret
        returns.append(ret)
    data["returns"] = returns[::-1]
    return data


def expert_rollouts(
    configs: list[PlantConfig],
    env_kwargs: dict,
    expert: str,
    episodes: int,
    gamma: float,
    seed: int = 0,
    workers: int = 1,
) -> dict[str, np.ndarray]:
    """Label every visited state with the expert's action. Each episode runs on a random plant from
    `configs`; actions are randomized with rising epsilon across episodes so the data also covers states
    off the expert's path. Episodes run in `workers` processes (useful for the slow lookahead expert)."""
    rng = np.random.default_rng(seed)
    tasks = [
        (
            configs[rng.integers(len(configs))].model_dump(),
            env_kwargs,
            expert,
            0.3 * i / max(episodes - 1, 1),
            seed + i,
            gamma,
        )
        for i in range(episodes)
    ]
    if workers > 1:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            parts = pool.map(_episode, tasks)
    else:
        parts = [_episode(task) for task in tasks]
    return {
        "obs": np.asarray([o for p in parts for o in p["obs"]], dtype=np.float32),
        "actions": np.asarray([a for p in parts for a in p["actions"]]),
        "masks": np.asarray([m for p in parts for m in p["masks"]]),
        "returns": np.asarray([r for p in parts for r in p["returns"]], dtype=np.float32),
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
