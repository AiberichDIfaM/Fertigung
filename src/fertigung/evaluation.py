from statistics import mean

from fertigung.core.config import PlantConfig
from fertigung.core.plant import Plant
from fertigung.core.reward import Reward
from fertigung.core.simulation import Simulation
from fertigung.heuristics import POLICIES

METRICS = (
    "reward",
    "reward_unshaped",
    "profit",
    "shipped",
    "orders_on_time",
    "total_lateness",
    "utilization",
    "avg_buffer",
)


def run_episode(config: PlantConfig, policy, ticks: int) -> tuple[Simulation, Reward]:
    sim = Simulation(Plant(config))
    reward = Reward(sim)
    sim.run(policy, ticks, reward)
    return sim, reward


def episode_metrics(sim: Simulation, reward: Reward) -> dict:
    kpis = sim.kpis()
    return kpis | {
        "shipped": sum(kpis["shipped"].values()),
        "reward": reward.total,
        "reward_unshaped": reward.total - reward.totals["shaping"],
    }


def evaluate(config: PlantConfig, policy_factory, ticks: int, episodes: int = 1, seed: int = 0) -> dict:
    runs = [episode_metrics(*run_episode(config, policy_factory(seed + i), ticks)) for i in range(episodes)]
    return {k: mean(r[k] for r in runs) for k in METRICS}


def compare(config: PlantConfig, ticks: int, extra: dict | None = None, episodes: int = 5) -> dict[str, dict]:
    """Metrics per policy; deterministic policies run once, the random baseline `episodes` times."""
    factories = dict(POLICIES) | (extra or {})
    return {
        name: evaluate(config, factory, ticks, episodes if name == "random" else 1)
        for name, factory in factories.items()
    }


def format_table(results: dict[str, dict]) -> str:
    header = f"{'policy':<10}" + "".join(f"{m:>16}" for m in METRICS)
    rows = [f"{name:<10}" + "".join(f"{r[m]:>16.2f}" for m in METRICS) for name, r in results.items()]
    return "\n".join([header, *rows])
