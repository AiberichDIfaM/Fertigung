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


def compare(
    config: PlantConfig,
    ticks: int,
    extra: dict | None = None,
    episodes: int = 5,
    policies: list[str] | None = None,
) -> dict[str, dict]:
    """Metrics per policy; deterministic policies run once, the random baseline `episodes` times."""
    factories = {name: POLICIES[name] for name in (policies or POLICIES)} | (extra or {})
    return {
        name: evaluate(config, factory, ticks, episodes if name == "random" else 1)
        for name, factory in factories.items()
    }


def format_table(results: dict[str, dict]) -> str:
    header = f"{'policy':<10}" + "".join(f"{m:>16}" for m in METRICS)
    rows = [f"{name:<10}" + "".join(f"{r[m]:>16.2f}" for m in METRICS) for name, r in results.items()]
    return "\n".join([header, *rows])


def benchmark(configs: list[PlantConfig], factories: dict, horizon_for) -> dict[str, dict]:
    """Mean metrics per policy over several plants, and on how many plants each policy beats or trails pull
    on the unshaped reward (differences below 1% of pull's absolute value count as ties)."""
    per_plant = [
        {
            name: evaluate(c, lambda seed, f=f, c=c: f(c, seed), horizon_for(c))
            for name, f in factories.items()
        }
        for c in configs
    ]

    def compare_to_pull(p, name):
        diff = p[name]["reward_unshaped"] - p["pull"]["reward_unshaped"]
        tolerance = 0.01 * max(abs(p["pull"]["reward_unshaped"]), 1e-6)
        return (diff > tolerance) - (diff < -tolerance)

    return {
        name: {k: mean(p[name][k] for p in per_plant) for k in METRICS}
        | {
            "better_than_pull": sum(compare_to_pull(p, name) > 0 for p in per_plant),
            "worse_than_pull": sum(compare_to_pull(p, name) < 0 for p in per_plant),
        }
        for name in factories
    }
