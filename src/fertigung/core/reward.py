from collections import Counter

from fertigung.core.config import RewardConfig
from fertigung.core.simulation import Simulation
from fertigung.core.validation import material_costs

COMPONENTS = ("revenue", "material_cost", "holding_cost", "lateness", "idle", "shaping")


class Reward:
    """Weighted reward from changes in the simulation ledger since the previous call.

    Shaping uses the potential phi = WIP valued at raw material cost (buffer plus running jobs),
    so buying material is offset until the product is sold or the WIP is held for long.
    """

    def __init__(self, sim: Simulation, config: RewardConfig | None = None):
        self.config = config or sim.plant.config.reward
        cost = material_costs(sim.plant)
        self._value = {p: c for p, c in cost.items() if c != float("inf")}
        self.reset(sim)

    def reset(self, sim: Simulation):
        self._last = self._totals(sim)
        self._last_potential = self.potential(sim)
        self.totals = Counter({c: 0.0 for c in COMPONENTS})
        self.total = 0.0

    def potential(self, sim: Simulation) -> float:
        wip = [p.type for p in sim.buffer] + [
            sim.plant.transformations[job.transformation].output for jobs in sim.jobs for job in jobs
        ]
        return sum(self._value.get(p, 0.0) for p in wip)

    def __call__(self, sim: Simulation) -> float:
        now = self._totals(sim)
        delta = {k: now[k] - self._last[k] for k in now}
        potential = self.potential(sim)
        c = self.config
        parts = {
            "revenue": c.revenue * delta["revenue"],
            "material_cost": -c.material_cost * delta["material_cost"],
            "holding_cost": -c.holding_cost * delta["holding"],
            "lateness": -c.lateness * delta["late"],
            "idle": -c.idle * delta["idle"],
            "shaping": c.shaping * (c.gamma * potential - self._last_potential),
        }
        parts = {k: v * c.scale for k, v in parts.items()}
        self._last, self._last_potential = now, potential
        self.totals.update(parts)
        reward = sum(parts.values())
        self.total += reward
        self.last_components = parts
        return reward

    @staticmethod
    def _totals(sim: Simulation) -> dict:
        ledger = sim.ledger
        return {
            "revenue": ledger.revenue,
            "material_cost": ledger.material_cost,
            "holding": ledger.holding_part_ticks,
            "late": ledger.late_unit_ticks,
            "idle": ledger.idle_slot_ticks,
        }
