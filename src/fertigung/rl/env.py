import gymnasium as gym
import numpy as np

from fertigung.core.config import PlantConfig
from fertigung.core.plant import Plant
from fertigung.core.reward import Reward
from fertigung.core.simulation import Simulation


def default_horizon(config: PlantConfig) -> int:
    deadlines = [o.deadline for o in config.orders]
    return int(max(deadlines) * 1.2) if deadlines else 300


class Observer:
    """Encodes a simulation state as a fixed-size vector in [-1, 1] and maps actions to dispatches.

    Action 0 advances time by one tick; action i > 0 dispatches `pairs[i - 1]`.
    """

    def __init__(self, plant: Plant, horizon: int):
        self.plant = plant
        self.horizon = horizon
        self.pairs = [(m, t) for m, machine in enumerate(plant.machines) for t in machine.transformations]
        self.stocked = [p for p in plant.part_types if not plant.is_raw(p) and not plant.is_final(p)]
        self.produced = [p for p in plant.part_types if not plant.is_raw(p)]
        self.ordered = {
            p: sum(o.quantity for o in plant.config.orders if o.product == p) for p in plant.final
        }
        self.total_slots = sum(m.slots for m in plant.machines)
        self.n_actions = 1 + len(self.pairs)
        self.size = (
            len(self.stocked)
            + len(self.produced)
            + len(self.pairs)
            + 2 * len(plant.machines)
            + 2 * len(plant.final)
            + 2
        )

    def observe(self, sim: Simulation) -> np.ndarray:
        plant = self.plant
        counts = sim.buffer_counts()
        in_flight, running, blocked = {}, {}, [0] * len(plant.machines)
        for m, jobs in enumerate(sim.jobs):
            for job in jobs:
                out = plant.transformations[job.transformation].output
                in_flight[out] = in_flight.get(out, 0) + 1
                if job.remaining > 0:
                    running[(m, job.transformation)] = running.get((m, job.transformation), 0) + 1
                else:
                    blocked[m] += 1

        obs = [counts[p] / plant.buffer_capacity for p in self.stocked]
        obs += [in_flight.get(p, 0) / self.total_slots for p in self.produced]
        obs += [running.get(pair, 0) / plant.machines[pair[0]].slots for pair in self.pairs]
        for m, machine in enumerate(plant.machines):
            obs += [sim.free_slots(m) / machine.slots, blocked[m] / machine.slots]
        for p in plant.final:
            open_orders = [o for o in sim.orders if o.open and o.product == p]
            outstanding = sum(o.quantity - o.delivered for o in open_orders)
            slack = min((o.deadline for o in open_orders), default=sim.time + self.horizon) - sim.time
            obs += [outstanding / self.ordered[p] if self.ordered[p] else 0.0, slack / self.horizon]
        obs += [sim.time / self.horizon, len(sim.buffer) / plant.buffer_capacity]
        return np.clip(np.asarray(obs, dtype=np.float32), -1.0, 1.0)

    def action_mask(self, sim: Simulation) -> np.ndarray:
        counts = sim.buffer_counts()
        return np.array([True] + [sim.can_dispatch(m, t, counts) for m, t in self.pairs])

    def to_dispatch(self, action: int) -> tuple[int, int] | None:
        return None if action == 0 else self.pairs[action - 1]


class JobShopEnv(gym.Env):
    """Flat dispatching env. Decisions with only one legal action (waiting) are skipped automatically."""

    metadata = {"render_modes": []}

    def __init__(self, config: PlantConfig, horizon: int | None = None):
        self.config = config
        self.sim = Simulation(Plant(config))
        self.observer = Observer(self.sim.plant, horizon or default_horizon(config))
        self.reward = Reward(self.sim)
        self.action_space = gym.spaces.Discrete(self.observer.n_actions)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (self.observer.size,), np.float32)

    @property
    def horizon(self) -> int:
        return self.observer.horizon

    def action_masks(self) -> np.ndarray:
        return self.observer.action_mask(self.sim)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self.reward.reset(self.sim)
        self._skip_forced_waits()
        return self.observer.observe(self.sim), {}

    def step(self, action):
        choice = self.observer.to_dispatch(int(action))
        if choice is not None and self.sim.can_dispatch(*choice):
            self.sim.dispatch(*choice)
        else:
            self.sim.advance()
        total = self.reward(self.sim) + self._skip_forced_waits()
        truncated = self.sim.time >= self.horizon
        return self.observer.observe(self.sim), total, False, truncated, {}

    def _skip_forced_waits(self) -> float:
        total = 0.0
        while self.sim.time < self.horizon and not self.action_masks()[1:].any():
            self.sim.advance()
            total += self.reward(self.sim)
        return total
