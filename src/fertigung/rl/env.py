import math
from functools import cache

import gymnasium as gym
import networkx as nx
import numpy as np

from fertigung.core.config import PlantConfig
from fertigung.core.plant import Plant
from fertigung.core.reward import Reward
from fertigung.core.simulation import Simulation
from fertigung.core.validation import material_costs
from fertigung.heuristics import pull_plan


def default_horizon(config: PlantConfig) -> int:
    deadlines = [o.deadline for o in config.orders]
    return int(max(deadlines) * 1.2) if deadlines else 300


class Observer:
    """Plant-specific encoding: one entry per part type, one action per (machine, transformation) pair.

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

    def to_dispatch(self, action: int, sim: Simulation) -> tuple[int, int] | None:
        return None if action == 0 else self.pairs[action - 1]

    def action_for(self, choice: tuple[int, int] | None, sim: Simulation) -> int:
        return 0 if choice is None else self.pairs.index(choice) + 1


@cache
def _plant_stats(plant: Plant) -> dict:
    cost = material_costs(plant)
    feeds = {
        p: {f for f in plant.final if f == p or f in nx.descendants(plant.graph, p)} for p in plant.part_types
    }
    distances = [d for d in plant.distance_to_final.values() if d is not None]
    return {
        "cost": {p: (0.0 if math.isinf(c) else c) for p, c in cost.items()},
        "max_price": max([plant.price[p] for p in plant.final] + [1.0]),
        "max_distance": max(distances + [1]),
        "max_duration": max(t.duration for t in plant.transformations),
        "feeds": feeds,
        "slots": sum(m.slots for m in plant.machines),
    }


class CandidateObserver:
    """Plant-independent encoding: one feature row per currently possible dispatch plus global features.

    The same weights can score candidates on any plant; slots beyond the current candidates are masked.
    Action 0 advances time; action i > 0 dispatches the i-th candidate in `sim.valid_dispatches()` order.
    """

    CANDIDATE_FEATURES = 16
    GLOBAL_FEATURES = 9

    def __init__(self, max_candidates: int = 64):
        self.max_candidates = max_candidates
        self.n_actions = 1 + max_candidates
        self.size = max_candidates * self.CANDIDATE_FEATURES + self.GLOBAL_FEATURES
        self.horizon = 1

    def bind(self, plant: Plant, horizon: int):
        self.horizon = horizon

    def candidates(self, sim: Simulation) -> list[tuple[int, int]]:
        return sim.valid_dispatches()[: self.max_candidates]

    def observe(self, sim: Simulation) -> np.ndarray:
        plant = sim.plant
        stats = _plant_stats(plant)
        cap = plant.buffer_capacity
        target, need, suggestion = pull_plan(sim)
        counts = sim.buffer_counts()
        in_flight, blocked = {}, 0
        for jobs in sim.jobs:
            for job in jobs:
                out = plant.transformations[job.transformation].output
                in_flight[out] = in_flight.get(out, 0) + 1
                blocked += job.remaining == 0
        reserved = len(sim.buffer) + sum(n for p, n in in_flight.items() if not plant.is_final(p))
        ordered = sum(o.quantity for o in sim.orders) or 1
        open_orders = [o for o in sim.orders if o.open]

        def slack(orders):
            return min((o.deadline - sim.time for o in orders), default=self.horizon) / self.horizon

        features = np.zeros((self.max_candidates, self.CANDIDATE_FEATURES), dtype=np.float32)
        candidates = self.candidates(sim)
        for i, (m, t) in enumerate(candidates):
            tr = plant.transformations[t]
            out = tr.output
            final = plant.is_final(out)
            wip = sum(n for p, n in tr.inputs.items() if not plant.is_raw(p))
            raw_cost = sum(n * plant.cost[p] for p, n in tr.inputs.items() if plant.is_raw(p))
            after = (reserved - wip + (0 if final else 1)) / cap
            relevant = [o for o in open_orders if o.product in stats["feeds"][out]]
            distance = plant.distance_to_final[out]
            features[i] = [
                1.0,
                final,
                (m, t) == suggestion,
                min(need[out], cap) / cap,
                need[out] > 0,
                (distance if distance is not None else stats["max_distance"]) / stats["max_distance"],
                stats["cost"][out] / stats["max_price"],
                wip / cap,
                raw_cost / stats["max_price"],
                tr.duration / stats["max_duration"],
                sim.free_slots(m) / plant.machines[m].slots,
                after,
                (not final) and after > 1,
                (counts[out] + in_flight.get(out, 0)) / cap,
                slack(relevant),
                sum(o.quantity - o.delivered for o in relevant) / ordered,
            ]
        glob = [
            sim.time / self.horizon,
            len(sim.buffer) / cap,
            reserved / cap,
            sum(sim.free_slots(m) for m in range(len(plant.machines))) / stats["slots"],
            blocked / stats["slots"],
            sum(o.quantity - o.delivered for o in open_orders) / ordered,
            slack(open_orders),
            suggestion is None,
            len(candidates) / self.max_candidates,
        ]
        obs = np.concatenate([features.ravel(), np.asarray(glob, dtype=np.float32)])
        return np.clip(obs, -1.0, 1.0)

    def action_mask(self, sim: Simulation) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=bool)
        mask[0] = True
        mask[1 : 1 + len(self.candidates(sim))] = True
        return mask

    def to_dispatch(self, action: int, sim: Simulation) -> tuple[int, int] | None:
        candidates = self.candidates(sim)
        return candidates[action - 1] if 0 < action <= len(candidates) else None

    def action_for(self, choice: tuple[int, int] | None, sim: Simulation) -> int:
        return 0 if choice is None else self.candidates(sim).index(choice) + 1


class JobShopEnv(gym.Env):
    """Dispatching env. Decisions with only one legal action (waiting) are skipped automatically.

    With several plants (transfer architecture) each episode runs on one of them: sampled at random,
    or in order with `cycle=True` (for deterministic evaluation).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: PlantConfig | list[PlantConfig],
        horizon: int | None = None,
        architecture: str = "plant",
        max_candidates: int = 64,
        cycle: bool = False,
    ):
        self.configs = config if isinstance(config, list) else [config]
        if architecture == "plant" and len(self.configs) != 1:
            raise ValueError("the plant architecture trains on exactly one plant")
        self.plants = [Plant(c) for c in self.configs]
        self.fixed_horizon = horizon
        self.cycle = cycle
        self._next = 0
        if architecture == "plant":
            self.observer = Observer(self.plants[0], horizon or default_horizon(self.configs[0]))
        else:
            self.observer = CandidateObserver(max_candidates)
        self._load(0)
        self.action_space = gym.spaces.Discrete(self.observer.n_actions)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (self.observer.size,), np.float32)

    def _load(self, index: int):
        self.config = self.configs[index]
        self.sim = Simulation(self.plants[index])
        self.reward = Reward(self.sim)
        self.horizon = self.fixed_horizon or default_horizon(self.config)
        self.sim.end = self.horizon
        if isinstance(self.observer, CandidateObserver):
            self.observer.bind(self.plants[index], self.horizon)

    def action_masks(self) -> np.ndarray:
        return self.observer.action_mask(self.sim)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if len(self.configs) > 1:
            if self.cycle:
                index, self._next = self._next, (self._next + 1) % len(self.configs)
            else:
                index = int(self.np_random.integers(len(self.configs)))
            self._load(index)
        self.sim.reset()
        self.reward.reset(self.sim)
        self._skip_forced_waits()
        return self.observer.observe(self.sim), {}

    def step(self, action):
        choice = self.observer.to_dispatch(int(action), self.sim)
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
