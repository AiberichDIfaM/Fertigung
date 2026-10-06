"""Index-based, read-only view of a ShopConfig: parts, transformations, machines, distances, calendar."""

import math
from collections import Counter
from dataclasses import dataclass

from fertigung.shop.config import ShopConfig, minutes_of_day

DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
WEEK = 7 * 24 * 60


@dataclass(frozen=True)
class Transformation:
    name: str
    inputs: Counter
    output: str
    duration: int
    family: str | None
    interruptible: bool


@dataclass(frozen=True)
class Machine:
    name: str
    type: str
    position: tuple[float, float]
    slots: int
    operators: int
    setup_operators: int
    input_buffer: int
    output_buffer: int
    transformations: tuple[int, ...]


class ShopModel:
    def __init__(self, config: ShopConfig):
        self.config = config
        self.parts = [p.name for p in config.part_types]
        self.cost = {p.name: p.cost for p in config.part_types}
        self.price = {p.name: p.price for p in config.part_types}
        self.transformations = [
            Transformation(t.name, Counter(t.inputs), t.output, t.duration, t.setup_family, t.interruptible)
            for t in config.transformations
        ]
        t_index = {t.name: i for i, t in enumerate(self.transformations)}
        types = {mt.name: mt for mt in config.machine_types}
        self.machines = [
            Machine(
                m.name,
                m.type,
                tuple(m.position),
                types[m.type].slots,
                types[m.type].operators,
                types[m.type].setup.operators,
                m.input_buffer,
                m.output_buffer,
                tuple(t_index[t] for t in types[m.type].transformations),
            )
            for m in config.machines
        ]
        self._setup = {mt.name: mt.setup for mt in config.machine_types}

        assigned = {t for m in self.machines for t in m.transformations}
        produced = {self.transformations[t].output for t in assigned}
        consumed = {p for t in assigned for p in self.transformations[t].inputs}
        self.raw = frozenset(p for p in self.parts if p not in produced)
        self.final = frozenset(p for p in self.parts if p in produced and p not in consumed)
        self.producers = {}
        for t in sorted(assigned):
            self.producers.setdefault(self.transformations[t].output, t)
        self.runners = {
            t: [m for m, machine in enumerate(self.machines) if t in machine.transformations]
            for t in assigned
        }

        self.distance_to_final = {p: 0 for p in self.final}
        for _ in self.parts:
            for t in assigned:
                tr = self.transformations[t]
                if tr.output in self.distance_to_final:
                    for p in tr.inputs:
                        d = self.distance_to_final[tr.output] + 1
                        if d < self.distance_to_final.get(p, d + 1):
                            self.distance_to_final[p] = d

        stores = {s.kind: s for s in config.layout.stores}
        self.store_position = {kind: tuple(s.position) for kind, s in stores.items()}
        self.intermediate_capacity = stores["intermediate"].capacity if "intermediate" in stores else 0
        self.has_intermediate = "intermediate" in stores
        self.metric = config.layout.distance
        self.speed = config.transport.speed
        self.handling = config.transport.handling

        # Calendar: workers per minute of the week, pickups as minutes of the week, offset of minute 0.
        self.workers_by_minute = [0] * WEEK
        for shift in config.staff.shifts:
            for day in shift.days:
                base = DAYS.index(day) * 1440
                for minute in range(minutes_of_day(shift.start), minutes_of_day(shift.end)):
                    self.workers_by_minute[base + minute] += shift.workers
        self.pickups = {
            DAYS.index(day) * 1440 + minutes_of_day(p.time)
            for p in config.logistics.pickups
            for day in p.days
        }
        self.offset = DAYS.index(config.start.day) * 1440 + minutes_of_day(config.start.time)
        self.weights = config.objective.resolved()

    def workers(self, t: int) -> int:
        return self.workers_by_minute[(self.offset + t) % WEEK]

    def is_pickup(self, t: int) -> bool:
        return (self.offset + t) % WEEK in self.pickups

    def staffed_stretch(self, t: int, limit: int) -> int:
        """Consecutive minutes from t on with at least one worker, counted up to `limit`."""
        n = 0
        while n < limit and self.workers(t + n) > 0:
            n += 1
        return n

    def setup_time(self, machine: int, current: str | None, target: str | None) -> int:
        if target is None or current == target:
            return 0
        setup = self._setup[self.machines[machine].type]
        if current is None:
            return setup.initial
        for st in setup.times:
            if st.from_family == current and st.to_family == target:
                return st.minutes
        return setup.default

    def position(self, location: tuple) -> tuple[float, float]:
        kind, where = location
        return self.store_position[where] if kind == "store" else self.machines[where].position

    def travel(self, a: tuple[float, float], b: tuple[float, float]) -> int:
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        distance = dx + dy if self.metric == "manhattan" else math.hypot(dx, dy)
        return math.ceil(distance / self.speed)
