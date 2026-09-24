from collections import Counter
from dataclasses import dataclass, field

from fertigung.core.plant import Plant


@dataclass
class Part:
    id: int
    type: str


@dataclass
class Job:
    transformation: int
    inputs: list[Part]
    remaining: int
    blocked_since: int | None = None


@dataclass
class Order:
    product: str
    quantity: int
    deadline: int
    price: float
    delivered: int = 0
    completed_at: int | None = None

    @property
    def open(self) -> bool:
        return self.delivered < self.quantity

    def lateness(self, now: int) -> int:
        end = self.completed_at if self.completed_at is not None else now
        return max(0, end - self.deadline)


@dataclass(frozen=True)
class Event:
    time: int
    kind: str
    machine: str | None = None
    transformation: str | None = None
    part_id: int | None = None
    part_type: str | None = None
    order: int | None = None
    amount: float = 0.0


@dataclass
class Ledger:
    revenue: float = 0.0
    material_cost: float = 0.0
    busy_slot_ticks: int = 0
    blocked_slot_ticks: int = 0
    shipped: Counter = field(default_factory=Counter)


class Simulation:
    """Discrete-time job shop.

    `dispatch` starts a transformation on a machine immediately, taking intermediate inputs from the
    shared WIP buffer and buying raw inputs. `advance` moves time forward by one tick. A finished job
    places its output in the buffer, or ships it if it is a final product; if the buffer is full the
    job stays blocked in its slot.
    """

    def __init__(self, plant: Plant):
        self.plant = plant
        self.reset()

    def reset(self):
        self.time = 0
        self.buffer: list[Part] = []
        self.jobs: list[list[Job]] = [[] for _ in self.plant.machines]
        self.orders = [
            Order(
                o.product,
                o.quantity,
                o.deadline,
                o.price if o.price is not None else self.plant.price[o.product],
            )
            for o in self.plant.config.orders
        ]
        self.ledger = Ledger()
        self.events: list[Event] = []
        self._next_id = 0

    def buffer_counts(self) -> Counter:
        return Counter(p.type for p in self.buffer)

    def free_slots(self, m: int) -> int:
        return self.plant.machines[m].slots - len(self.jobs[m])

    def can_dispatch(self, m: int, t: int, counts: Counter | None = None) -> bool:
        if t not in self.plant.machines[m].transformations or self.free_slots(m) <= 0:
            return False
        counts = counts if counts is not None else self.buffer_counts()
        return all(
            self.plant.is_raw(p) or counts[p] >= n for p, n in self.plant.transformations[t].inputs.items()
        )

    def valid_dispatches(self) -> list[tuple[int, int]]:
        counts = self.buffer_counts()
        return [
            (m, t)
            for m, machine in enumerate(self.plant.machines)
            for t in machine.transformations
            if self.can_dispatch(m, t, counts)
        ]

    def dispatch(self, m: int, t: int):
        if not self.can_dispatch(m, t):
            raise ValueError(
                f"cannot dispatch {self.plant.transformations[t].name} on {self.plant.machines[m].name}"
            )
        transformation = self.plant.transformations[t]
        needed = Counter({p: n for p, n in transformation.inputs.items() if not self.plant.is_raw(p)})
        inputs, rest = [], []
        for part in self.buffer:
            if needed[part.type] > 0:
                needed[part.type] -= 1
                inputs.append(part)
            else:
                rest.append(part)
        self.buffer = rest
        for p, n in transformation.inputs.items():
            if self.plant.is_raw(p):
                inputs += [self._new_part(p) for _ in range(n)]
                self.ledger.material_cost += n * self.plant.cost[p]

        self.jobs[m].append(Job(t, inputs, transformation.duration))
        self._log(
            "dispatch", m, t, amount=sum(self.plant.cost[p.type] for p in inputs if self.plant.is_raw(p.type))
        )

    def advance(self):
        self.time += 1
        finished = []
        for m, jobs in enumerate(self.jobs):
            for job in jobs:
                if job.remaining > 0:
                    self.ledger.busy_slot_ticks += 1
                    job.remaining -= 1
                else:
                    self.ledger.blocked_slot_ticks += 1
                if job.remaining == 0:
                    finished.append(
                        (job.blocked_since if job.blocked_since is not None else self.time, m, job)
                    )

        for _, m, job in sorted(finished, key=lambda f: (f[0], f[1])):
            output = self.plant.transformations[job.transformation].output
            if self.plant.is_final(output):
                self._ship(m, job, self._new_part(output))
            elif len(self.buffer) < self.plant.buffer_capacity:
                part = self._new_part(output)
                self.buffer.append(part)
                self.jobs[m].remove(job)
                self._log("complete", m, job.transformation, part)
            elif job.blocked_since is None:
                job.blocked_since = self.time
                self._log("blocked", m, job.transformation)

    def run(self, policy, ticks: int):
        for _ in range(ticks):
            while (choice := policy(self)) is not None:
                self.dispatch(*choice)
            self.advance()

    def kpis(self) -> dict:
        slot_ticks = sum(m.slots for m in self.plant.machines) * max(self.time, 1)
        return {
            "time": self.time,
            "revenue": self.ledger.revenue,
            "material_cost": self.ledger.material_cost,
            "profit": self.ledger.revenue - self.ledger.material_cost,
            "shipped": dict(self.ledger.shipped),
            "wip": len(self.buffer) + sum(len(jobs) for jobs in self.jobs),
            "utilization": self.ledger.busy_slot_ticks / slot_ticks,
            "blocked_ratio": self.ledger.blocked_slot_ticks / slot_ticks,
            "orders_completed": sum(not o.open for o in self.orders),
            "orders_on_time": sum(not o.open and o.lateness(self.time) == 0 for o in self.orders),
            "total_lateness": sum(o.lateness(self.time) for o in self.orders),
        }

    def _ship(self, m: int, job: Job, part: Part):
        self.jobs[m].remove(job)
        candidates = [i for i, o in enumerate(self.orders) if o.open and o.product == part.type]
        order = min(candidates, key=lambda i: self.orders[i].deadline, default=None)
        if order is None:
            amount = self.plant.price[part.type]
        else:
            o = self.orders[order]
            amount = o.price
            o.delivered += 1
            if not o.open:
                o.completed_at = self.time
        self.ledger.revenue += amount
        self.ledger.shipped[part.type] += 1
        self._log("ship", m, job.transformation, part, order=order, amount=amount)

    def _new_part(self, part_type: str) -> Part:
        self._next_id += 1
        return Part(self._next_id, part_type)

    def _log(self, kind, m, t, part=None, order=None, amount=0.0):
        self.events.append(
            Event(
                self.time,
                kind,
                self.plant.machines[m].name,
                self.plant.transformations[t].name,
                part.id if part else None,
                part.type if part else self.plant.transformations[t].output,
                order,
                amount,
            )
        )
