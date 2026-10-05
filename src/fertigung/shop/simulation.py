"""Minute-based simulation of the shop-floor model.

The planner only assigns operations to machines (`assign`). Everything else is automatic:
material is allocated to queued jobs as soon as their machine's input buffer has room (raw store, intermediate
store or other machines' output buffers), vehicles carry it there, a job starts when its inputs are present,
a slot is free and staff is available (after a setup if the tooling family changes), finished outputs go to
the output buffer (or block the slot while it is full), final products go to the finished store and leave
with the next truck pickup.

Simplifications: parts leave their source when a vehicle is dispatched; a setup occupies the whole machine;
staff is one pool, non-interruptible jobs keep running even if a shift change leaves too few workers.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from fertigung.shop.model import ShopModel

RAW, INTERMEDIATE, FINISHED = ("store", "raw"), ("store", "intermediate"), ("store", "finished")


def IN(m: int) -> tuple:
    return ("in", m)


def OUT(m: int) -> tuple:
    return ("out", m)


@dataclass
class Job:
    id: int
    transformation: int
    machine: int
    needs: Counter
    incoming: Counter = field(default_factory=Counter)
    present: Counter = field(default_factory=Counter)
    state: str = "queued"  # queued, setup, running, blocked, done
    setup_left: int = 0
    remaining: int = 0
    started: int | None = None
    finished: int | None = None


@dataclass
class Request:
    part: str
    qty: int
    src: tuple
    dst: tuple
    job: int | None
    created: int


@dataclass
class Vehicle:
    kind: str
    capacity: int
    position: tuple[float, float]
    free_at: int = 0


@dataclass
class Order:
    product: str
    quantity: int
    release: int
    deadline: int
    priority: int
    price: float
    delivered: int = 0
    completed_at: int | None = None

    @property
    def open(self) -> bool:
        return self.delivered < self.quantity


@dataclass(frozen=True)
class Event:
    time: int
    kind: str
    machine: str | None = None
    job: int | None = None
    detail: str = ""


@dataclass
class Ledger:
    revenue: float = 0.0
    material_cost: float = 0.0
    holding_part_minutes: int = 0
    late_unit_minutes: int = 0
    setup_minutes: int = 0
    transport_minutes: int = 0
    busy: Counter = field(default_factory=Counter)
    shipped: Counter = field(default_factory=Counter)


class ShopSimulation:
    def __init__(self, model: ShopModel):
        self.model = model
        self.reset()

    def reset(self):
        m = self.model
        self.time = 0
        self.end: int | None = None
        self.stock: defaultdict[tuple, Counter] = defaultdict(Counter)
        self.locked: defaultdict[tuple, Counter] = defaultdict(Counter)
        self.jobs: dict[int, Job] = {}
        self.queues: list[list[int]] = [[] for _ in m.machines]
        self.family: list[str | None] = [None] * len(m.machines)
        self.requests: list[Request] = []
        self.arrivals: list[tuple[int, Request]] = []
        self.vehicles = [
            Vehicle(v.name, v.capacity, m.store_position["raw"])
            for v in m.config.transport.vehicles
            for _ in range(v.count)
        ]
        self.orders = [
            Order(
                o.product,
                o.quantity,
                o.release,
                o.deadline,
                o.priority,
                o.price if o.price is not None else m.price[o.product],
            )
            for o in m.config.orders
        ]
        self.ledger = Ledger()
        self.events: list[Event] = []
        self._next_job = 0

    # ----- planning interface

    def can_assign(self, t: int, m: int) -> bool:
        return t in self.model.machines[m].transformations

    def assign(self, t: int, m: int) -> int:
        if not self.can_assign(t, m):
            raise ValueError(f"{self.model.machines[m].name} cannot run {self.model.transformations[t].name}")
        self._next_job += 1
        job = Job(self._next_job, t, m, Counter(self.model.transformations[t].inputs))
        self.jobs[job.id] = job
        self.queues[m].append(job.id)
        self._log("assign", m, job.id, self.model.transformations[t].name)
        return job.id

    def run(self, policy, minutes: int):
        self.end = self.time + minutes
        while self.time < self.end:
            while (choice := policy(self)) is not None:
                self.assign(*choice)
            self._settle()
            self._minute()

    # ----- queries

    def active_jobs(self, m: int | None = None) -> list[Job]:
        return [j for j in self.jobs.values() if j.state != "done" and (m is None or j.machine == m)]

    def free(self, location: tuple, part: str) -> int:
        return self.stock[location][part] - self.locked[location][part]

    def input_space(self, m: int) -> int:
        incoming = sum(sum(self.jobs[j].incoming.values()) for j in self.queues[m])
        return self.model.machines[m].input_buffer - sum(self.stock[IN(m)].values()) - incoming

    def kpis(self) -> dict:
        m, ledger, w = self.model, self.ledger, self.model.weights
        hours = {
            "holding_part_hours": ledger.holding_part_minutes / 60,
            "late_unit_hours": ledger.late_unit_minutes / 60,
            "setup_hours": ledger.setup_minutes / 60,
            "transport_hours": ledger.transport_minutes / 60,
        }
        objective = (
            w["revenue"] * ledger.revenue
            - w["material_cost"] * ledger.material_cost
            - w["holding_cost"] * hours["holding_part_hours"]
            - w["lateness"] * hours["late_unit_hours"]
            - w["setup"] * hours["setup_hours"]
            - w["transport"] * hours["transport_hours"]
        )
        staffed = max(sum(m.workers(t) > 0 for t in range(self.time)), 1)
        return {
            "time": self.time,
            "objective": objective,
            "revenue": ledger.revenue,
            "material_cost": ledger.material_cost,
            "profit": ledger.revenue - ledger.material_cost,
            "shipped": dict(ledger.shipped),
            "orders_on_time": sum(
                o.completed_at is not None and o.completed_at <= o.deadline for o in self.orders
            ),
            "orders_completed": sum(not o.open for o in self.orders),
            **hours,
            "utilization": {  # running minutes per slot over staffed minutes
                mc.name: round(ledger.busy[i] / (mc.slots * staffed), 3) for i, mc in enumerate(m.machines)
            },
        }

    # ----- automatic behaviour

    def _settle(self):
        self._route()
        self._allocate()
        self._dispatch()
        self._start_jobs()

    def _route(self):
        """Final products to the finished store; free parts out of full output buffers that block a job."""
        m = self.model
        for i in range(len(m.machines)):
            out = OUT(i)
            for part in list(self.stock[out]):
                if part in m.final and self.free(out, part) > 0:
                    self._request(part, self.free(out, part), out, FINISHED, None)
        if not m.has_intermediate:
            return
        room = m.intermediate_capacity or float("inf")
        room -= sum(self.stock[INTERMEDIATE].values())
        room -= sum(r.qty for r in self.requests if r.dst == INTERMEDIATE)
        room -= sum(r.qty for _, r in self.arrivals if r.dst == INTERMEDIATE)
        for i in range(len(m.machines)):
            out = OUT(i)
            blocked = sum(j.state == "blocked" for j in self.active_jobs(i))
            for part in list(self.stock[out]):
                while blocked and room > 0 and self.free(out, part) > 0:
                    self._request(part, 1, out, INTERMEDIATE, None)
                    blocked -= 1
                    room -= 1

    def _allocate(self):
        m = self.model
        for i in range(len(m.machines)):
            space = self.input_space(i)
            for jid in self.queues[i]:
                job = self.jobs[jid]
                for part in list(job.needs):
                    while job.needs[part] > 0 and space > 0:
                        source = self._source(part, i)
                        if source is None:
                            break
                        qty = min(job.needs[part], space)
                        if source != RAW:
                            qty = min(qty, self.free(source, part))
                        else:
                            self.ledger.material_cost += qty * m.cost[part]
                        self._request(part, qty, source, IN(i), jid)
                        job.needs[part] -= qty
                        job.incoming[part] += qty
                        space -= qty
                    if job.needs[part] == 0:
                        del job.needs[part]

    def _source(self, part: str, machine: int) -> tuple | None:
        m = self.model
        if part in m.raw:
            return RAW
        candidates = [
            loc
            for loc in [INTERMEDIATE] + [OUT(i) for i in range(len(m.machines))]
            if self.free(loc, part) > 0
        ]
        if not candidates:
            return None
        target = m.machines[machine].position
        return min(candidates, key=lambda loc: m.travel(m.position(loc), target))

    def _request(self, part, qty, src, dst, job):
        if src != RAW:
            self.locked[src][part] += qty
        self.requests.append(Request(part, qty, src, dst, job, self.time))

    def _dispatch(self):
        m = self.model
        for vehicle in sorted(self.vehicles, key=lambda v: -v.capacity):
            if vehicle.free_at > self.time or not self.requests:
                continue
            self.requests.sort(key=lambda r: (r.job is not None, r.job or 0, r.created))
            first = self.requests[0]
            load, room = [], vehicle.capacity
            for r in list(self.requests):
                if room == 0:
                    break
                if (r.src, r.dst) != (first.src, first.dst):
                    continue
                take = min(r.qty, room)
                if take < r.qty:
                    r.qty -= take
                    r = Request(r.part, take, r.src, r.dst, r.job, r.created)
                else:
                    self.requests.remove(r)
                load.append(r)
                room -= take
            src_pos, dst_pos = m.position(first.src), m.position(first.dst)
            arrival = (
                self.time + m.travel(vehicle.position, src_pos) + m.handling + m.travel(src_pos, dst_pos)
            )
            arrival = max(arrival, self.time + 1)
            for r in load:
                if r.src != RAW:
                    self.stock[r.src][r.part] -= r.qty
                    self.locked[r.src][r.part] -= r.qty
                self.arrivals.append((arrival, r))
            self.ledger.transport_minutes += arrival - self.time
            vehicle.free_at, vehicle.position = arrival, dst_pos
            self._log(
                "transport",
                None,
                first.job,
                f"{sum(r.qty for r in load)} parts {first.src}->{first.dst} until {arrival}",
            )

    def _start_jobs(self):
        m = self.model
        in_use = sum(self._staff_need(j) for j in self.active_jobs() if j.state in ("setup", "running"))
        for i, machine in enumerate(m.machines):
            busy = self.active_jobs(i)
            if any(j.state == "setup" for j in busy):
                continue
            slots = machine.slots - sum(j.state in ("running", "blocked") for j in busy)
            for jid in list(self.queues[i]):
                if slots <= 0:
                    break
                job = self.jobs[jid]
                tr = m.transformations[job.transformation]
                if job.needs or job.incoming or job.present != tr.inputs:
                    continue
                setup = m.setup_time(i, self.family[i], tr.family)
                if setup and any(j.state in ("running", "blocked") for j in busy):
                    continue
                need = machine.setup_operators if setup else machine.operators
                if need and m.workers(self.time) - in_use < need:
                    continue
                if not tr.interruptible and machine.operators:
                    if m.staffed_stretch(self.time, setup + tr.duration) < setup + tr.duration:
                        continue
                self.queues[i].remove(jid)
                self.stock[IN(i)] -= tr.inputs
                job.present.clear()
                job.started = self.time
                in_use += need
                if setup:
                    job.state, job.setup_left = "setup", setup
                    self._log("setup", i, jid, f"{self.family[i]}->{tr.family} {setup} min")
                    break
                job.state, job.remaining = "running", tr.duration
                self._log("start", i, jid, tr.name)
                slots -= 1

    def _staff_need(self, job: Job) -> int:
        machine = self.model.machines[job.machine]
        return machine.setup_operators if job.state == "setup" else machine.operators

    def _minute(self):
        """Advance one minute: work on jobs, account, then handle what happens at the new time."""
        m, ledger = self.model, self.ledger
        staff = m.workers(self.time)
        working = sorted(
            (j for j in self.jobs.values() if j.state in ("setup", "running")),
            key=lambda j: (m.transformations[j.transformation].interruptible, j.state != "setup", j.started),
        )
        set_up = []
        for job in working:
            need = self._staff_need(job)
            interruptible = m.transformations[job.transformation].interruptible or job.state == "setup"
            if need and staff < need and interruptible:
                continue
            staff -= min(need, staff)
            if job.state == "setup":
                job.setup_left -= 1
                ledger.setup_minutes += 1
                if job.setup_left == 0:
                    self.family[job.machine] = m.transformations[job.transformation].family
                    job.state, job.remaining = "running", m.transformations[job.transformation].duration
                    set_up.append(job)
            else:
                job.remaining -= 1
                ledger.busy[job.machine] += 1
                if job.remaining == 0:
                    job.state = "blocked"

        in_transit = sum(r.qty for _, r in self.arrivals)
        ledger.holding_part_minutes += (
            in_transit
            + sum(sum(c.values()) for loc, c in self.stock.items() if loc != RAW)
            + sum(j.state in ("setup", "running", "blocked") for j in self.jobs.values())
        )
        ledger.late_unit_minutes += sum(
            o.quantity - o.delivered for o in self.orders if o.open and self.time >= o.deadline
        )
        self.time += 1
        for job in set_up:
            self._log("start", job.machine, job.id, m.transformations[job.transformation].name)

        for item in [a for a in self.arrivals if a[0] <= self.time]:
            self.arrivals.remove(item)
            _, r = item
            self.stock[r.dst][r.part] += r.qty
            if r.job is not None:
                job = self.jobs[r.job]
                job.incoming[r.part] -= r.qty
                if job.incoming[r.part] == 0:
                    del job.incoming[r.part]
                job.present[r.part] += r.qty
        for job in sorted((j for j in self.jobs.values() if j.state == "blocked"), key=lambda j: j.id):
            out = OUT(job.machine)
            if sum(self.stock[out].values()) < m.machines[job.machine].output_buffer:
                self.stock[out][m.transformations[job.transformation].output] += 1
                job.state, job.finished = "done", self.time
                self._log("finish", job.machine, job.id, m.transformations[job.transformation].name)
        if m.is_pickup(self.time):
            self._ship()

    def _ship(self):
        """Load orders in deadline order. Stock is kept for an earlier order of the same product that cannot
        be completed yet instead of going to a later one (unless partial shipments are allowed)."""
        partial = self.model.config.logistics.partial_shipments
        stock = self.stock[FINISHED]
        held = Counter()
        for o in sorted((o for o in self.orders if o.open), key=lambda o: (o.deadline, -o.priority)):
            missing = o.quantity - o.delivered
            available = stock[o.product] - held[o.product]
            qty = missing if available >= missing else (max(available, 0) if partial else 0)
            if qty < missing and not partial:
                held[o.product] += missing
            if qty == 0:
                continue
            stock[o.product] -= qty
            o.delivered += qty
            self.ledger.revenue += qty * o.price
            self.ledger.shipped[o.product] += qty
            if not o.open:
                o.completed_at = self.time
            self._log("ship", None, None, f"{qty} x {o.product}")

    def _log(self, kind, machine, job, detail=""):
        name = self.model.machines[machine].name if machine is not None else None
        self.events.append(Event(self.time, kind, name, job, detail))
