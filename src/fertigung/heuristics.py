import math
import random
from collections import Counter
from functools import cache

from fertigung.core.plant import Plant
from fertigung.core.simulation import Simulation
from fertigung.core.validation import material_costs


@cache
def _producers(plant: Plant) -> dict[str, int]:
    cost = material_costs(plant)
    best = {}
    for i in plant.assigned_transformations:
        t = plant.transformations[i]
        c = sum(n * cost[p] for p, n in t.inputs.items())
        if c < math.inf and (t.output not in best or c < best[t.output][0]):
            best[t.output] = (c, i)
    return {p: i for p, (_, i) in best.items()}


@cache
def _best_margin_product(plant: Plant) -> str | None:
    cost = material_costs(plant)
    producible = [p for p in plant.final if cost[p] < math.inf]
    return max(producible, key=lambda p: plant.price[p] - cost[p], default=None)


@cache
def _headroom(plant: Plant) -> int:
    """Buffer space kept free for the most urgent unit when working ahead on later ones."""
    wip = [sum(n for p, n in t.inputs.items() if not plant.is_raw(p)) for t in plant.transformations]
    return min(max(wip, default=0), plant.buffer_capacity // 2)


def _targets(sim: Simulation, in_flight: Counter, units: int) -> list[str]:
    """The next `units` final products to make: open orders by deadline, then the best-margin product."""
    covered = Counter({p: n for p, n in in_flight.items() if sim.plant.is_final(p)})
    targets = []
    for o in sorted((o for o in sim.orders if o.open), key=lambda o: o.deadline):
        remaining = o.quantity - o.delivered
        used = min(covered[o.product], remaining)
        covered[o.product] -= used
        targets += [o.product] * (remaining - used)
        if len(targets) >= units:
            return targets[:units]
    fallback = _best_margin_product(sim.plant)
    return targets + [fallback] * (units - len(targets)) if fallback else targets


def pull(sim: Simulation) -> tuple[int, int] | None:
    """Explode the bill of materials of the next final product and only produce net requirements."""
    return pull_plan(sim)[2]


def pull_plan(
    sim: Simulation, units: int = 1, headroom: int | None = None
) -> tuple[str | None, Counter, tuple[int, int] | None]:
    """Target product, net requirements per part type and the dispatch the pull heuristic would make.

    With `units > 1` the requirements of the next units are exploded as well; work for later units only
    starts when it leaves room in the buffer for the most urgent one.
    """
    plant = sim.plant
    producers = _producers(plant)
    in_flight = _in_flight(sim)
    targets = _targets(sim, in_flight, units)
    if not targets:
        return None, Counter(), None

    available = sim.buffer_counts() + Counter({p: n for p, n in in_flight.items() if not plant.is_final(p)})
    need = Counter()
    priority = {}

    def explode(p, n, unit):
        if plant.is_raw(p):
            return
        used = min(available[p], n)
        available[p] -= used
        if n > used:
            need[p] += n - used
            priority.setdefault(p, unit)
            for q, k in plant.transformations[producers[p]].inputs.items():
                explode(q, k * (n - used), unit)

    for unit, target in enumerate(targets):
        explode(target, 1, unit)

    reserved = len(sim.buffer) + sum(n for p, n in in_flight.items() if not plant.is_final(p))
    reserve = _headroom(plant) if headroom is None else headroom
    counts = sim.buffer_counts()
    best, best_key = None, None
    # Only the producers of needed parts qualify; (m, position) keeps the tie-break of the machine order.
    for out, n in need.items():
        if n == 0:
            continue
        t = producers[out]
        tr = plant.transformations[t]
        wip_inputs = sum(k for q, k in tr.inputs.items() if not plant.is_raw(q))
        limit = plant.buffer_capacity - (reserve if priority[out] > 0 else 0)
        if not plant.is_final(out) and reserved - wip_inputs + 1 > limit:
            continue
        for m, position in _runners(plant)[t]:
            if not sim.can_dispatch(m, t, counts):
                continue
            key = (priority[out], plant.distance_to_final[out], -sim.free_slots(m), m, position)
            if best_key is None or key < best_key:
                best, best_key = (m, t), key
    return targets[0], need, best


@cache
def _runners(plant: Plant) -> dict[int, list[tuple[int, int]]]:
    """Machines (and the transformation's position on them) that can run each transformation."""
    runners = {}
    for m, machine in enumerate(plant.machines):
        for position, t in enumerate(machine.transformations):
            runners.setdefault(t, []).append((m, position))
    return runners


class PullMulti:
    """Pull heuristic that plans the next `units` final products at once to use idle machines."""

    def __init__(self, units: int = 3, headroom: int | None = None):
        self.units, self.headroom = units, headroom

    def __call__(self, sim: Simulation) -> tuple[int, int] | None:
        return pull_plan(sim, self.units, self.headroom)[2]


class DrumBufferRope:
    """Drum-buffer-rope: keep the bottleneck machine type (the drum) busy and release other work only as fast
    as the drum will need it.

    The drum is the machine type with the highest load per slot for the open orders. Drum operations start
    whenever some open order unit needs them. Feeding work for a later unit starts only once the drum work
    queued for the units before it fits into `rope` times the supply lead time of the drum, so parts arrive
    in time without piling up. The buffer is never overfilled; choices follow deadlines, then closeness to
    the final product.
    """

    def __init__(self, rope: float = 0.5, rope_drum: bool = True, reserve: bool = True):
        self.rope, self.rope_drum, self.reserve = rope, rope_drum, reserve
        self._setup = None

    def _plan(self, sim: Simulation):
        plant = sim.plant
        producers = _producers(plant)
        work = Counter()
        types = {m.type for m in plant.machines}
        slots = Counter()
        for m in plant.machines:
            slots[m.type] += m.slots
        runs_on = {
            t: {m.type for m in plant.machines if t in m.transformations}
            for t in plant.assigned_transformations
        }
        count = Counter()

        def explode(part, n):
            if not plant.is_raw(part):
                count[producers[part]] += n
                for q, k in plant.transformations[producers[part]].inputs.items():
                    explode(q, k * n)

        for o in sim.orders:
            if o.open:
                explode(o.product, o.quantity - o.delivered)
        for t, n in count.items():
            for mt in runs_on[t]:
                work[mt] += n * plant.transformations[t].duration / len(runs_on[t])
        drum = max(types, key=lambda mt: work[mt] / slots[mt])
        drum_ops = {t for t, kinds in runs_on.items() if drum in kinds}

        @cache
        def supply(part):
            """Longest processing chain needed to make `part` from raw materials."""
            if plant.is_raw(part):
                return 0
            t = plant.transformations[producers[part]]
            return t.duration + max((supply(q) for q in t.inputs), default=0)

        lead = max((supply(q) for t in drum_ops for q in plant.transformations[t].inputs), default=0)
        return drum_ops, slots[drum], max(lead, 1) * self.rope

    def __call__(self, sim: Simulation) -> tuple[int, int] | None:
        if self._setup is None or self._setup[0] != (id(sim), sim.plant):
            self._setup = ((id(sim), sim.plant), self._plan(sim))
        drum_ops, drum_slots, rope = self._setup[1]
        plant = sim.plant
        producers = _producers(plant)
        in_flight = _in_flight(sim)
        targets = _targets(sim, in_flight, sum(o.quantity - o.delivered for o in sim.orders if o.open) or 1)
        if not targets:
            return None
        available = sim.buffer_counts() + Counter(
            {p: n for p, n in in_flight.items() if not plant.is_final(p)}
        )
        need = [Counter() for _ in targets]

        def explode(part, n, unit):
            if plant.is_raw(part):
                return
            used = min(available[part], n)
            available[part] -= used
            if n > used:
                need[unit][part] += n - used
                for q, k in plant.transformations[producers[part]].inputs.items():
                    explode(q, k * (n - used), unit)

        for unit, target in enumerate(targets):
            explode(target, 1, unit)

        # Drum time queued ahead of each unit decides whether its feeding work may be released.
        backlog, queued = [], 0.0
        for unit_need in need:
            backlog.append(queued / drum_slots)
            queued += sum(
                n * plant.transformations[producers[p]].duration
                for p, n in unit_need.items()
                if producers[p] in drum_ops
            )

        reserved = len(sim.buffer) + sum(n for p, n in in_flight.items() if not plant.is_final(p))
        counts = sim.buffer_counts()
        best, best_key = None, None
        for unit, unit_need in enumerate(need):
            for out in unit_need:
                t = producers[out]
                if unit > 0 and backlog[unit] > rope and (self.rope_drum or t not in drum_ops):
                    continue
                tr = plant.transformations[t]
                wip = sum(k for q, k in tr.inputs.items() if not plant.is_raw(q))
                limit = plant.buffer_capacity - (_headroom(plant) if self.reserve and unit > 0 else 0)
                if not plant.is_final(out) and reserved - wip + 1 > limit:
                    continue
                for m, position in _runners(plant)[t]:
                    if not sim.can_dispatch(m, t, counts):
                        continue
                    key = (unit, plant.distance_to_final[out], -sim.free_slots(m), m, position)
                    if best_key is None or key < best_key:
                        best, best_key = (m, t), key
        return best


class Lookahead:
    """Rollout algorithm on top of a base heuristic.

    At every decision it tries the base heuristic's choice, the choices of pull and PullMulti, waiting and the
    other dispatches pull would consider on copies of the simulation, lets the base heuristic finish each copy
    until the end of the episode (`until`, else `sim.end`) and takes the one with the best unshaped reward.

    `base="auto"` simulates pull and PullMulti once at the start of an episode and uses the better one: pull
    suits lightly loaded plants, PullMulti (several orders in parallel) busy ones. The simulation is
    deterministic and the base choice is always a candidate (and wins ties), so the result is never worse
    than the base heuristic.
    """

    def __init__(self, until: int | None = None, units: int = 1, max_candidates: int = 6, base: str = "auto"):
        self.until, self.units, self.max_candidates = until, units, max_candidates
        self.base_name = base
        self.multi = PullMulti(3, 0)
        self.dbr = DrumBufferRope()
        self.base = None
        self._episode = None

    def _pick_base(self, sim: Simulation, end: int):
        options = {"pull": pull, "multi": self.multi, "dbr": self.dbr}
        if self.base_name != "auto":
            return options[self.base_name]
        values = {name: self._finish(sim.clone(), policy, end) for name, policy in options.items()}
        return options[max(values, key=lambda name: round(values[name], 9))]

    @staticmethod
    def _finish(rollout: Simulation, policy, end: int) -> float:
        while rollout.time < end:
            while (c := policy(rollout)) is not None:
                rollout.dispatch(*c)
            rollout.advance()
        return _objective(rollout)

    def candidates(self, sim: Simulation) -> list[tuple[int, int] | None]:
        plant = sim.plant
        producers = _producers(plant)
        _, need, _ = pull_plan(sim, self.units)
        ahead = []
        for m, t in sim.valid_dispatches():
            out = plant.transformations[t].output
            if need[out] > 0 and producers.get(out) == t:
                ahead.append((plant.distance_to_final[out], -sim.free_slots(m), (m, t)))
        choices = [self.base(sim), pull(sim), self.multi(sim), self.dbr(sim), None] + [
            c for *_, c in sorted(ahead)
        ]
        return list(dict.fromkeys(choices))[: self.max_candidates]

    def rollout(self, sim: Simulation, choice: tuple[int, int] | None, end: int) -> tuple[tuple, float]:
        """State right after `choice` and the objective at `end` when the base heuristic takes over."""
        rollout = sim.clone()
        if choice is None:
            rollout.advance()
        else:
            rollout.dispatch(*choice)
        return _state(rollout), self._finish(rollout, self.base, end)

    def __call__(self, sim: Simulation) -> tuple[int, int] | None:
        end = self.until or sim.end or _episode_end(sim.plant)
        if self._episode != (id(sim), sim.plant):
            self._episode = (id(sim), sim.plant)
            self.base = self._pick_base(sim, end)
            self._memo = None
        options = self.candidates(sim)
        if len(options) == 1:
            return options[0]
        # The chosen option's rollout is exactly what the base option of the next decision would simulate.
        memo, self._memo = getattr(self, "_memo", None), None
        reuse = memo is not None and memo[0] == _state(sim)
        results = [memo if i == 0 and reuse else self.rollout(sim, c, end) for i, c in enumerate(options)]
        best = max(range(len(options)), key=lambda i: (round(results[i][1], 9), i == 0))
        self._memo = results[best]
        return options[best]


def _state(sim: Simulation) -> tuple:
    return (id(sim.plant), sim.time, sim._next_job, len(sim.buffer), sum(len(jobs) for jobs in sim.jobs))


def _objective(sim: Simulation) -> float:
    """Unshaped reward accumulated so far (same weights as Reward without shaping)."""
    c, ledger = sim.plant.config.reward, sim.ledger
    return c.scale * (
        c.revenue * ledger.revenue
        - c.material_cost * ledger.material_cost
        - c.holding_cost * ledger.holding_part_ticks
        - c.lateness * ledger.late_unit_ticks
        - c.idle * ledger.idle_slot_ticks
    )


@cache
def _episode_end(plant: Plant) -> int:
    deadlines = [o.deadline for o in plant.config.orders]
    return int(max(deadlines) * 1.2) if deadlines else 300


def _in_flight(sim: Simulation) -> Counter:
    return Counter(sim.plant.transformations[job.transformation].output for jobs in sim.jobs for job in jobs)


def fifo(sim: Simulation) -> tuple[int, int] | None:
    """Consume the oldest buffered part first; otherwise start raw-only work for the scarcest part type."""
    plant = sim.plant
    valid = sim.valid_dispatches()
    oldest = {}
    for part in sim.buffer:
        oldest.setdefault(part.type, part.id)

    consuming = [
        (min(oldest[p] for p in plant.transformations[t].inputs if not plant.is_raw(p)), m, t)
        for m, t in valid
        if any(not plant.is_raw(p) for p in plant.transformations[t].inputs)
    ]
    if consuming:
        _, m, t = min(consuming)
        return m, t

    in_flight = _in_flight(sim)
    reserved = len(sim.buffer) + sum(n for p, n in in_flight.items() if not plant.is_final(p))
    counts = sim.buffer_counts() + in_flight
    raw_only = [
        (counts[plant.transformations[t].output], m, t)
        for m, t in valid
        if plant.is_final(plant.transformations[t].output) or reserved < plant.buffer_capacity
    ]
    if raw_only:
        _, m, t = min(raw_only)
        return m, t
    return None


class RandomPolicy:
    """Uniform over valid dispatches; waits with probability `wait`."""

    def __init__(self, seed: int | None = None, wait: float = 0.3):
        self.rng = random.Random(seed)
        self.wait = wait

    def __call__(self, sim: Simulation) -> tuple[int, int] | None:
        valid = sim.valid_dispatches()
        if not valid or self.rng.random() < self.wait:
            return None
        return self.rng.choice(valid)


POLICIES = {
    "pull": lambda seed=None: pull,
    "lookahead": lambda seed=None: Lookahead(),
    "pull_multi": lambda seed=None: PullMulti(3, 0),
    "dbr": lambda seed=None: DrumBufferRope(),
    "fifo": lambda seed=None: fifo,
    "random": lambda seed=None: RandomPolicy(seed),
}


def make_policy(name: str, seed: int | None = None):
    return POLICIES[name](seed)
