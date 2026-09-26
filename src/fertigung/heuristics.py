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


class Lookahead:
    """Rollout algorithm on top of the pull heuristic.

    Rollouts end at `until`, else at the end of the running episode (`sim.end`).
    At every decision it tries the pull choice, waiting and work-ahead candidates (from planning `units`
    products at once) on copies of the simulation, follows each with the pull heuristic until the end of the
    episode and takes the one with the best unshaped reward. Because the simulation is deterministic and the
    pull choice is always among the candidates (and wins ties), the result is never worse than pull itself.
    """

    def __init__(self, until: int | None = None, units: int = 1, max_candidates: int = 4):
        self.until, self.units, self.max_candidates = until, units, max_candidates

    def candidates(self, sim: Simulation) -> list[tuple[int, int] | None]:
        plant = sim.plant
        producers = _producers(plant)
        _, need, _ = pull_plan(sim, self.units)
        ahead = []
        for m, t in sim.valid_dispatches():
            out = plant.transformations[t].output
            if need[out] > 0 and producers.get(out) == t:
                ahead.append((plant.distance_to_final[out], -sim.free_slots(m), (m, t)))
        choices = [pull(sim), None] + [c for *_, c in sorted(ahead)]
        return list(dict.fromkeys(choices))[: self.max_candidates]

    def rollout(self, sim: Simulation, choice: tuple[int, int] | None, end: int) -> tuple[tuple, float]:
        """State right after `choice` and the objective at `end` when pull takes over from there."""
        rollout = sim.clone()
        if choice is None:
            rollout.advance()
        else:
            rollout.dispatch(*choice)
        after = _state(rollout)
        while rollout.time < end:
            while (c := pull(rollout)) is not None:
                rollout.dispatch(*c)
            rollout.advance()
        return after, _objective(rollout)

    def __call__(self, sim: Simulation) -> tuple[int, int] | None:
        options = self.candidates(sim)
        if len(options) == 1:
            return options[0]
        end = self.until or sim.end or _episode_end(sim.plant)
        # The chosen option's rollout is exactly what the pull option of the next decision would simulate.
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
    "fifo": lambda seed=None: fifo,
    "random": lambda seed=None: RandomPolicy(seed),
}


def make_policy(name: str, seed: int | None = None):
    return POLICIES[name](seed)
