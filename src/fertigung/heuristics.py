import math
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


def _target(sim: Simulation, in_flight: Counter) -> str | None:
    pending = Counter()
    for o in sorted((o for o in sim.orders if o.open), key=lambda o: o.deadline):
        pending[o.product] += o.quantity - o.delivered
        if pending[o.product] > in_flight[o.product]:
            return o.product
    cost = material_costs(sim.plant)
    producible = [p for p in sim.plant.final if cost[p] < math.inf]
    return max(producible, key=lambda p: sim.plant.price[p] - cost[p], default=None)


def pull(sim: Simulation) -> tuple[int, int] | None:
    """Explode the bill of materials of the next final product and only produce net requirements."""
    plant = sim.plant
    producers = _producers(plant)
    in_flight = Counter(plant.transformations[job.transformation].output for jobs in sim.jobs for job in jobs)
    target = _target(sim, in_flight)
    if target is None:
        return None

    available = sim.buffer_counts() + Counter({p: n for p, n in in_flight.items() if not plant.is_final(p)})
    need = Counter()

    def explode(p, n):
        if plant.is_raw(p):
            return
        used = min(available[p], n)
        available[p] -= used
        if n > used:
            need[p] += n - used
            for q, k in plant.transformations[producers[p]].inputs.items():
                explode(q, k * (n - used))

    explode(target, 1)

    reserved = len(sim.buffer) + sum(n for p, n in in_flight.items() if not plant.is_final(p))
    best, best_key = None, None
    for m, t in sim.valid_dispatches():
        tr = plant.transformations[t]
        if need[tr.output] == 0 or producers.get(tr.output) != t:
            continue
        wip_inputs = sum(n for p, n in tr.inputs.items() if not plant.is_raw(p))
        if not plant.is_final(tr.output) and reserved - wip_inputs + 1 > plant.buffer_capacity:
            continue
        key = (plant.distance_to_final[tr.output], -sim.free_slots(m))
        if best_key is None or key < best_key:
            best, best_key = (m, t), key
    return best


POLICIES = {"pull": pull}
