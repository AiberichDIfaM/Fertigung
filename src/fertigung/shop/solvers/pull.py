from collections import Counter

from fertigung.shop.model import ShopModel
from fertigung.shop.plan import Assignment, ShopState


class ShopPull:
    """Demand-driven assignment: explode the open orders (by deadline) against everything that exists or is
    already planned, and assign the most urgent missing operation closest to the final product. Prefers
    machines already set up for its tooling family, then short queues; at most `queue_limit` jobs wait per
    machine. When the room for intermediate parts (intermediate store and output buffers) runs low, only the
    most urgent order unit and missing inputs of planned jobs may produce intermediate parts; otherwise
    work for later orders fills every buffer and the shop deadlocks. The default reserve is one part per
    possible queued job.

    `batch` lets work for the tooling family a machine will have next jump ahead by that many order units,
    which saves setups (on random order books better than no batching on 17 of 24, worse on 5).
    `queue_minutes` limits the queue by processing minutes per slot instead of jobs."""

    def __init__(
        self,
        queue_limit: int = 2,
        reserve: int | None = None,
        queue_minutes: float | None = None,
        batch: int = 20,
    ):
        self.queue_limit = queue_limit
        self.reserve = reserve
        self.queue_minutes = queue_minutes
        self.batch = batch

    def __call__(self, state: ShopState) -> Assignment | None:
        m = state.model
        available = state.available()
        need, priority = Counter(), {}

        def explode(part, n, unit, per_unit):
            """n parts for consecutive order units starting at `unit`, `per_unit` parts each."""
            if part in m.raw or part not in m.producers:
                return
            used = min(available[part], n)
            available[part] -= used
            if n > used:
                first = unit + int(used // per_unit)  # first unit that lacks this part
                need[part] += n - used
                priority.setdefault(part, first)
                for q, k in m.transformations[m.producers[part]].inputs.items():
                    explode(q, k * (n - used), first, per_unit * k)

        # Inputs that planned jobs still lack come first: their outputs are already counted as available.
        for part, n in state.missing_inputs().items():
            explode(part, n, 0, float("inf"))
        unit = 1
        released = (o for o in state.open_orders() if o.release <= state.time)
        for o in sorted(released, key=lambda o: (o.deadline, -o.priority)):
            explode(o.product, o.missing, unit, 1)
            unit += o.missing

        reserve = self.queue_limit * len(m.machines) if self.reserve is None else self.reserve
        tight = state.intermediate_parts() + reserve >= wip_capacity(m)
        best, best_key = None, None
        for part in need:
            if tight and priority[part] > 1 and part not in m.final:
                continue
            t = m.producers[part]
            family = m.transformations[t].family
            for i in m.runners[t]:
                queue = state.queue(i)
                if self.queue_minutes is None:
                    if len(queue) >= self.queue_limit:
                        continue
                elif queue and sum(m.transformations[q].duration for q in queue) >= (
                    self.queue_minutes * m.machines[i].slots
                ):
                    continue
                switch = family is not None and state.upcoming_family(i) != family
                urgency = priority[part]
                if self.batch and not switch and family is not None and urgency > 0:
                    urgency = max(urgency - self.batch, 1)
                key = (urgency, m.distance_to_final.get(part, 99), switch, len(queue) + state.load(i), i)
                if best_key is None or key < best_key:
                    best, best_key = Assignment(t, i), key
        return best


def wip_capacity(m: ShopModel) -> float:
    """Room for intermediate parts: intermediate store and all output buffers."""
    if m.has_intermediate and m.intermediate_capacity is None:
        return float("inf")
    return (m.intermediate_capacity or 0) + sum(mc.output_buffer for mc in m.machines)
