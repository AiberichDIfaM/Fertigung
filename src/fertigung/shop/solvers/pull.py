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
    possible queued job."""

    def __init__(self, queue_limit: int = 2, reserve: int | None = None):
        self.queue_limit = queue_limit
        self.reserve = reserve

    def __call__(self, state: ShopState) -> Assignment | None:
        m = state.model
        available = state.available()
        need, priority = Counter(), {}

        def explode(part, n, unit):
            if part in m.raw or part not in m.producers:
                return
            used = min(available[part], n)
            available[part] -= used
            if n > used:
                need[part] += n - used
                priority.setdefault(part, unit)
                for q, k in m.transformations[m.producers[part]].inputs.items():
                    explode(q, k * (n - used), unit)

        # Inputs that planned jobs still lack come first: their outputs are already counted as available.
        for part, n in state.missing_inputs().items():
            explode(part, n, 0)
        unit = 1
        released = (o for o in state.open_orders() if o.release <= state.time)
        for o in sorted(released, key=lambda o: (o.deadline, -o.priority)):
            for _ in range(o.missing):
                explode(o.product, 1, unit)
                unit += 1

        reserve = self.queue_limit * len(m.machines) if self.reserve is None else self.reserve
        tight = state.intermediate_parts() + reserve >= wip_capacity(m)
        best, best_key = None, None
        for part in need:
            if tight and priority[part] > 1 and part not in m.final:
                continue
            t = m.producers[part]
            family = m.transformations[t].family
            for i in m.runners[t]:
                queued = len(state.queue(i))
                if queued >= self.queue_limit:
                    continue
                key = (
                    priority[part],
                    m.distance_to_final.get(part, 99),
                    family is not None and state.upcoming_family(i) != family,
                    queued + state.load(i),
                    i,
                )
                if best_key is None or key < best_key:
                    best, best_key = Assignment(t, i), key
        return best


def wip_capacity(m: ShopModel) -> float:
    """Room for intermediate parts: intermediate store and all output buffers."""
    if m.has_intermediate and m.intermediate_capacity is None:
        return float("inf")
    return (m.intermediate_capacity or 0) + sum(mc.output_buffer for mc in m.machines)
