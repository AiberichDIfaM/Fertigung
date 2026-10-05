"""Baseline planning rules for the shop-floor simulation."""

from collections import Counter

from fertigung.shop.simulation import RAW, ShopSimulation


class ShopPull:
    """Demand-driven assignment: explode the open orders (by deadline) against everything that exists or is
    already planned, and assign the most urgent missing operation closest to the final product. Prefers
    machines already set up for its tooling family, then short queues; at most `queue_limit` jobs wait per
    machine."""

    def __init__(self, queue_limit: int = 2):
        self.queue_limit = queue_limit

    def __call__(self, sim: ShopSimulation) -> tuple[int, int] | None:
        m = sim.model
        available = Counter()
        for loc, stock in sim.stock.items():
            if loc != RAW and loc[0] != "in":
                available.update(stock)
        for _, r in sim.arrivals:
            if r.dst[0] != "in":
                available[r.part] += r.qty
        for job in sim.active_jobs():
            available[m.transformations[job.transformation].output] += 1

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
        for job in sim.active_jobs():
            for part, n in job.needs.items():
                explode(part, n, 0)
        unit = 1
        for o in sorted(
            (o for o in sim.orders if o.open and o.release <= sim.time),
            key=lambda o: (o.deadline, -o.priority),
        ):
            for _ in range(o.quantity - o.delivered):
                explode(o.product, 1, unit)
                unit += 1

        best, best_key = None, None
        for part in need:
            t = m.producers[part]
            family = m.transformations[t].family
            for i in m.runners[t]:
                if len(sim.queues[i]) >= self.queue_limit:
                    continue
                last = sim.jobs[sim.queues[i][-1]] if sim.queues[i] else None
                upcoming = m.transformations[last.transformation].family if last else sim.family[i]
                key = (
                    priority[part],
                    m.distance_to_final.get(part, 99),
                    family is not None and upcoming != family,
                    len(sim.queues[i]) + len(sim.active_jobs(i)),
                    i,
                )
                if best_key is None or key < best_key:
                    best, best_key = (t, i), key
        return best
