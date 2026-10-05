"""Semantic checks of a shop configuration beyond names and references."""

import math
from collections import Counter

from fertigung.core.validation import Issue
from fertigung.shop.config import ShopConfig


def validate_shop(config: ShopConfig) -> list[Issue]:
    issues = []

    def error(msg):
        issues.append(Issue("error", msg))

    def warning(msg):
        issues.append(Issue("warning", msg))

    transformations = {t.name: t for t in config.transformations}
    machine_types = {mt.name: mt for mt in config.machine_types}
    used_types = {m.type for m in config.machines}
    assigned = {t for mt in config.machine_types if mt.name in used_types for t in mt.transformations}

    for name in transformations:
        if name not in assigned:
            warning(f"transformation '{name}' is not available on any machine")
    for mt in config.machine_types:
        if mt.name not in used_types:
            warning(f"machine type '{mt.name}' has no machines")

    produced = {transformations[t].output for t in assigned}
    consumed = {p for t in assigned for p in transformations[t].inputs}
    prices = {p.name: p.price for p in config.part_types}
    costs = {p.name: p.cost for p in config.part_types}
    raw = {p.name for p in config.part_types if p.name not in produced}
    final = produced - consumed

    # Cheapest raw material cost and shortest production time per part; math.inf: cannot be produced.
    cost = {p: (costs[p] if p in raw else math.inf) for p in prices}
    lead = {p: (0 if p in raw else math.inf) for p in prices}
    for _ in prices:
        for t in assigned:
            tr = transformations[t]
            counts = Counter(tr.inputs)
            cost[tr.output] = min(cost[tr.output], sum(n * cost[p] for p, n in counts.items()))
            lead[tr.output] = min(lead[tr.output], tr.duration + max(lead[p] for p in counts))

    for p in sorted(produced | consumed):
        if cost[p] == math.inf:
            (error if p in final else warning)(f"part type '{p}' cannot be produced")
    for p in sorted(final):
        if prices[p] <= 0:
            warning(f"final product '{p}' has no sale price")
        elif cost[p] < math.inf and prices[p] <= cost[p]:
            warning(f"final product '{p}' sells for {prices[p]:g} but needs {cost[p]:g} in raw material")
    for p in sorted(raw & {p.name for p in config.part_types if p.price > 0}):
        warning(f"raw material '{p}' has a sale price but is never sold")

    max_workers = max(s.workers for s in config.staff.shifts)
    for m in config.machines:
        mt = machine_types[m.type]
        for t in mt.transformations:
            if t in transformations and len(transformations[t].inputs) > m.input_buffer:
                error(
                    f"machine '{m.name}': input buffer {m.input_buffer} is smaller than the "
                    f"{len(transformations[t].inputs)} inputs of '{t}'"
                )
        if mt.operators > max_workers:
            error(f"machine '{m.name}' needs {mt.operators} operators, shifts have at most {max_workers}")
        if mt.setup.operators > max_workers:
            error(
                f"setting up machine '{m.name}' needs {mt.setup.operators} workers, "
                f"shifts have at most {max_workers}"
            )

    positions = Counter(tuple(m.position) for m in config.machines)
    for position, n in positions.items():
        if n > 1:
            warning(f"{n} machines share the position {list(position)}")
    if not any(s.kind == "intermediate" for s in config.layout.stores):
        warning("no intermediate store: parts wait in output buffers when the target input buffer is full")

    for o in config.orders:
        if o.product not in final:
            error(f"order for '{o.product}': not a final product")
        elif lead[o.product] == math.inf:
            error(f"order for '{o.product}': product cannot be produced")
        elif o.deadline - o.release < lead[o.product]:
            warning(
                f"order for '{o.product}': {o.deadline - o.release} minutes between release and deadline, "
                f"production alone takes at least {lead[o.product]}"
            )
    return issues
