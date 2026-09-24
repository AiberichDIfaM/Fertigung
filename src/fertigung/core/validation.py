import math
from dataclasses import dataclass

from fertigung.core.config import PlantConfig
from fertigung.core.plant import Plant


@dataclass(frozen=True)
class Issue:
    level: str
    message: str


def material_costs(plant: Plant) -> dict[str, float]:
    """Cheapest raw material cost per part type; math.inf means the part cannot be produced."""
    cost = {p: plant.cost[p] if plant.is_raw(p) else math.inf for p in plant.part_types}
    for _ in plant.part_types:
        changed = False
        for i in plant.assigned_transformations:
            t = plant.transformations[i]
            c = sum(n * cost[p] for p, n in t.inputs.items())
            if c < cost[t.output]:
                cost[t.output] = c
                changed = True
        if not changed:
            break
    return cost


def validate(config: PlantConfig) -> list[Issue]:
    plant = Plant(config)
    issues = []

    def error(msg):
        issues.append(Issue("error", msg))

    def warning(msg):
        issues.append(Issue("warning", msg))

    assigned = set(plant.assigned_transformations)
    unassigned = [t for i, t in enumerate(plant.transformations) if i not in assigned]
    for t in unassigned:
        warning(f"transformation '{t.name}' is not available on any machine")

    used_types = {m.type for m in plant.machines}
    for mt in config.machine_types:
        if mt.name not in used_types:
            warning(f"machine type '{mt.name}' has no machines")

    involved = {p for i in assigned for p in plant.transformations[i].inputs} | {
        plant.transformations[i].output for i in assigned
    }
    for p in plant.raw:
        producers = [t.name for t in unassigned if t.output == p]
        if producers:
            warning(
                f"part type '{p}' is treated as raw material because {', '.join(producers)} is not available"
            )
        elif p not in involved:
            warning(f"part type '{p}' is not used by any transformation")

    cost = material_costs(plant)
    for p in plant.part_types:
        if p in involved and cost[p] == math.inf:
            (error if plant.is_final(p) else warning)(f"part type '{p}' cannot be produced")

    for p in plant.final:
        if plant.price[p] <= 0:
            warning(f"final product '{p}' has no sale price")
        elif cost[p] < math.inf and plant.price[p] <= cost[p]:
            warning(f"final product '{p}' sells for {plant.price[p]:g} but needs {cost[p]:g} in raw material")

    for i in assigned:
        t = plant.transformations[i]
        wip_inputs = sum(n for p, n in t.inputs.items() if not plant.is_raw(p))
        if wip_inputs > plant.buffer_capacity:
            error(
                f"transformation '{t.name}' needs {wip_inputs} intermediate parts "
                f"but the buffer holds {plant.buffer_capacity}"
            )

    for m in plant.machines:
        if not any(
            all(cost[p] < math.inf for p in plant.transformations[t].inputs) for t in m.transformations
        ):
            warning(f"machine '{m.name}' can never start a transformation")

    for o in config.orders:
        if not plant.is_final(o.product):
            error(f"order for '{o.product}': not a final product")
        elif cost[o.product] == math.inf:
            error(f"order for '{o.product}': product cannot be produced")

    return issues
