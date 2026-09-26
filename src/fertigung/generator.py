import math
import random

from fertigung.core.config import PlantConfig
from fertigung.core.plant import Plant
from fertigung.core.simulation import Simulation
from fertigung.core.validation import material_costs, validate
from fertigung.heuristics import pull


def _layers(rng: random.Random) -> tuple[list[list[str]], list[str]]:
    raw = [f"r{i}" for i in range(rng.randint(3, 7))]
    layers = [raw]
    for depth in range(1, rng.randint(2, 4) + 1):
        layers.append([f"i{depth}_{i}" for i in range(rng.randint(2, 5))])
    finals = [f"p{i}" for i in range(rng.randint(1, 3))]
    return layers, finals


def _draft(rng: random.Random) -> dict:
    layers, finals = _layers(rng)
    transformations = []
    for depth, layer in enumerate(layers[1:], start=1):
        for part in layer:
            below = [p for lower in layers[:depth] for p in lower]
            inputs = [rng.choice(layers[depth - 1])] + rng.sample(below, rng.randint(0, 2))
            transformations.append(
                {"name": f"t_{part}", "inputs": inputs, "output": part, "duration": rng.randint(1, 8)}
            )
    upper = [p for layer in layers[-2:] for p in layer if not p.startswith("r")]
    for part in finals:
        inputs = rng.sample(upper, min(len(upper), rng.randint(2, 4)))
        transformations.append(
            {"name": f"t_{part}", "inputs": inputs, "output": part, "duration": rng.randint(5, 15)}
        )

    # Every intermediate must be consumed, otherwise it would become a final product.
    consumed = {p for t in transformations for p in t["inputs"]}
    for part in [p for layer in layers[1:] for p in layer if p not in consumed]:
        rng.choice(transformations[-len(finals) :])["inputs"].append(part)

    names = [t["name"] for t in transformations]
    rng.shuffle(names)
    machine_types = []
    while names:
        group = [names.pop() for _ in range(min(len(names), rng.randint(1, 3)))]
        shared = rng.choice(machine_types)["transformations"][0] if machine_types else None
        if shared and shared not in group and rng.random() < 0.4:
            group.append(shared)
        machine_types.append(
            {"name": f"mt{len(machine_types)}", "slots": rng.randint(1, 5), "transformations": group}
        )
    machines = [
        {"name": f"{mt['name']}_{k}", "type": mt["name"]}
        for mt in machine_types
        for k in range(rng.choice([1, 1, 2]))
    ]

    used = {p for t in transformations for p in t["inputs"]}
    parts = [{"name": p, "cost": rng.randint(5, 15)} for p in layers[0] if p in used]
    parts += [{"name": p} for layer in layers[1:] for p in layer]
    parts += [{"name": p} for p in finals]
    wip_inputs = max(sum(1 for p in t["inputs"] if not p.startswith("r")) for t in transformations)
    return {
        "name": "generated",
        "buffer_capacity": wip_inputs + rng.randint(3, 8),
        "part_types": parts,
        "transformations": transformations,
        "machine_types": machine_types,
        "machines": machines,
    }


def _price_and_orders(data: dict, rng: random.Random) -> dict:
    plant = Plant(PlantConfig.model_validate(data))
    cost = material_costs(plant)
    for p in data["part_types"]:
        if plant.is_final(p["name"]):
            p["price"] = round(cost[p["name"]] * rng.uniform(1.2, 1.6))
    avg_price = sum(p.get("price", 0) for p in data["part_types"]) / len(plant.final)

    orders = [
        {"product": rng.choice(plant.final), "quantity": rng.randint(1, 4), "deadline": 10_000}
        for _ in range(rng.randint(2, 5))
    ]
    data["orders"] = orders
    sim = Simulation(Plant(PlantConfig.model_validate(data)))
    sim.run(pull, 600)
    # Deadlines around the time the pull heuristic needs, so some plants are tight and some are easy.
    data["orders"] = [
        o | {"deadline": max(5, int(s.completed_at * rng.uniform(0.85, 1.3)))}
        for o, s in zip(orders, sim.orders, strict=True)
        if s.completed_at is not None
    ]
    data["reward"] = {
        "holding_cost": round(avg_price / 650, 4),
        "lateness": round(avg_price / 65, 4),
        "shaping": 1.0,
        "scale": round(3.25 / avg_price, 6),
    }
    return data


def random_plant(seed: int, name: str | None = None) -> PlantConfig:
    """A random but valid plant with calibrated order deadlines and reward weights scaled to its prices."""
    rng = random.Random(seed)
    for _ in range(100):
        data = _draft(rng)
        config = PlantConfig.model_validate(data)
        if any(i.level == "error" for i in validate(config)):
            continue
        plant = Plant(config)
        if not plant.final or any(math.isinf(c) for c in material_costs(plant).values()):
            continue
        data = _price_and_orders(data, rng)
        if not data["orders"]:
            continue
        data["name"] = name or f"generated-{seed}"
        return PlantConfig.model_validate(data)
    raise RuntimeError(f"no valid plant for seed {seed}")
