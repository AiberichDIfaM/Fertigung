import pytest

from fertigung.core.config import load_config, reference_config
from fertigung.core.plant import Plant
from fertigung.core.reward import Reward
from fertigung.core.simulation import Simulation
from fertigung.core.validation import validate
from fertigung.generator import random_plant
from fertigung.heuristics import Lookahead, pull

TINY = {
    "name": "tiny",
    "buffer_capacity": 2,
    "part_types": [{"name": "r", "cost": 1}, {"name": "i"}, {"name": "f", "price": 5}],
    "transformations": [
        {"name": "make", "inputs": ["r"], "output": "i", "duration": 2},
        {"name": "finish", "inputs": ["i", "i"], "output": "f", "duration": 1},
    ],
    "machine_types": [{"name": "mt", "slots": 3, "transformations": ["make", "finish"]}],
    "machines": [{"name": "m", "type": "mt"}],
}


def test_reference_is_valid():
    assert validate(reference_config()) == []


def test_validation_reports_unassigned_transformation():
    data = reference_config().model_dump()
    data["machine_types"][0]["transformations"].remove("tr10")
    messages = [i.message for i in validate(load_config(data))]
    assert "part type 'b7' is treated as raw material because tr10 is not available" in messages


def test_config_rejects_unknown_references():
    with pytest.raises(ValueError, match="unknown part type 'x'"):
        load_config(
            {**TINY, "transformations": [{"name": "t", "inputs": ["x"], "output": "i", "duration": 1}]}
        )


def test_timing_blocking_and_sales():
    sim = Simulation(Plant(load_config(TINY)))
    for _ in range(3):
        sim.dispatch(0, 0)
    sim.advance()
    assert sim.buffer == []
    sim.advance()
    assert [p.type for p in sim.buffer] == ["i", "i"]
    assert sim.events[-1].kind == "blocked"
    sim.dispatch(0, 1)
    sim.advance()
    assert sim.ledger.shipped["f"] == 1 and sim.ledger.revenue == 5
    assert [p.type for p in sim.buffer] == ["i"] and sim.jobs[0] == []
    assert sim.ledger.material_cost == 3


def test_pull_meets_reference_orders():
    sim = Simulation(Plant(reference_config()))
    sim.run(pull, 300)
    kpis = sim.kpis()
    assert kpis["orders_on_time"] == len(sim.orders)
    assert kpis["revenue"] == sum(e.amount for e in sim.events if e.kind == "ship")


def test_lookahead_is_never_worse_than_pull():
    config = random_plant(50019)
    results = {}
    for name, policy in [("pull", pull), ("lookahead", Lookahead())]:
        sim = Simulation(Plant(config))
        reward = Reward(sim, config.reward.model_copy(update={"shaping": 0.0}))
        sim.run(policy, 200, reward)
        results[name] = reward.total
    assert results["lookahead"] >= results["pull"]
