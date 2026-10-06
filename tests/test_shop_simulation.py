from importlib import resources

import yaml

from fertigung.shop.config import load_shop
from fertigung.shop.heuristics import ShopPull
from fertigung.shop.model import ShopModel
from fertigung.shop.simulation import ShopSimulation

# Machine halfway between the stores (30 m, 1 minute at 60 m/min), 2 minutes handling per trip.
TINY = {
    "name": "tiny",
    "start": {"day": "mon", "time": "00:00"},
    "layout": {
        "stores": [
            {"name": "raw", "kind": "raw", "position": [0, 0]},
            {"name": "finished", "kind": "finished", "position": [60, 0]},
        ]
    },
    "part_types": [{"name": "r", "cost": 1}, {"name": "f", "price": 10}],
    "transformations": [{"name": "make", "inputs": ["r"], "output": "f", "duration": 5, "setup_family": "a"}],
    "machine_types": [{"name": "mt", "transformations": ["make"], "setup": {"initial": 10}}],
    "machines": [{"name": "m", "type": "mt", "position": [30, 0], "input_buffer": 1, "output_buffer": 1}],
    "transport": {"speed": 60, "handling": 2, "vehicles": [{"name": "cart"}]},
    "staff": {"shifts": [{"days": ["mon"], "start": "00:00", "end": "08:00", "workers": 1}]},
    "logistics": {"pickups": [{"days": ["mon"], "time": "00:30"}]},
    "orders": [{"product": "f", "quantity": 1, "deadline": 30}],
}


def run_tiny(config):
    sim = ShopSimulation(ShopModel(load_shop(config)))
    sim.run(ShopPull(), 40)
    return sim, {e.kind: e.time for e in sim.events}


def test_transport_setup_run_and_pickup():
    sim, times = run_tiny(TINY)
    # raw part arrives at 0 + 2 + 1 = 3, setup 3..13, run 13..18, to the finished store by 21, truck at 30
    assert (times["setup"], times["start"], times["finish"], times["ship"]) == (3, 13, 18, 30)
    assert sim.kpis()["orders_on_time"] == 1 and sim.ledger.revenue == 10


def test_shift_gap_pauses_interruptible_job():
    config = TINY | {
        "staff": {
            "shifts": [
                {"days": ["mon"], "start": "00:00", "end": "00:15", "workers": 1},
                {"days": ["mon"], "start": "00:20", "end": "08:00", "workers": 1},
            ]
        }
    }
    _, times = run_tiny(config)
    assert times["finish"] == 23  # 5 minutes without staff in between


def test_workshop_week_meets_all_orders():
    workshop = yaml.safe_load(
        resources.files("fertigung.configs").joinpath("workshop.yaml").read_text("utf-8")
    )
    sim = ShopSimulation(ShopModel(load_shop(workshop)))
    sim.run(ShopPull(), 6400)
    assert sim.kpis()["orders_on_time"] == len(sim.orders)
    assert all(j.state == "done" for j in sim.jobs.values())
