import pytest

from fertigung.core.config import load_config
from fertigung.core.plant import Plant
from fertigung.core.reward import Reward
from fertigung.core.simulation import Simulation
from tests.test_core import TINY


def test_reward_components():
    config = load_config(
        TINY
        | {
            "orders": [{"product": "f", "quantity": 2, "deadline": 1}],
            "reward": {"holding_cost": 0.5, "idle": 2, "lateness": 1},
        }
    )
    sim = Simulation(Plant(config))
    reward = Reward(sim)
    for _ in range(3):
        sim.dispatch(0, 0)
    sim.advance()
    sim.advance()
    sim.dispatch(0, 1)
    sim.advance()

    assert reward(sim) == pytest.approx(-4.5)
    assert reward.totals == pytest.approx(
        {"revenue": 5, "material_cost": -3, "holding_cost": -1.5, "lateness": -3, "idle": -2, "shaping": 0}
    )
