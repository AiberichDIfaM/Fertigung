"""Planning strategies. They only use `fertigung.shop.plan` and the model, never the simulation."""

from fertigung.shop.plan import Planner
from fertigung.shop.solvers.auto import ShopAuto
from fertigung.shop.solvers.pull import ShopPull

SOLVERS = {
    "pull": "Demand-driven assignment with a fixed queue limit",
    "auto": "Simulates several pull settings once a day and follows the best",
}


def make_solver(name: str, **params) -> Planner:
    if name == "pull":
        return ShopPull(**params)
    if name == "auto":
        return ShopAuto(**params)
    raise ValueError(f"unknown solver {name}")
