"""Planning strategies. They only use `fertigung.shop.plan` and the model, never the simulation."""

from fertigung.shop.plan import Planner
from fertigung.shop.solvers.auto import ShopAuto
from fertigung.shop.solvers.pull import ShopPull

SOLVERS = {
    "pull": "Demand-driven assignment with a fixed queue limit",
    "auto": "Simulates every queue limit (1-4) once a day and follows the best",
}


def make_solver(name: str, queue_limit: int = 2) -> Planner:
    if name == "pull":
        return ShopPull(queue_limit)
    if name == "auto":
        return ShopAuto()
    raise ValueError(f"unknown solver {name}")
