import multiprocessing
import os
from concurrent.futures import Executor, ProcessPoolExecutor

from fertigung.shop.plan import Assignment, Planner, ShopState, Simulator
from fertigung.shop.solvers.pull import ShopPull

OPTIONS = [{"queue_limit": 1}, {"queue_limit": 2}, {"queue_limit": 4}, {"queue_minutes": 120}]
_pool: Executor | None = None


def _executor() -> Executor:
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(min(8, os.cpu_count() or 1))
    return _pool


def _trial(args: tuple[Simulator, Planner, int]) -> float:
    sim, planner, minutes = args
    sim.run(planner, minutes)
    return sim.objective()


class ShopAuto:
    """Simulates each option (ShopPull parameters) from the current state to the end of the run, follows the
    best one and decides again every `interval` minutes. The trial runs go to worker
    processes unless `parallel` is false or this already is a child process (nested pools hang on Windows)."""

    def __init__(self, options: list[dict] | None = None, interval: int = 1440, parallel: bool = True):
        self.params = options or OPTIONS
        self.options = [ShopPull(**params) for params in self.params]
        self.interval = interval
        self.parallel = parallel and multiprocessing.parent_process() is None
        self.current, self.due = None, 0
        self.decisions: list[dict] = []

    def __call__(self, state: ShopState) -> Assignment | None:
        if self.current is None or state.time >= self.due:
            self.due = state.time + self.interval
            trials = [(state.fork(), option, state.end - state.time) for option in self.options]
            values = list(_executor().map(_trial, trials) if self.parallel else map(_trial, trials))
            best = max(range(len(values)), key=values.__getitem__)
            self.current = self.options[best]
            self.decisions.append({"time": state.time, "option": self.params[best]})
        return self.current(state)
