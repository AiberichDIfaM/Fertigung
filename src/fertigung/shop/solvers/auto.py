from fertigung.shop.plan import Assignment, ShopState
from fertigung.shop.solvers.pull import ShopPull


class ShopAuto:
    """Simulates ShopPull with each queue limit from the current state to the end of the run, follows the best
    one and decides again every `interval` minutes. Costs one full simulation per option and decision."""

    def __init__(self, queue_limits: tuple[int, ...] = (1, 2, 3, 4), interval: int = 1440):
        self.options = [ShopPull(q) for q in queue_limits]
        self.interval = interval
        self.current, self.due = None, 0
        self.decisions: list[dict] = []

    def __call__(self, state: ShopState) -> Assignment | None:
        if self.current is None or state.time >= self.due:
            self.due = state.time + self.interval
            self.current = max(self.options, key=lambda option: self._value(state, option))
            self.decisions.append({"time": state.time, "queue_limit": self.current.queue_limit})
        return self.current(state)

    @staticmethod
    def _value(state: ShopState, option: ShopPull) -> float:
        trial = state.fork()
        trial.run(option, state.end - state.time)
        return trial.objective()
