"""Interface between the simulation and the solvers.

A solver is a `Planner`: it sees the shop only through `ShopState` and answers with `Assignment`s. For what-if
runs it forks the state into a separate `Simulator`. The simulation imports nothing from the solvers, the
solvers import nothing from the simulation.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from fertigung.shop.model import ShopModel


@dataclass(frozen=True)
class Assignment:
    transformation: int
    machine: int


@dataclass(frozen=True)
class OrderState:
    product: str
    quantity: int
    delivered: int
    release: int
    deadline: int
    priority: int

    @property
    def missing(self) -> int:
        return self.quantity - self.delivered


class ShopState(Protocol):
    """Read-only view of the shop at the current minute."""

    model: ShopModel

    @property
    def time(self) -> int: ...

    @property
    def end(self) -> int:
        """Last minute of the current run."""

    def open_orders(self) -> list[OrderState]: ...

    def queue(self, machine: int) -> list[int]:
        """Transformations assigned to the machine and not started yet, in order."""

    def load(self, machine: int) -> int:
        """Jobs on the machine that are not done (queued, in setup, running or blocked)."""

    def upcoming_family(self, machine: int) -> str | None:
        """Tooling family the machine will have after its queue."""

    def available(self) -> Counter:
        """Produced parts outside the input buffers (output buffers, intermediate and finished store, on the
        way there) plus the outputs of all jobs that are not done."""

    def missing_inputs(self) -> Counter:
        """Inputs of planned jobs that have no material allocated yet."""

    def intermediate_parts(self) -> int:
        """Intermediate parts in output buffers, the intermediate store or on the way there, plus intermediate
        outputs of jobs that are not done."""

    def fork(self) -> "Simulator":
        """A separate simulation that continues from this state."""


class Planner(Protocol):
    def __call__(self, state: ShopState) -> Assignment | None:
        """Next assignment at this minute, or None to let time pass."""


class Simulator(Protocol):
    @property
    def state(self) -> ShopState: ...

    def run(self, planner: Planner, minutes: int) -> None: ...

    def objective(self) -> float: ...
