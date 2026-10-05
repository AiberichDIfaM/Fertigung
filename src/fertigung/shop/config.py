"""Configuration schema for the shop-floor model: machines with position, buffers, setup matrix and operators;
stores; automatic transports; a staff pool per shift; truck pickups; selectable objectives.

Time is measured in minutes from `start`.
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Day = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
WEEKDAYS: list[Day] = ["mon", "tue", "wed", "thu", "fri"]
_CLOCK = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


def _clock(value: str) -> str:
    if not _CLOCK.match(value):
        raise ValueError(f"'{value}' is not a time of day (HH:MM)")
    return value


def minutes_of_day(clock: str) -> int:
    hours, minutes = clock.split(":")
    return int(hours) * 60 + int(minutes)


class StoreConfig(_Model):
    name: str
    kind: Literal["raw", "intermediate", "finished"]
    position: tuple[float, float]
    capacity: int | None = Field(None, ge=1, description="Parts; empty means unlimited")


class LayoutConfig(_Model):
    distance: Literal["manhattan", "euclidean"] = "manhattan"
    stores: list[StoreConfig]


class PartTypeConfig(_Model):
    name: str
    cost: float = Field(0.0, ge=0, description="Purchase price, relevant for raw materials")
    price: float = Field(0.0, ge=0, description="Sale price, relevant for final products")


class TransformationConfig(_Model):
    name: str
    inputs: list[str] = Field(min_length=1)
    output: str
    duration: int = Field(ge=1, description="Minutes")
    setup_family: str | None = Field(
        None, description="Tooling family; changing it on a machine costs setup time"
    )
    interruptible: bool = Field(
        True,
        description="May pause outside staffed hours and resume later; false: must finish in one stretch",
    )


class SetupTimeConfig(_Model):
    from_family: str = Field(alias="from")
    to_family: str = Field(alias="to")
    minutes: int = Field(ge=0)


class SetupConfig(_Model):
    operators: int = Field(1, ge=0, description="Workers needed while setting up")
    initial: int = Field(0, ge=0, description="Minutes to set up a machine that has no family yet")
    default: int = Field(0, ge=0, description="Minutes for a family change not listed in `times`")
    times: list[SetupTimeConfig] = []


class MachineTypeConfig(_Model):
    name: str
    transformations: list[str] = Field(min_length=1)
    slots: int = Field(1, ge=1, description="Jobs running in parallel")
    operators: int = Field(1, ge=0, description="Workers needed per running job; 0 runs unattended")
    setup: SetupConfig = SetupConfig()


class MachineConfig(_Model):
    name: str
    type: str
    position: tuple[float, float]
    input_buffer: int = Field(ge=1, description="Parts")
    output_buffer: int = Field(ge=1, description="Parts")


class VehicleConfig(_Model):
    name: str
    count: int = Field(1, ge=1)
    capacity: int = Field(1, ge=1, description="Parts per trip")


class TransportConfig(_Model):
    speed: float = Field(gt=0, description="Distance units per minute")
    handling: int = Field(0, ge=0, description="Minutes to load and unload per trip")
    vehicles: list[VehicleConfig] = Field(min_length=1)


class ShiftConfig(_Model):
    days: list[Day] = Field(WEEKDAYS, min_length=1)
    start: str
    end: str
    workers: int = Field(ge=0)

    check_clock = field_validator("start", "end")(_clock)

    @model_validator(mode="after")
    def _order(self):
        if minutes_of_day(self.end) <= minutes_of_day(self.start):
            raise ValueError(f"shift {self.start}-{self.end}: end must be after start (split night shifts)")
        return self


class StaffConfig(_Model):
    shifts: list[ShiftConfig] = Field(min_length=1)


class PickupConfig(_Model):
    days: list[Day] = Field(WEEKDAYS, min_length=1)
    time: str

    check_clock = field_validator("time")(_clock)


class LogisticsConfig(_Model):
    pickups: list[PickupConfig] = Field(min_length=1, description="When the truck collects finished orders")
    partial_shipments: bool = Field(False, description="Ship parts of an order or only complete orders")


class OrderConfig(_Model):
    product: str
    quantity: int = Field(ge=1)
    release: int = Field(0, ge=0, description="Minute from which the order may be worked on")
    deadline: int = Field(ge=0, description="Minute by which the order must have left with a truck")
    priority: int = Field(1, ge=1)
    price: float | None = Field(None, ge=0, description="Overrides the product's sale price")


class WeightsConfig(_Model):
    revenue: float | None = Field(None, ge=0, description="Per unit of revenue")
    material_cost: float | None = Field(None, ge=0, description="Per unit of raw material spend")
    holding_cost: float | None = Field(None, ge=0, description="Per part in progress or stored per hour")
    lateness: float | None = Field(None, ge=0, description="Per unit of an order per hour after its deadline")
    setup: float | None = Field(None, ge=0, description="Per setup hour")
    transport: float | None = Field(None, ge=0, description="Per vehicle hour")


PROFILES: dict[str, dict[str, float]] = {
    "balanced": {
        "revenue": 1,
        "material_cost": 1,
        "holding_cost": 0.5,
        "lateness": 5,
        "setup": 1,
        "transport": 1,
    },
    "on_time": {
        "revenue": 1,
        "material_cost": 1,
        "holding_cost": 0.2,
        "lateness": 50,
        "setup": 0.5,
        "transport": 0.5,
    },
    "revenue": {
        "revenue": 1,
        "material_cost": 1,
        "holding_cost": 0.2,
        "lateness": 1,
        "setup": 1,
        "transport": 1,
    },
    "low_stock": {
        "revenue": 1,
        "material_cost": 1,
        "holding_cost": 5,
        "lateness": 5,
        "setup": 1,
        "transport": 1,
    },
    "utilization": {
        "revenue": 1,
        "material_cost": 1,
        "holding_cost": 0.2,
        "lateness": 2,
        "setup": 5,
        "transport": 1,
    },
}


class ObjectiveConfig(_Model):
    profile: Literal["balanced", "on_time", "revenue", "low_stock", "utilization"] = "balanced"
    weights: WeightsConfig = WeightsConfig()

    def resolved(self) -> dict[str, float]:
        """Profile weights with the explicitly set weights applied on top."""
        return PROFILES[self.profile] | self.weights.model_dump(exclude_none=True)


class StartConfig(_Model):
    day: Day = "mon"
    time: str = "06:00"

    check_clock = field_validator("time")(_clock)


class ShopConfig(_Model):
    name: str
    time_unit: Literal["minute"] = "minute"
    start: StartConfig = StartConfig()
    layout: LayoutConfig
    part_types: list[PartTypeConfig]
    transformations: list[TransformationConfig]
    machine_types: list[MachineTypeConfig]
    machines: list[MachineConfig] = Field(min_length=1)
    transport: TransportConfig
    staff: StaffConfig
    logistics: LogisticsConfig
    orders: list[OrderConfig] = []
    objective: ObjectiveConfig = ObjectiveConfig()

    @model_validator(mode="after")
    def _check_references(self):
        errors = []
        for label, names in [
            ("store", [s.name for s in self.layout.stores]),
            ("part type", [p.name for p in self.part_types]),
            ("transformation", [t.name for t in self.transformations]),
            ("machine type", [m.name for m in self.machine_types]),
            ("machine", [m.name for m in self.machines]),
            ("vehicle", [v.name for v in self.transport.vehicles]),
        ]:
            errors += [f"duplicate {label} '{n}'" for n, c in Counter(names).items() if c > 1]

        kinds = Counter(s.kind for s in self.layout.stores)
        for kind in ("raw", "finished"):
            if kinds[kind] != 1:
                errors.append(f"layout needs exactly one {kind} store, found {kinds[kind]}")
        if kinds["intermediate"] > 1:
            errors.append("layout allows at most one intermediate store")

        parts = {p.name for p in self.part_types}
        for t in self.transformations:
            errors += [
                f"transformation '{t.name}': unknown part type '{p}'" for p in t.inputs if p not in parts
            ]
            if t.output not in parts:
                errors.append(f"transformation '{t.name}': unknown part type '{t.output}'")
            if t.output in t.inputs:
                errors.append(f"transformation '{t.name}': output is also an input")

        transformations = {t.name: t for t in self.transformations}
        for mt in self.machine_types:
            known = [t for t in mt.transformations if t in transformations]
            errors += [
                f"machine type '{mt.name}': unknown transformation '{t}'"
                for t in mt.transformations
                if t not in transformations
            ]
            families = {transformations[t].setup_family for t in known} - {None}
            for st in mt.setup.times:
                for family in (st.from_family, st.to_family):
                    if family not in families:
                        errors.append(
                            f"machine type '{mt.name}': setup family '{family}' is not used "
                            "by its transformations"
                        )

        machine_types = {m.name for m in self.machine_types}
        errors += [
            f"machine '{m.name}': unknown machine type '{m.type}'"
            for m in self.machines
            if m.type not in machine_types
        ]
        errors += [f"order: unknown product '{o.product}'" for o in self.orders if o.product not in parts]
        errors += [
            f"order for '{o.product}': deadline {o.deadline} is before its release {o.release}"
            for o in self.orders
            if o.deadline < o.release
        ]
        if errors:
            raise ValueError("; ".join(errors))
        return self


def load_shop(source: str | Path | dict) -> ShopConfig:
    if isinstance(source, dict):
        return ShopConfig.model_validate(source)
    path = Path(source)
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    return ShopConfig.model_validate(data)
