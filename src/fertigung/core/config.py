import json
from collections import Counter
from importlib import resources
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PartTypeConfig(_Model):
    name: str
    cost: float = Field(0.0, ge=0, description="Purchase price, relevant for raw materials")
    price: float = Field(0.0, ge=0, description="Sale price, relevant for final products")


class TransformationConfig(_Model):
    name: str
    inputs: list[str] = Field(min_length=1)
    output: str
    duration: int = Field(ge=1)


class MachineTypeConfig(_Model):
    name: str
    slots: int = Field(ge=1)
    transformations: list[str]


class MachineConfig(_Model):
    name: str
    type: str


class OrderConfig(_Model):
    product: str
    quantity: int = Field(ge=1)
    deadline: int = Field(ge=0)
    price: float | None = Field(None, ge=0, description="Overrides the product's sale price")


class PlantConfig(_Model):
    name: str
    buffer_capacity: int = Field(ge=1)
    part_types: list[PartTypeConfig]
    transformations: list[TransformationConfig]
    machine_types: list[MachineTypeConfig]
    machines: list[MachineConfig] = Field(min_length=1)
    orders: list[OrderConfig] = []

    @model_validator(mode="after")
    def _check_references(self):
        errors = []
        for label, names in [
            ("part type", [p.name for p in self.part_types]),
            ("transformation", [t.name for t in self.transformations]),
            ("machine type", [m.name for m in self.machine_types]),
            ("machine", [m.name for m in self.machines]),
        ]:
            errors += [f"duplicate {label} '{n}'" for n, c in Counter(names).items() if c > 1]

        parts = {p.name for p in self.part_types}
        for t in self.transformations:
            errors += [
                f"transformation '{t.name}': unknown part type '{p}'" for p in t.inputs if p not in parts
            ]
            if t.output not in parts:
                errors.append(f"transformation '{t.name}': unknown part type '{t.output}'")
            if t.output in t.inputs:
                errors.append(f"transformation '{t.name}': output is also an input")

        transformations = {t.name for t in self.transformations}
        for mt in self.machine_types:
            errors += [
                f"machine type '{mt.name}': unknown transformation '{t}'"
                for t in mt.transformations
                if t not in transformations
            ]

        machine_types = {m.name for m in self.machine_types}
        errors += [
            f"machine '{m.name}': unknown machine type '{m.type}'"
            for m in self.machines
            if m.type not in machine_types
        ]
        errors += [f"order: unknown product '{o.product}'" for o in self.orders if o.product not in parts]

        if errors:
            raise ValueError("; ".join(errors))
        return self


def load_config(source: str | Path | dict) -> PlantConfig:
    if isinstance(source, dict):
        return PlantConfig.model_validate(source)
    path = Path(source)
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    return PlantConfig.model_validate(data)


def reference_config() -> PlantConfig:
    text = resources.files("fertigung.configs").joinpath("reference.yaml").read_text(encoding="utf-8")
    return PlantConfig.model_validate(yaml.safe_load(text))
