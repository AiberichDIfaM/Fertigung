from collections import Counter
from dataclasses import dataclass

import networkx as nx

from fertigung.core.config import PlantConfig


@dataclass(frozen=True)
class Transformation:
    name: str
    inputs: Counter
    output: str
    duration: int


@dataclass(frozen=True)
class Machine:
    name: str
    type: str
    slots: int
    transformations: tuple[int, ...]


class Plant:
    """Immutable, index-based view of a PlantConfig.

    Raw materials are part types that no transformation produces; they are bought on demand.
    Final products are part types that no transformation consumes; they are sold on completion.
    """

    def __init__(self, config: PlantConfig):
        self.config = config
        self.name = config.name
        self.buffer_capacity = config.buffer_capacity
        self.part_types = [p.name for p in config.part_types]
        self.cost = {p.name: p.cost for p in config.part_types}
        self.price = {p.name: p.price for p in config.part_types}

        self.transformations = [
            Transformation(t.name, Counter(t.inputs), t.output, t.duration) for t in config.transformations
        ]
        t_index = {t.name: i for i, t in enumerate(self.transformations)}
        machine_types = {mt.name: mt for mt in config.machine_types}
        self.machines = [
            Machine(
                m.name,
                m.type,
                machine_types[m.type].slots,
                tuple(t_index[t] for t in machine_types[m.type].transformations),
            )
            for m in config.machines
        ]

        assigned = {i for m in self.machines for i in m.transformations}
        self.assigned_transformations = sorted(assigned)
        produced = {self.transformations[i].output for i in assigned}
        consumed = {p for i in assigned for p in self.transformations[i].inputs}
        self.raw = [p for p in self.part_types if p not in produced]
        self.final = [p for p in self.part_types if p in produced and p not in consumed]
        self.intermediate = [p for p in self.part_types if p in produced and p in consumed]

        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.part_types)
        for i in assigned:
            t = self.transformations[i]
            self.graph.add_edges_from((p, t.output) for p in t.inputs)

        self.distance_to_final = {}
        for p in self.part_types:
            lengths = nx.single_source_shortest_path_length(self.graph, p)
            reachable = [lengths[f] for f in self.final if f in lengths]
            self.distance_to_final[p] = min(reachable) if reachable else None

        self._raw = frozenset(self.raw)
        self._final = frozenset(self.final)

    def is_raw(self, part_type: str) -> bool:
        return part_type in self._raw

    def is_final(self, part_type: str) -> bool:
        return part_type in self._final
