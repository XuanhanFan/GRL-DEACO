from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class GridPoint:
    x: int
    y: int
    z: int

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __eq__(self, other):
        if not isinstance(other, GridPoint):
            return False
        return self.x == other.x and self.y == other.y and (self.z == other.z)

    def to_tuple(self):
        return (self.x, self.y, self.z)

    @staticmethod
    def from_tuple(t):
        return GridPoint(t[0], t[1], t[2])

@dataclass
class FitnessData:
    J_total: float
    E_op: float
    CO2_op: float
    CO2_emb: float
    L: int
    N_bend: int
    viol: float
    alt: float
    f_Energy: float
    f_Install: float
    f_Height: float

@dataclass
class RoutingResult:
    success: bool
    path: Optional[list]
    fitness: float
    fitness_data: Optional[FitnessData]
    elapsed_time: float
    diagnostics: dict[str, Any]

def snap(p, res):
    return tuple([round(round(x / res) * res, 4) for x in p])
