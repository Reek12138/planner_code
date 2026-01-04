# ai_planner/costmap/composite.py
from __future__ import annotations
import numpy as np
from typing import Tuple
from .base import BaseCostmap2D
from .numpy_costmap import NumpyCostmap2D

class CompositeCostmap2D(BaseCostmap2D):
    """
    将两个 costmap 合成为一个视图：cost = max(static, dynamic)
    也可按需改成加权和或优先级覆盖。
    """
    def __init__(self, static_cm: NumpyCostmap2D, dynamic_cm: NumpyCostmap2D):
        assert static_cm.size == dynamic_cm.size
        assert static_cm.resolution == dynamic_cm.resolution
        assert static_cm.origin == dynamic_cm.origin
        self.static = static_cm
        self.dynamic = dynamic_cm

    # properties
    @property
    def resolution(self) -> float:
        return self.static.resolution

    @property
    def origin(self) -> Tuple[float, float]:
        return self.static.origin

    @property
    def size(self) -> Tuple[int, int]:
        return self.static.size

    # transforms
    def world_to_cell(self, x: float, y: float):
        return self.static.world_to_cell(x, y)

    def cell_to_world(self, i: int, j: int):
        return self.static.cell_to_world(i, j)

    # queries
    def is_occupied(self, x: float, y: float) -> bool:
        return (self.static.cost_at(x, y) > 0.5) or (self.dynamic.cost_at(x, y) > 0.5)

    def cost_at(self, x: float, y: float) -> float:
        return max(self.static.cost_at(x, y), self.dynamic.cost_at(x, y))

    def raycast_free(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        # 简化：两层都必须通过
        return self.static.raycast_free(x0, y0, x1, y1) and self.dynamic.raycast_free(x0, y0, x1, y1)

    def inflate(self, radius: float) -> None:
        # 通常只对静态层做膨胀；动态层的半径直接包含安全裕度
        self.static.inflate(radius)
