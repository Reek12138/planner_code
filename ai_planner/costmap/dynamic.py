# ai_planner/costmap/dynamic.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Iterable, Optional
from .base import BaseCostmap2D
from .numpy_costmap import NumpyCostmap2D

@dataclass
class DynamicObstacle:
    """简化：圆形障碍，常速度模型"""
    x: float
    y: float
    vx: float
    vy: float
    radius: float

    def step(self, dt: float):
        self.x += self.vx * dt
        self.y += self.vy * dt

class DynamicLayer:
    """
    用一张与静态图同尺寸的grid存动态障碍；每帧重绘。
    - occ_val: 写入的占据值（例如 1.0 表示硬障碍）
    """
    def __init__(self,
                 shape: Tuple[int, int],
                 resolution: float,
                 origin: Tuple[float, float],
                 occ_val: float = 1.0,
                 oob_as_occ: bool = True):
        H, W = shape
        self.grid = np.zeros((H, W), dtype=float)
        self.cm = NumpyCostmap2D(self.grid, resolution, origin,
                                 occ_thresh=0.5,
                                 treat_out_of_bounds_as_occupied=oob_as_occ)
        self.occ_val = float(occ_val)
        self.objs: List[DynamicObstacle] = []

    def set_obstacles(self, obstacles: Iterable[DynamicObstacle]):
        self.objs = list(obstacles)

    def clear(self):
        self.grid.fill(0.0)

    def step(self, dt: float):
        """推进障碍状态（常速度）"""
        for o in self.objs:
            o.step(dt)

    def _rasterize_disc(self, cx: float, cy: float, r: float):
        """把圆形写进 grid"""
        i0, j0 = self.cm.world_to_cell(cx, cy)
        r_cells = int(np.ceil(r / self.cm.resolution))
        H, W = self.grid.shape
        for di in range(-r_cells, r_cells + 1):
            for dj in range(-r_cells, r_cells + 1):
                if di*di + dj*dj <= r_cells*r_cells:
                    i = i0 + di
                    j = j0 + dj
                    if 0 <= i < H and 0 <= j < W:
                        self.grid[i, j] = self.occ_val

    def redraw(self):
        """按当前障碍位置重绘动态层"""
        self.clear()
        for o in self.objs:
            self._rasterize_disc(o.x, o.y, o.radius)

    def as_costmap(self) -> NumpyCostmap2D:
        """返回一个可被 planner 使用的 costmap 视图（与内部 grid 共享内存）"""
        return self.cm
