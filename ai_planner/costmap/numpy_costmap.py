# ai_planner/costmap/numpy_costmap.py
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional
from .base import BaseCostmap2D

class NumpyCostmap2D(BaseCostmap2D):
    """
    简单可用的 2D costmap 实现：
    - grid: np.ndarray[H, W], 0 表示自由, >0 表示占据/代价
    - occ_thresh: 判定“占据”的阈值
    - cost_at: 归一化 [0,1] 输出（若 grid 为 0/1 则直接返回 0/1）
    """

    def __init__(
        self,
        grid: np.ndarray,
        resolution: float,
        origin: Tuple[float, float] = (0.0, 0.0),
        occ_thresh: float = 0.5,
        treat_out_of_bounds_as_occupied: bool = True,
    ):
        assert grid.ndim == 2, "grid must be 2D"
        self._grid = grid.astype(float, copy=False)
        self._res = float(resolution)
        self._origin = (float(origin[0]), float(origin[1]))
        self._occ_thresh = float(occ_thresh)
        self._oob_occ = bool(treat_out_of_bounds_as_occupied)

    # ---- properties ----
    @property
    def resolution(self) -> float:
        return self._res

    @property
    def origin(self) -> Tuple[float, float]:
        return self._origin

    @property
    def size(self) -> Tuple[int, int]:
        return self._grid.shape  # (H, W)

    # ---- coordinate transforms ----
    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        x0, y0 = self._origin
        j = int(np.floor((x - x0) / self._res))
        i = int(np.floor((y - y0) / self._res))
        return (i, j)

    def cell_to_world(self, i: int, j: int) -> Tuple[float, float]:
        x0, y0 = self._origin
        x = x0 + (j + 0.5) * self._res
        y = y0 + (i + 0.5) * self._res
        return (x, y)

    # ---- helpers ----
    def _in_bounds(self, i: int, j: int) -> bool:
        H, W = self._grid.shape
        return (0 <= i < H) and (0 <= j < W)

    def _occupied_ij(self, i: int, j: int) -> bool:
        if not self._in_bounds(i, j):
            return self._oob_occ
        return self._grid[i, j] > self._occ_thresh

    # ---- queries ----
    def is_occupied(self, x: float, y: float) -> bool:
        i, j = self.world_to_cell(x, y)
        return self._occupied_ij(i, j)

    def cost_at(self, x: float, y: float) -> float:
        i, j = self.world_to_cell(x, y)
        if not self._in_bounds(i, j):
            return 1.0 if self._oob_occ else 0.0
        val = self._grid[i, j]
        gmax = self._grid.max()
        if gmax <= 1.0:
            return float(val)
        return float(val / gmax)

    # ---- raycast (Bresenham 风格) ----
    def raycast_free(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        i0, j0 = self.world_to_cell(x0, y0)
        i1, j1 = self.world_to_cell(x1, y1)

        di = i1 - i0
        dj = j1 - j0
        si = 1 if di >= 0 else -1
        sj = 1 if dj >= 0 else -1
        di = abs(di)
        dj = abs(dj)

        i, j = i0, j0
        if dj > di:
            err = dj / 2
            while j != j1:
                if self._occupied_ij(i, j):
                    return False
                err -= di
                if err < 0:
                    i += si
                    err += dj
                j += sj
            return not self._occupied_ij(i, j)
        else:
            err = di / 2
            while i != i1:
                if self._occupied_ij(i, j):
                    return False
                err -= dj
                if err < 0:
                    j += sj
                    err += di
                i += si
            return not self._occupied_ij(i, j)

    # ---- inflate （圆盘膨胀）----
    def inflate(self, radius: float) -> None:
        H, W = self._grid.shape
        r_cells = int(np.ceil(radius / self._res))
        if r_cells <= 0:
            return

        # 圆形结构元素
        offsets = []
        for di in range(-r_cells, r_cells + 1):
            for dj in range(-r_cells, r_cells + 1):
                if di * di + dj * dj <= r_cells * r_cells:
                    offsets.append((di, dj))
        offsets = np.asarray(offsets, dtype=int)

        occ = np.argwhere(self._grid > self._occ_thresh)
        if occ.size == 0:
            return

        inflated = self._grid.copy()
        vmax = max(self._grid.max(), 1.0)
        for (i, j) in occ:
            neigh = offsets + np.array([i, j])
            mask = (
                (0 <= neigh[:, 0]) & (neigh[:, 0] < H) &
                (0 <= neigh[:, 1]) & (neigh[:, 1] < W)
            )
            neigh = neigh[mask]
            inflated[neigh[:, 0], neigh[:, 1]] = vmax
        self._grid = inflated

    # ---- optional ----
    def update_grid(self, grid: np.ndarray) -> None:
        assert grid.shape == self._grid.shape
        self._grid = grid.astype(float, copy=False)

    def set_origin(self, x0: float, y0: float) -> None:
        self._origin = (float(x0), float(y0))

    def set_resolution(self, res: float) -> None:
        self._res = float(res)

    def clone(self) -> "NumpyCostmap2D":
        return NumpyCostmap2D(
            grid=self._grid.copy(),
            resolution=self._res,
            origin=self._origin,
            occ_thresh=self._occ_thresh,
            treat_out_of_bounds_as_occupied=self._oob_occ,
        )
    
    def add_rectangle(self, x_min: float, y_min: float, x_max: float, y_max: float, value: float = 1.0):
        """在物理坐标系中添加矩形障碍"""
        i_min, j_min = self.world_to_cell(x_min, y_min)
        i_max, j_max = self.world_to_cell(x_max, y_max)
        H, W = self._grid.shape
        i_min, i_max = max(0, min(H-1, i_min)), max(0, min(H-1, i_max))
        j_min, j_max = max(0, min(W-1, j_min)), max(0, min(W-1, j_max))
        self._grid[i_min:i_max+1, j_min:j_max+1] = value

    def add_circle(self, x_c: float, y_c: float, radius: float, value: float = 1.0):
        """在物理坐标系中添加圆形障碍"""
        i_c, j_c = self.world_to_cell(x_c, y_c)
        r_cells = int(np.ceil(radius / self._res))
        H, W = self._grid.shape
        for di in range(-r_cells, r_cells + 1):
            for dj in range(-r_cells, r_cells + 1):
                if di*di + dj*dj <= r_cells*r_cells:
                    i, j = i_c + di, j_c + dj
                    if 0 <= i < H and 0 <= j < W:
                        self._grid[i, j] = value

