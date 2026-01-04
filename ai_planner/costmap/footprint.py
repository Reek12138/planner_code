# ai_planner/costmap/footprint.py
from __future__ import annotations
import numpy as np
from typing import Iterable, Tuple, List
from .base import BaseCostmap2D, FootprintChecker

class CircleFootprintChecker(FootprintChecker):
    """圆形足迹：简单好用，常用于差速/小车。"""
    def __init__(self, costmap: BaseCostmap2D, radius_m: float):
        self.cm = costmap
        self.radius = float(radius_m)

    def collides(self, x: float, y: float, yaw: float = 0.0) -> bool:
        r_cells = int(np.ceil(self.radius / self.cm.resolution))
        i0, j0 = self.cm.world_to_cell(x, y)
        for di in range(-r_cells, r_cells + 1):
            for dj in range(-r_cells, r_cells + 1):
                if di * di + dj * dj <= r_cells * r_cells:
                    i = i0 + di
                    j = j0 + dj
                    # 越界按占据更安全
                    H, W = self.cm.size
                    if not (0 <= i < H and 0 <= j < W):
                        return True
                    # 访问内部方法需谨慎；也可用 world 坐标再 is_occupied
                    xw, yw = self.cm.cell_to_world(i, j)
                    if self.cm.is_occupied(xw, yw):
                        return True
        return False


class PolygonFootprintChecker(FootprintChecker):
    """多边形足迹：适合长方体/组合轮廓（近似）。"""
    def __init__(self, costmap: BaseCostmap2D, vertices_m: Iterable[Tuple[float, float]]):
        """
        vertices_m: 局部车体坐标系下的多边形顶点（逆/顺时针）
        """
        self.cm = costmap
        self.verts_local = np.asarray(vertices_m, dtype=float)

    def collides(self, x: float, y: float, yaw: float = 0.0) -> bool:
        # 1) 将多边形从局部坐标系变换到世界坐标
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        verts_world = (self.verts_local @ R.T) + np.array([x, y])

        # 2) 取包围盒，枚举覆盖的格子并做点在多边形内 + 占据检查
        xs, ys = verts_world[:, 0], verts_world[:, 1]
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        # 扫描栅格中心点
        i_min, j_min = self.cm.world_to_cell(x_min, y_min)
        i_max, j_max = self.cm.world_to_cell(x_max, y_max)

        H, W = self.cm.size
        i_min = max(0, min(H - 1, i_min))
        i_max = max(0, min(H - 1, i_max))
        j_min = max(0, min(W - 1, j_min))
        j_max = max(0, min(W - 1, j_max))

        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                xw, yw = self.cm.cell_to_world(i, j)
                if point_in_polygon(xw, yw, verts_world):
                    if self.cm.is_occupied(xw, yw):
                        return True
        return False


def point_in_polygon(x: float, y: float, poly_xy: np.ndarray) -> bool:
    """射线法：点是否在多边形内（含边界）"""
    cnt = 0
    n = len(poly_xy)
    for i in range(n):
        x1, y1 = poly_xy[i]
        x2, y2 = poly_xy[(i + 1) % n]
        # 检查是否跨越水平射线
        if ((y1 > y) != (y2 > y)):
            xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x <= xinters:
                cnt += 1
    return (cnt % 2) == 1
