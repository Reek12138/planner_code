# ai_planner/costmap/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Iterable, Optional, Any, Protocol, runtime_checkable

class BaseCostmap2D(ABC):
    """
    抽象 2D Costmap 接口。任何具体实现都应遵循它。
    与 planner.BasePlanner 里的 Protocol 方法一一对应，并略有扩展。
    """

    # ---- 只读属性 ----
    @property
    @abstractmethod
    def resolution(self) -> float:
        """米/格"""
        ...

    @property
    @abstractmethod
    def origin(self) -> Tuple[float, float]:
        """世界坐标下 (x0, y0) 对应栅格 (0,0) 的左下角（或你的基准定义）"""
        ...

    @property
    @abstractmethod
    def size(self) -> Tuple[int, int]:
        """(H, W)"""
        ...

    # ---- 坐标转换 ----
    @abstractmethod
    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        ...

    @abstractmethod
    def cell_to_world(self, i: int, j: int) -> Tuple[float, float]:
        ...

    # ---- 查询 ----
    @abstractmethod
    def is_occupied(self, x: float, y: float) -> bool:
        ...

    @abstractmethod
    def cost_at(self, x: float, y: float) -> float:
        """返回归一化代价（推荐 0.0=自由，1.0=不可通行或高代价）"""
        ...

    @abstractmethod
    def raycast_free(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """两点间是否无遮挡（离散射线检测）"""
        ...

    @abstractmethod
    def inflate(self, radius: float) -> None:
        """按机器人半径/安全裕度对障碍进行膨胀"""
        ...

    # ---- 可选扩展（默认空实现/或抛异常）----
    def set_origin(self, x0: float, y0: float) -> None:
        raise NotImplementedError

    def set_resolution(self, res: float) -> None:
        raise NotImplementedError

    def update_grid(self, *args, **kwargs) -> None:
        """将传感器/上游更新写入底层栅格"""
        raise NotImplementedError

    def clone(self) -> "BaseCostmap2D":
        """返回一个浅/深拷贝（具体实现自行选择）"""
        raise NotImplementedError


# ---- Footprint Checker 的通用协议（可被 planner 引用）----
@runtime_checkable
class FootprintChecker(Protocol):
    def collides(self, x: float, y: float, yaw: float = 0.0) -> bool:
        ...
