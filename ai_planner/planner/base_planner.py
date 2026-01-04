# -*- coding: utf-8 -*-
"""
Abstract base class for planners (grid-map version).

This module defines a unified interface for different planners
(e.g., AIPlanner, AStarPlanner, MPCPlanner) with costmap support.
Downstream code should depend on this interface only.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple, Protocol, runtime_checkable
import numpy as np
import time
import logging

# -----------------------------------------------------------------------------
# Typed protocols for grid-based local planning
# -----------------------------------------------------------------------------

@runtime_checkable
class Costmap2D(Protocol):
    """2D occupancy/ cost map for local collision avoidance."""
    @property
    def resolution(self) -> float: ...            # m/cell
    @property
    def origin(self) -> Tuple[float, float]: ...  # world (x0, y0) of cell (0,0)
    @property
    def size(self) -> Tuple[int, int]: ...        # (H, W)
    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]: ...
    def cell_to_world(self, i: int, j: int) -> Tuple[float, float]: ...
    def is_occupied(self, x: float, y: float) -> bool: ...
    def cost_at(self, x: float, y: float) -> float: ...
    def raycast_free(self, x0: float, y0: float, x1: float, y1: float) -> bool: ...
    def inflate(self, radius: float) -> None: ...

@runtime_checkable
class FootprintChecker(Protocol):
    """Robot footprint collision checker on a costmap."""
    def collides(self, x: float, y: float, yaw: float) -> bool: ...

@runtime_checkable
class Dynamics(Protocol):
    """e.g., differential drive kinematics"""
    def step(self, state: "EgoState", control, dt: float) -> "EgoState": ...
    def clamp_control(self, v: float, w: float) -> Tuple[float, float]: ...

@runtime_checkable
class Predictor(Protocol):
    """Target/actor prediction interface."""
    def predict(self, world: "WorldState", horizon: int, dt: float) -> Any: ...

@runtime_checkable
class Rules(Protocol):
    """Collision/constraints provider (optional)."""
    def build_obstacles(self, world: "WorldState") -> Any: ...
    def hard_constraints(self, world: "WorldState") -> Any: ...

# -----------------------------------------------------------------------------
# Core data structures for a planning step
# -----------------------------------------------------------------------------

@dataclass
class EgoState:
    t: float                      # timestamp [s]
    x: float; y: float; yaw: float
    v: float; a: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorldState:
    """Aggregated world state passed to planner at each step."""
    ego: EgoState
    actors: Any                                      # list/dict of other agents
    # 可选：也可仅由 BasePlanner 持有全局 costmap
    costmap: Optional[Costmap2D] = None
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskSpec:
    """Planning task/goal and constraints."""
    goal_xy: Optional[Tuple[float, float]] = None
    time_limit_s: Optional[float] = None
    hard: Dict[str, Any] = field(default_factory=dict)    # hard constraints
    soft: Dict[str, float] = field(default_factory=dict)  # weights for costs

@dataclass
class Plan:
    """Planner output in a unified format."""
    # discrete high-level plan (optional)
    waypoints: Optional[np.ndarray] = None     # [N, 2 or 3] (x,y[,yaw])
    # continuous trajectory for tracking (optional)
    traj_xyv: Optional[np.ndarray] = None      # [T, D] e.g., x,y,yaw,v
    controls: Optional[np.ndarray] = None      # [T, U] e.g., v, w
    info: Dict[str, Any] = field(default_factory=dict)

class PlanStatus(Enum):
    OK = auto()
    FAIL_SEARCH = auto()
    FAIL_OPT = auto()
    TIMEOUT = auto()
    INVALID_INPUT = auto()
    UNKNOWN = auto()

@dataclass
class PlanResult:
    status: PlanStatus
    plan: Optional[Plan] = None
    cost: Optional[float] = None
    runtime_ms: float = 0.0
    msg: str = ""

# -----------------------------------------------------------------------------
# Base Planner (grid-ready)
# -----------------------------------------------------------------------------

class BasePlanner(ABC):
    """
    Abstract planner interface (grid-map ready).

    Child classes must implement `plan()` at minimum. Optionally override
    lifecycle hooks: `reset`, `update_world`, `warm_start`, `close`, etc.
    """

    def __init__(
        self,
        name: str,
        *,
        costmap: Optional[Costmap2D] = None,
        footprint: Optional[FootprintChecker] = None,
        dynamics: Optional[Dynamics] = None,
        predictor: Optional[Predictor] = None,
        rules: Optional[Rules] = None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.name = name
        self.costmap = costmap
        self.footprint = footprint
        self.dynamics = dynamics
        self.predictor = predictor
        self.rules = rules
        self.config: Dict[str, Any] = config or {}
        self.rng = np.random.default_rng(self.config.get("seed", None))
        self.log = logger or logging.getLogger(name)
        self.log.setLevel(self.config.get("log_level", logging.INFO))

    # ---- configuration & lifecycle -----------------------------------------

    def set_config(self, **kwargs: Any) -> None:
        """Update planner hyper-parameters on the fly."""
        self.config.update(kwargs)

    def seed(self, seed: Optional[int]) -> None:
        self.rng = np.random.default_rng(seed)
        self.config["seed"] = seed

    def reset(self) -> None:
        """Clear any internal state between episodes."""
        pass

    def warm_start(self, last_plan: Optional[Plan]) -> None:
        """Provide previous step's plan to accelerate next planning."""
        pass

    def update_world(self, world: WorldState) -> None:
        """
        Optional hook to cache heavy data from world state.
        This also allows updating costmap reference per-step if the world provides one.
        """
        if world.costmap is not None:
            self.costmap = world.costmap

    def close(self) -> None:
        """Release resources if needed."""
        pass

    # ---- main API -----------------------------------------------------------

    @abstractmethod
    def plan(self, world: WorldState, task: TaskSpec) -> PlanResult:
        """
        Compute a plan for the given world and task.

        Returns:
            PlanResult: status + plan + diagnostics.
        """
        raise NotImplementedError

    # ---- helper for timing --------------------------------------------------

    def _timed(self, fn, *args, **kwargs) -> Tuple[Any, float]:
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        dt = (time.perf_counter() - t0) * 1e3
        return out, dt

    # ---- grid/costmap convenience helpers ----------------------------------

    # 这些辅助方法让派生类（A* / DWA / AI）写法更简洁

    def require_costmap(self) -> Costmap2D:
        if self.costmap is None:
            raise RuntimeError(f"[{self.name}] Costmap is required but not set.")
        return self.costmap

    def point_free(self, x: float, y: float, yaw: float = 0.0) -> bool:
        """点位是否无碰（考虑占据和 footprint）"""
        cm = self.require_costmap()
        if cm.is_occupied(x, y):
            return False
        if self.footprint is not None and self.footprint.collides(x, y, yaw):
            return False
        return True

    def ray_free(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """线段可视（中途没有穿越障碍）"""
        cm = self.require_costmap()
        return cm.raycast_free(x0, y0, x1, y1)

    def clamp_to_map(self, x: float, y: float) -> Tuple[float, float]:
        """把点钳制到地图边界内（用于数值稳健）"""
        cm = self.require_costmap()
        H, W = cm.size
        # 把世界坐标转格子，再限幅，再转回世界
        i, j = cm.world_to_cell(x, y)
        i = max(0, min(H - 1, i))
        j = max(0, min(W - 1, j))
        return cm.cell_to_world(i, j)

    def project_to_free_space(self, x: float, y: float, max_iter: int = 30) -> Tuple[float, float]:
        """
        若点落在障碍或边界外，尝试用小半径随机/邻域扫描把它投到最近的自由格。
        """
        cm = self.require_costmap()
        if self.point_free(x, y):
            return x, y
        res = cm.resolution
        for r in range(1, max_iter + 1):
            # 简单八连通扩张搜索
            for di in (-r, 0, r):
                for dj in (-r, 0, r):
                    if di == 0 and dj == 0: 
                        continue
                    xi = x + dj * res
                    yi = y + di * res
                    if self.point_free(xi, yi):
                        return xi, yi
        # 实在找不到就钳到边界中心点
        return self.clamp_to_map(x, y)

    def goal_visible(self, world: WorldState, task: TaskSpec) -> bool:
        """自车到目标点在栅格上是否无遮挡可视"""
        if task.goal_xy is None:
            return False
        cm = self.require_costmap()
        return cm.raycast_free(world.ego.x, world.ego.y, task.goal_xy[0], task.goal_xy[1])
