# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import math
import numpy as np

import torch
import torch.nn as nn

from .base_planner import (
    BasePlanner, WorldState, TaskSpec, Plan, PlanResult, PlanStatus, EgoState
)


# ============================================================
# 2) 纯 NN 的 AIPlanner —— 同时支持“动作序列”和“相对轨迹”两种输出
# ============================================================
class AIPlanner(BasePlanner, nn.Module):
    def __init__(
        self,
        *,
        policy: Optional[nn.Module],
        device: Optional[torch.device] = None,
        action_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.0, 1.0), (-2.0, 2.0)),
        #（保留）环形代价参数（当前未作为输入）
        obs_ring_dirs: int = 12,
        obs_ring_radius: float = 0.6,
        # 激光特征配置
        lidar_n_sectors: int = 30,
        lidar_max_range: float = 8.0,
        lidar_samples_per_sector: int = 3,
        lidar_sector_spread: float = 0.8,
        # ✨ 新增：policy 输出类型 & 序列长度
        policy_output: str = "controls",   # "controls" 或 "traj_rel"
        seq_len: Optional[int] = None,     # 默认用 horizon/dt
        config: Optional[Dict[str, Any]] = None,
        **base_kwargs,
    ) -> None:
        BasePlanner.__init__(self, name="AIPlanner", config=config, **base_kwargs)
        nn.Module.__init__(self)

        self.device = device or torch.device("cpu")

        # ====== 激光参数 ======
        self.lidar_n_sectors = int(lidar_n_sectors)
        self.lidar_max_range = float(lidar_max_range)
        self.lidar_samples_per_sector = int(max(1, lidar_samples_per_sector))
        self.lidar_sector_spread = float(np.clip(lidar_sector_spread, 0.0, 1.0))

        # ====== 输入维度：5 基础 + lidar 扇区 ======
        # 基础 5 维: [dx, dy, dist, heading_err, ego_v]
        self.input_dim = 5 + self.lidar_n_sectors

        # ====== 策略网络（外部提供；未提供则延迟到 plan() 报错） ======
        self.policy = policy
        self._policy_ready = self.policy is not None
        if self._policy_ready:
            self.policy.to(self.device)

        # ====== 动作范围（注册为 buffer，便于保存/迁移） ======
        (self.v_min, self.v_max), (self.w_min, self.w_max) = action_bounds
        self.register_buffer("act_low",  torch.tensor([self.v_min, self.w_min], dtype=torch.float32))
        self.register_buffer("act_high", torch.tensor([self.v_max, self.w_max], dtype=torch.float32))

        # ====== rollout 参数 ======
        self.dt = float(self.config.get("dt", 0.1))
        self.horizon = float(self.config.get("horizon", 1.5))  # 秒
        self.max_steps = max(1, int(self.horizon / self.dt))

        # ====== 输出形态配置 ======
        self.policy_output = policy_output
        self.seq_len = int(seq_len) if seq_len is not None else self.max_steps

    # ----------------- 工具函数 -----------------
    @staticmethod
    def _angle_wrap(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _controls_from_traj_world(self, xs: np.ndarray, ys: np.ndarray, yaws: np.ndarray) -> np.ndarray:
        
        # 由世界系轨迹 (T,) 通过有限差分得到控制 (T,2): v,w，并限幅到物理范围。
        
        T = len(xs)
        ctrls = np.zeros((T, 2), dtype=np.float32)
        dt = self.dt
        for t in range(T - 1):
            dx = xs[t + 1] - xs[t]
            dy = ys[t + 1] - ys[t]
            ds = math.hypot(dx, dy)
            v = ds / dt
            dyaw = self._angle_wrap(yaws[t + 1] - yaws[t])
            w = dyaw / dt
            ctrls[t, 0] = float(np.clip(v, self.v_min, self.v_max))
            ctrls[t, 1] = float(np.clip(w, self.w_min, self.w_max))
        if T > 1:
            ctrls[-1] = ctrls[-2]
        return ctrls

    def _apply_rel_traj(self, ego0: EgoState, rel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        rel: (T, 2) 或 (T, 3)，自车坐标系相对增量 (dx, dy[, dyaw])。
        返回世界系 (xs, ys, yaws)，长度 T。
        """
        T = rel.shape[0]
        xs = np.zeros(T, np.float32)
        ys = np.zeros(T, np.float32)
        yaws = np.zeros(T, np.float32)
        x, y, yaw = ego0.x, ego0.y, ego0.yaw
        for t in range(T):
            dx_body, dy_body = float(rel[t, 0]), float(rel[t, 1])
            # 车体系 -> 世界系
            x += math.cos(yaw) * dx_body - math.sin(yaw) * dy_body
            y += math.sin(yaw) * dx_body + math.cos(yaw) * dy_body
            if rel.shape[1] >= 3:
                yaw += float(rel[t, 2])
            xs[t], ys[t], yaws[t] = x, y, yaw
        return xs, ys, yaws

    # --------- 栅格DDA：返回最近障碍距离（米）；无命中则返回 max_range ----------
    def _raycast_distance(self, cm, x0: float, y0: float, ang: float, max_range: float) -> float:
        res = float(cm.resolution)
        H, W = cm.size
        i, j = cm.world_to_cell(x0, y0)

        if not (0 <= i < H and 0 <= j < W):
            return 0.0
        if cm.is_occupied(x0, y0):
            return 0.0

        dx = math.cos(ang); dy = math.sin(ang)
        dx = dx if abs(dx) > 1e-12 else 1e-12
        dy = dy if abs(dy) > 1e-12 else 1e-12

        x = (x0 - cm.origin[0]) / res
        y = (y0 - cm.origin[1]) / res

        step_x = 1 if dx > 0 else -1
        step_y = 1 if dy > 0 else -1

        t_max_x = (((j + (step_x > 0)) + 0.5) - x) / dx
        t_max_y = (((i + (step_y > 0)) + 0.5) - y) / dy
        t_delta_x = step_x / dx
        t_delta_y = step_y / dy

        max_steps = int(max_range / res) + 1
        for _ in range(max_steps):
            if abs(t_max_x) < abs(t_max_y):
                j += step_x
                t_hit = t_max_x
                t_max_x += t_delta_x
            else:
                i += step_y
                t_hit = t_max_y
                t_max_y += t_delta_y

            if i < 0 or i >= H or j < 0 or j >= W:
                return min(max_range, abs(t_hit) * res)

            x_hit, y_hit = cm.cell_to_world(i, j)
            if cm.is_occupied(x_hit, y_hit):
                return min(max_range, abs(t_hit) * res)

        return max_range

    # --------- 生成 “扇区最近障碍距离” 特征，归一化到 [0,1] ----------
    def _lidar_sector_feats(self, ego: EgoState) -> List[float]:
        cm = self.require_costmap()
        full = 2.0 * math.pi
        sector_w = full / self.lidar_n_sectors

        if self.lidar_samples_per_sector <= 1:
            offsets = (0.0,)
        else:
            half_span = (sector_w * self.lidar_sector_spread) / 2.0
            idxs = np.linspace(-1.0, 1.0, self.lidar_samples_per_sector)
            offsets = tuple(float(t * half_span) for t in idxs)

        feats: List[float] = []
        for s in range(self.lidar_n_sectors):
            center_ang = (-math.pi + (s + 0.5) * sector_w) + ego.yaw  # 车体朝向对齐
            d_min = self.lidar_max_range
            for off in offsets:
                ang = center_ang + off
                d = self._raycast_distance(cm, ego.x, ego.y, ang, self.lidar_max_range)
                if d < d_min:
                    d_min = d
                    if d_min <= cm.resolution:  # 已非常近，提前结束
                        break
            feats.append(d_min / self.lidar_max_range)
        return feats

    # ----------------- 构造单步观测特征 -----------------
    def _build_obs(self, ego: EgoState, goal_xy: Tuple[float, float]) -> np.ndarray:
        """
        构造观测：
        - 将目标相对位置从世界系转换到自车系
        - 在自车系下计算距离与朝向误差
        - 拼接激光扇区特征
        返回:
        obs = [dx_car, dy_car, dist, heading_err, ego.v] + lidar_feats
        """
        assert goal_xy is not None
        gx, gy = goal_xy

        # 世界系相对位置
        dx_w = gx - ego.x
        dy_w = gy - ego.y

        # 世界系 -> 自车系（车头为 +x，左为 +y）
        cy = math.cos(ego.yaw)
        sy = math.sin(ego.yaw)
        dx_car =  cy * dx_w + sy * dy_w
        dy_car = -sy * dx_w + cy * dy_w

        # 在自车系下计算距离与朝向误差
        dist = math.hypot(dx_car, dy_car)
        if dist > 1e-6:
            heading_err = math.atan2(dy_car, dx_car)  # 目标方向相对车头的夹角 ∈ (-π, π]
        else:
            heading_err = 0.0  # 目标就在当前点附近，误差置 0

        # 扇区“激光雷达”特征（长度 = lidar_n_sectors，已归一化到 [0,1]）
        feats = self._lidar_sector_feats(ego)

        # 组装观测向量
        obs = np.array([dx_car, dy_car, dist, heading_err, ego.v] + feats, dtype=np.float32)
        return obs


    # ----------------- [-1,1] → 物理动作范围 -----------------
    def _map_action(self, a_unit: torch.Tensor) -> torch.Tensor:
        a = (a_unit + 1.0) * 0.5 * (self.act_high - self.act_low) + self.act_low
        a = torch.clamp(a, min=self.act_low, max=self.act_high)
        return a

    # ----------------- 规划入口 -----------------
    def plan(self, world: WorldState, task: TaskSpec) -> PlanResult:
        if task.goal_xy is None:
            return PlanResult(PlanStatus.INVALID_INPUT, msg="goal_xy required")

        if not self._policy_ready:
            return PlanResult(PlanStatus.INVALID_INPUT, msg="policy is None")

        # 若 world 提供了临时 costmap，更新引用
        if getattr(world, "costmap", None) is not None:
            self.costmap = world.costmap

        # 起点/终点鲁棒修复
        sx, sy = self.project_to_free_space(world.ego.x, world.ego.y)
        gx, gy = self.project_to_free_space(*task.goal_xy)
        ego0 = EgoState(
            t=world.ego.t, x=sx, y=sy, yaw=world.ego.yaw,
            v=world.ego.v, a=world.ego.a, extra=world.ego.extra
        )

        # === 观测 ===
        obs_np = self._build_obs(ego0, (gx, gy))[None, :]   # [1, D]
        obs = torch.from_numpy(obs_np).to(self.device)

        # === 调 policy 并统一成控制序列 ctrls:(T,2) ===
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                out = self.policy(obs)

                if self.policy_output == "controls":
                    # 允许 [1,2]（单步）或 [1,T,2]
                    if out.dim() == 2:       # [1,2]
                        out = out.unsqueeze(1)
                    assert out.shape[0] == 1 and out.shape[2] == 2, \
                        f"controls expects [1,T,2], got {tuple(out.shape)}"
                    if out.shape[1] != self.seq_len:
                        # 若长度不一致，按需要截断/重复到 seq_len
                        if out.shape[1] > self.seq_len:
                            out = out[:, :self.seq_len, :]
                        else:
                            reps = self.seq_len // out.shape[1] + 1
                            out = out.repeat(1, reps, 1)[:, :self.seq_len, :]
                    a_unit = out.squeeze(0)                 # [T,2] in [-1,1]
                    a = self._map_action(a_unit)            # [T,2]
                    ctrls = a.cpu().numpy().astype(np.float32)

                elif self.policy_output == "traj_rel":
                    # 允许 [1,T,2] 或 [1,T,3]
                    if out.dim() == 2:   # 兼容 [1,2/3] 单步：扩成序列
                        out = out.unsqueeze(1)
                    assert out.shape[0] == 1 and out.shape[2] in (2, 3), \
                        f"traj_rel expects [1,T,2/3], got {tuple(out.shape)}"
                    if out.shape[1] != self.seq_len:
                        if out.shape[1] > self.seq_len:
                            out = out[:, :self.seq_len, :]
                        else:
                            reps = self.seq_len // out.shape[1] + 1
                            out = out.repeat(1, reps, 1)[:, :self.seq_len, :]
                    rel = out.squeeze(0).cpu().numpy().astype(np.float32)  # (T,2/3)
                    xs, ys, yaws = self._apply_rel_traj(ego0, rel)
                    ctrls = self._controls_from_traj_world(xs, ys, yaws)   # (T,2)

                else:
                    return PlanResult(PlanStatus.INVALID_INPUT, msg=f"unknown policy_output: {self.policy_output}")

        except NotImplementedError as e:
            return PlanResult(PlanStatus.INVALID_INPUT, msg=str(e))
        finally:
            if was_training:
                self.train()

        # === rollout（若无 dynamics，用简易积分） ===
        T = min(len(ctrls), self.max_steps)
        traj = np.zeros((T, 4), dtype=np.float32)  # x,y,yaw,v
        s = ego0
        collided = False
        for t in range(T):
            v, w = float(ctrls[t, 0]), float(ctrls[t, 1])
            if self.dynamics is None:
                s = EgoState(
                    t=s.t + self.dt,
                    x=s.x + v * math.cos(s.yaw) * self.dt,
                    y=s.y + v * math.sin(s.yaw) * self.dt,
                    yaw=s.yaw + w * self.dt,
                    v=v, a=0.0, extra=s.extra
                )
            else:
                s = self.dynamics.step(s, (v, w), self.dt)
            traj[t] = [s.x, s.y, s.yaw, v]
            if not self.point_free(s.x, s.y, s.yaw):
                collided = True
                break

        T_eff = t if collided else T
        plan = Plan(
            traj_xyv=traj[: max(1, T_eff), :],
            controls=ctrls[: max(1, T_eff), :],
            info={"algo": "ai", "policy_output": self.policy_output, "T": int(max(1, T_eff))}
        )
        if collided:
            return PlanResult(PlanStatus.FAIL_OPT, plan=plan, msg="collision in rollout")
        return PlanResult(status=PlanStatus.OK, plan=plan, cost=None, runtime_ms=0.0)
