# ai_planner/example/sim_dynamic_obstacles.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import time
import numpy as np
import matplotlib.pyplot as plt

from costmap.numpy_costmap import NumpyCostmap2D
from costmap.dynamic import DynamicLayer, DynamicObstacle
from costmap.composite import CompositeCostmap2D

def main():
    # 1) 静态地图：一道水平墙
    H, W = 1200, 1200 #格子数
    # grid = np.zeros((H, W), dtype=float)
    # grid[60:80, 20:100] = 1.0
    # static_cm = NumpyCostmap2D(grid, resolution=0.1, origin=(0.0, 0.0))
    static_cm = NumpyCostmap2D(np.zeros((H,W)), resolution=0.1, origin=(0,0))
    static_cm.add_rectangle(20.0, 30.0, 50.0, 40.0)
    static_cm.add_circle(80, 80, 5)
    static_cm.inflate(radius=2)

    # 2) 动态层：两个会动的圆障碍
    dyn = DynamicLayer(shape=(H, W), resolution=0.1, origin=(0.0, 0.0))
    dyn.set_obstacles([
        DynamicObstacle(x=2.0, y=4.0, vx= 5, vy= 0.0, radius=0.25),
        DynamicObstacle(x=9.0, y=2.0, vx=4, vy= 4, radius=0.30),
    ])
    dyn.redraw()

    # 3) 合成 costmap
    cm = CompositeCostmap2D(static_cm, dyn.as_costmap())

    # 4) 可视化动画（matplotlib）
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(static_cm._grid, cmap="gray_r", origin="lower",
                   extent=[0, W*0.1, 0, H*0.1], vmin=0, vmax=1)
    scat = ax.scatter([], [], s=[], c='r', marker='o')
    ax.set_title("Static (gray) + Dynamic (red discs)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    plt.ion(); plt.show()

    dt = 0.1
    for k in range(1000):
        # 推进一步动态障碍
        dyn.step(dt)
        dyn.redraw()

        # 画静态层
        im.set_data(static_cm._grid)

        # 画动态障碍（红点大小 ~ 半径）
        xs = [o.x for o in dyn.objs]
        ys = [o.y for o in dyn.objs]
        sizes = [2000*(o.radius**2) for o in dyn.objs]
        scat.set_offsets(np.c_[xs, ys])
        scat.set_sizes(sizes)

        plt.pause(0.05)

    plt.ioff(); plt.show()

if __name__ == "__main__":
    main()
