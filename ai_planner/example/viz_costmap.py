# ai_planner/example/viz_costmap.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np
from costmap.numpy_costmap import NumpyCostmap2D

def main():
    # 构造一个简单地图
    H, W = 200, 200
    grid = np.zeros((H, W))
    grid[30:40, 30:70] = 1.0   # 障碍条带

    cm = NumpyCostmap2D(grid, resolution=0.1, origin=(0, 0))

    # 可视化
    plt.figure(figsize=(6,6))
    plt.imshow(cm._grid, cmap="gray_r", origin="lower")  
    plt.title("Costmap")
    plt.xlabel("X (cells)")
    plt.ylabel("Y (cells)")
    plt.colorbar(label="Cost value")
    plt.show()

if __name__ == "__main__":
    main()
