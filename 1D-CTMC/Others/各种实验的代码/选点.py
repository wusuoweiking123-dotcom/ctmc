import numpy as np
import matplotlib.pyplot as plt

def generate_nonuniform_grid(data, num_bins, alpha):
    x_min, x_max = np.min(data), np.max(data)
    x_midpoint = (x_max + x_min) / 2  # 中心点
    # 根据x的最小值和最大值计算控制参数c1, c2
    c1 = np.arcsinh((x_max - x_midpoint) / alpha)
    c2 = np.arcsinh((x_min - x_midpoint) / alpha)

    # 生成均匀分布的序列，表示网格的相对位置
    uniform_points = np.linspace(0, 1, num_bins)

    # 使用 sinh 函数转换这些点，产生非均匀分布
    grid = x_midpoint + alpha * np.sinh(c2 * uniform_points + c1 * (1 - uniform_points))

    return grid

# 测试代码
data = np.random.normal(0, 1, 1000)  # 示例数据
num_bins = 30  # 网格点数
alpha = 0.3  # 拉伸因子

# 生成非均匀格点
grid = generate_nonuniform_grid(data, num_bins, alpha)

# 可视化结果
plt.plot(grid, np.zeros_like(grid), 'o', label="Non-uniform Grid")
plt.xlabel("Grid Points")
plt.title("Non-uniform Grid Distribution Using Sinh Function")
plt.legend()
plt.grid(True)
plt.show()
