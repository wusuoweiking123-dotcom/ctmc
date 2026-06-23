import numpy as np
import matplotlib.pyplot as plt

# OU过程样本生成（只用于展示采样分布）
def simulate_ou(theta, sigma, mu, T, N, x0=0):
    dt = T / N
    x = np.zeros(N)
    x[0] = x0
    np.random.seed(51)
    for i in range(1, N):
        x[i] = x[i-1] + theta * (mu - x[i-1]) * dt + sigma * np.sqrt(dt) * np.random.randn()
    return x

# 非均匀选点函数
def generate_nonuniform_grid(data, num_bins, alpha):
    x_min, x_max = np.min(data), np.max(data)
    x_midpoint = (x_max + x_min) / 2
    c1 = np.arcsinh((x_max - x_midpoint) / alpha)
    c2 = np.arcsinh((x_min - x_midpoint) / alpha)
    uniform_points = np.linspace(0, 1, num_bins)
    grid = x_midpoint + alpha * np.sinh(c2 * uniform_points + c1 * (1 - uniform_points))
    return grid

# 参数与数据生成
theta_true, sigma_true, mu_true = 1, 0.3, 0
T = 50
N = 1000
data = simulate_ou(theta_true, sigma_true, mu_true, T, N)

# 网格点数与 alpha
num_bins = 50
alpha = 0.15

# 均匀选点
uniform_grid = np.linspace(np.min(data), np.max(data), num_bins)

# 非均匀选点
nonuniform_grid = generate_nonuniform_grid(data, num_bins, alpha)

# 绘图：点状图 + 分层显示
plt.figure(figsize=(12, 6))

# 三层点图，不同 y 坐标展示
plt.plot(data, np.full_like(data, 3), 'k.', alpha=0.5, label="OU Samples (raw)")
plt.plot(uniform_grid, np.full_like(uniform_grid, 2), 'bo', label="Uniform Grid")
plt.plot(nonuniform_grid, np.full_like(nonuniform_grid, 1), 'ro', label="Non-uniform Grid (sinh)")

# 样式设置
plt.yticks([1, 2, 3], ["Non-uniform Grid", "Uniform Grid", "OU Samples"])
plt.xlabel("x")
plt.title("Grid Selection Comparison: OU Samples vs Uniform vs Non-uniform (Dots)")
plt.grid(True)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
