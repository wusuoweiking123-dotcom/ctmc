import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import minimize

# 非均匀网格生成函数
def generate_nonuniform_grid(data, num_bins, alpha):
    x_min, x_max = np.min(data), np.max(data)
    margin = 0.1 * (x_max - x_min)
    x_min -= margin
    x_max += margin
    x_midpoint = (x_max + x_min) / 2
    c1 = np.arcsinh((x_max - x_midpoint) / alpha)
    c2 = np.arcsinh((x_min - x_midpoint) / alpha)

    weights = c2 * np.linspace(0, 1, num_bins) + c1 * (1 - np.linspace(0, 1, num_bins))
    grid = x_midpoint + alpha * np.sinh(weights)
    return grid

# 1. Simulate CEV process data
def simulate_cev(sigma, beta, T, N, f0=1):
    dt = T / N
    f = np.zeros(N)
    f[0] = f0
    np.random.seed(42)

    for i in range(1, N):
        drift_term = -0.5 * sigma ** 2 * (f[i - 1]/ f0) ** (2 * beta) * dt
        diffusion_term = sigma * (f[i - 1]/ f0) ** beta * np.sqrt(dt) * np.random.randn()
        f[i] = f[i - 1] * np.exp(drift_term + diffusion_term)
        f[i] = max(f[i], 1e-10)
    return f, dt

# 数据离散化（映射到网格索引）
def discretize_data(data, bins):
    indices = np.digitize(data, bins) - 1
    indices = np.clip(indices, 0, len(bins) - 2)
    return indices

# 构造 Q 矩阵（适应非均匀格）
def theoretical_Q(grid, sigma, beta,f0):
    h = grid[1] - grid[0]
    N = len(grid)
    Q = np.zeros((N, N))

    for i, x in enumerate(grid):
        if x <= 0:
            continue
        common_factor = -(grid[i] / f0) ** (2 * beta) * x ** 2 * sigma ** 2

        if i > 0:
            if i < N - 1:
                denominator = grid[i + 1] * grid[i] + grid[i + 1] * grid[i - 1]  - grid[i] * grid[i - 1] - grid[i - 1]**2
            else:
                denominator = 2 * (grid[i]-grid[i-1]) ** 2
            if abs(denominator) > 1e-12:
                Q[i, i - 1] = max(common_factor / denominator, 0)

        if i < N - 1:
            if i > 0:
                denominator = (grid[i + 1] - grid[i - 1]) * (grid[i] - grid[i + 1])
            else:
                denominator = 2 * (grid[i+1]-grid[i]) ** 2
            if abs(denominator) > 1e-12:
                Q[i, i + 1] = max(common_factor / denominator, 0)

        Q[i, i] = -np.sum(Q[i])

    return Q


# 似然函数
def log_likelihood(Q, state_seq, dt):
    P = expm(Q * dt)
    eps = 1e-12
    loglik = 0.0
    for i in range(len(state_seq) - 1):
        s1, s2 = state_seq[i], state_seq[i + 1]
        prob = P[s1, s2]
        loglik += np.log(prob + eps)
    return -loglik

# 修正的参数拟合函数
def fit_params(state_seq, grid, dt, f0):
    # 直接使用grid，不计算centers
    def loss(params):
        sigma, beta = params
        Q = theoretical_Q(grid, sigma, beta, f0)
        return log_likelihood(Q, state_seq, dt)

    res = minimize(
        loss,
        x0=[0.3, -0.5],
        bounds=[(0.01, 1.0), (-2.0, 1.0)]
    )
    return res.x

# 主函数
def main():
    sigma_true, beta_true = 0.14, -0.6
    T, N = 50, 12500
    f0 = 1
    # bins_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,32,34,36,38,40]
    # bins_list = [30,40, 50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270]
    bins_list = [30, 50, 100, 200, 300, 400, 500, 600, 700, 800]

    alpha = 0.2


    result_table = []

    for num_bins in bins_list:
        data, dt = simulate_cev(sigma=sigma_true, beta=beta_true, T=T, N=N)
        grid = generate_nonuniform_grid(data, num_bins + 1, alpha)
        state_seq = discretize_data(data, grid)
        est_sigma, est_beta = fit_params(state_seq, grid[:-1], dt, f0)

        param_true = np.array([sigma_true, beta_true])
        param_est = np.array([ est_sigma, est_beta])
        mae = np.mean(np.abs(param_true - param_est))
        mse = np.mean((param_true - param_est) ** 2)

        result_table.append({
            'bins': num_bins,
            'sigma': est_sigma,
            'beta': est_beta,
            'mae': mae,
            'mse': mse
        })

    print("\nParameter Estimation Results:")
    for row in result_table:
        print(f"num_bins={row['bins']:3d} |σ={row['sigma']:.4f} | β={row['beta']:.4f} | MAE={row['mae']:.5f} | MSE={row['mse']:.5f}")

    print(f"MSE for NonUniform discretization is equal to {np.mean(mse):.6f}")

    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    bins = [row['bins'] for row in result_table]
    sigma_vals = [row['sigma'] for row in result_table]
    beta_vals = [row['beta'] for row in result_table]

    axs[0].plot(bins, sigma_vals, marker='s', label='Estimated σ')
    axs[0].axhline(sigma_true, color='r', linestyle='--', label='True σ')
    axs[0].set_ylabel('σ')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(bins, beta_vals, marker='^', label='Estimated β')
    axs[1].axhline(beta_true, color='r', linestyle='--', label='True β')
    axs[1].set_ylabel('β')
    axs[1].set_xlabel('Number of Bins')
    axs[1].legend()
    axs[1].grid(True)

    plt.suptitle('Parameter Estimation vs. Number of Bins (Non-uniform Grid)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
