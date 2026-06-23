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

# 模拟 OU 过程数据
def simulate_ou(theta, sigma, mu, T, N, x0=0):
    dt = T / N
    x = np.zeros(N)
    x[0] = x0
    np.random.seed(51)
    for i in range(1, N):
        x[i] = x[i - 1] + theta * (mu - x[i - 1]) * dt + sigma * np.sqrt(dt) * np.random.randn()
    return x, dt

# 数据离散化（映射到网格索引）
def discretize_data(data, bins):
    indices = np.digitize(data, bins) - 1
    indices = np.clip(indices, 0, len(bins) - 2)
    return indices

# 构造 Q 矩阵（适应非均匀格）
def theoretical_Q(grid, theta, sigma, mu):
    N = len(grid)
    Q = np.zeros((N, N))
    for i in range(1, N - 1):
        x = grid[i]
        h_left = grid[i] - grid[i - 1]
        h_right = grid[i + 1] - grid[i]
        drift = theta * (mu - x)
        diff = sigma ** 2

        Q[i, i - 1] = max(diff / (h_left * (h_left + h_right)) - drift / (h_left + h_right),0)
        Q[i, i + 1] = max(diff / (h_right * (h_left + h_right)) + drift / (h_left + h_right),0)
        Q[i, i] = -Q[i, i - 1] - Q[i, i + 1]
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

# 拟合参数
def fit_params(state_seq, grid, dt):
    def loss(params):
        theta, sigma, mu = params
        Q = theoretical_Q(grid, theta, sigma, mu)
        return log_likelihood(Q, state_seq, dt)

    res = minimize(loss, x0=[0.5, 0.1, 0.1], bounds=[(0.001, 5), (0.001, 3), (-3, 3)])
    return res.x

# 主函数
def main():
    theta_true, sigma_true, mu_true = 0.7, 0.2, 0.3
    T, N = 100, 1000
    bins_list = [10, 20, 35, 50, 100, 150, 200]
    alpha = 0.2

    result_table = []

    for num_bins in bins_list:
        data, dt = simulate_ou(theta=theta_true, sigma=sigma_true, mu=mu_true, T=T, N=N)
        grid = generate_nonuniform_grid(data, num_bins + 1, alpha)
        state_seq = discretize_data(data, grid)
        est_theta, est_sigma, est_mu = fit_params(state_seq, grid[:-1], dt)

        param_true = np.array([theta_true, sigma_true, mu_true])
        param_est = np.array([est_theta, est_sigma, est_mu])
        mae = np.mean(np.abs(param_true - param_est))
        mse = np.mean((param_true - param_est) ** 2)

        result_table.append({
            'bins': num_bins,
            'theta': est_theta,
            'sigma': est_sigma,
            'mu': est_mu,
            'mae': mae,
            'mse': mse
        })

    print("\nParameter Estimation Results:")
    for row in result_table:
        print(f"num_bins={row['bins']:3d} | θ={row['theta']:.4f} | σ={row['sigma']:.4f} | μ={row['mu']:.4f} | MAE={row['mae']:.5f} | MSE={row['mse']:.5f}")

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    bins = [row['bins'] for row in result_table]
    theta_vals = [row['theta'] for row in result_table]
    sigma_vals = [row['sigma'] for row in result_table]
    mu_vals = [row['mu'] for row in result_table]

    axs[0].plot(bins, theta_vals, marker='o', label='Estimated θ')
    axs[0].axhline(theta_true, color='r', linestyle='--', label='True θ')
    axs[0].set_ylabel('θ')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(bins, sigma_vals, marker='s', label='Estimated σ')
    axs[1].axhline(sigma_true, color='r', linestyle='--', label='True σ')
    axs[1].set_ylabel('σ')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(bins, mu_vals, marker='^', label='Estimated μ')
    axs[2].axhline(mu_true, color='r', linestyle='--', label='True μ')
    axs[2].set_ylabel('μ')
    axs[2].set_xlabel('Number of Bins')
    axs[2].legend()
    axs[2].grid(True)

    plt.suptitle('Parameter Estimation vs. Number of Bins (Non-uniform Grid)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
