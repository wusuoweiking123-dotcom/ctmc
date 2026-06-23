import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import minimize

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


# 2. Discretize data into bins (给定的数据压成特定的状态)
def discretize_data(data, num_bins):
    x_min, x_max = np.min(data), np.max(data)
    bins = np.linspace(x_min, x_max, num_bins + 1)
    indices = np.digitize(data, bins) - 1
    indices = np.clip(indices, 0, num_bins - 1)
    return indices, bins


# 3. Construct empirical Q matrix (not used)
def build_Q_matrix(state_seq, num_states, dt):
    N_ij = np.zeros((num_states, num_states))
    T_i = np.zeros(num_states)

    for i in range(len(state_seq) - 1):
        cur, nxt = state_seq[i], state_seq[i + 1]
        N_ij[cur, nxt] += dt
        T_i[cur] += dt

    Q = np.zeros_like(N_ij)
    for i in range(num_states):
        if T_i[i] > 0:
            Q[i] = N_ij[i] / T_i[i]
            Q[i, i] = -np.sum(Q[i]) + Q[i, i]
    return Q


# 4. Construct theoretical Q matrix
def theoretical_Q(grid, sigma, beta,f0):
    h = grid[1] - grid[0]
    N = len(grid)
    Q = np.zeros((N, N))

    for i, x in enumerate(grid):
        if x <= 0:
            continue
        common_factor = -(grid[i] / f0) ** (2 * beta) * grid[i] ** 2 * sigma ** 2

        if i > 0:
            if i < N - 1:
                denominator = - grid[i - 1]**2 - grid[i + 1] * grid[i] + grid[i - 1] * grid[i + 1] + grid[i] * grid[i - 1]
            else:
                denominator = 2 * h ** 2
            if abs(denominator) > 1e-12:
                Q[i, i - 1] = max(common_factor / denominator, 0)

        if i < N - 1:
            if i > 0:  # Interior point
                denominator = (grid[i + 1] - grid[i - 1]) * (grid[i] - grid[i + 1])
            else:
                denominator = 2 * h ** 2
            if abs(denominator) > 1e-12:
                Q[i, i + 1] = max(common_factor / denominator, 0)

        Q[i, i] = -np.sum(Q[i])

    return Q


# 5. Log-likelihood function (with improved stability)
def log_likelihood(Q, state_seq, dt):
    # Check Q matrix validity
    if np.any(np.diag(Q) > 1e-10):  # Diagonal should be non-positive
        return 1e10
    if np.any(Q < 0) and not np.allclose(Q - np.diag(np.diag(Q)), np.maximum(Q - np.diag(np.diag(Q)), 0)):
        return 1e10

    try:
        P = expm(Q * dt)
        # Ensure probability matrix validity
        P = np.maximum(P, 1e-15)
        P = P / P.sum(axis=1, keepdims=True)
    except:
        return 1e10

    loglik = 0.0
    for i in range(len(state_seq) - 1):
        s1, s2 = state_seq[i], state_seq[i + 1]
        prob = P[s1, s2]
        if prob <= 0:
            return 1e10
        loglik += np.log(prob)

    return -loglik


# fit the parameters
def fit_params(state_seq, bins, dt,f0):
    centers = (bins[:-1] + bins[1:]) / 2

    def loss(params):
        sigma, beta = params
        Q = theoretical_Q(centers, sigma, beta,f0)
        return log_likelihood(Q, state_seq, dt)

    res = minimize(
        loss,
        x0=[0.2, -0.5],
        bounds=[(0.01, 1.0), (-2.0, 1.0)]
    )
    return res.x


def main():
    sigma_true, beta_true = 0.14, -0.6
    T, N = 50, 12500
    f0 = 1
    # bins_list= [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40]
    # bins_list = [30,40, 50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270]
    bins_list = [30, 50, 100, 200, 300, 400, 500, 600, 700, 800]
    result_table = []

    for num_bins in bins_list:
        data, dt = simulate_cev(sigma=sigma_true, beta=beta_true, T=T, N=N)
        state_seq, bins = discretize_data(data, num_bins=num_bins)
        est_sigma, est_beta = fit_params(state_seq, bins, dt,1)

        param_true = np.array([sigma_true, beta_true])
        param_est = np.array([est_sigma, est_beta])
        mae = np.mean(np.abs(param_true - param_est))
        mse = np.mean((param_true - param_est) ** 2)

        result_table.append({
            'bins': num_bins,
            'sigma': est_sigma,
            'beta': est_beta,
            'mae': mae,
            'mse': mse
        })

    # 输出误差表格
    print("\nParamater Estimation Results：")
    for row in result_table:
        print(
            f"num_bins={row['bins']:3d} | σ={row['sigma']:.4f} | β={row['beta']:.4f} | MAE={row['mae']:.5f} | MSE={row['mse']:.5f}")

    print(f"MSE for Uniform discretization is equal to {np.mean(mse):.6f}")


    # 绘图：参数估计 vs bin 数
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    bins = [row['bins'] for row in result_table]
    sigma_vals = [row['sigma'] for row in result_table]
    beta_vals = [row['beta'] for row in result_table]

    axs[0].plot(bins, sigma_vals, color='b', marker='s', label='Estimated σ')
    axs[0].axhline(sigma_true, color='r', linestyle='--', label='True σ')
    axs[0].set_ylabel('σ')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(bins, beta_vals, color='b', marker='^', label='Estimated β')
    axs[1].axhline(beta_true, color='r', linestyle='--', label='True β')
    axs[1].set_ylabel('μ')
    axs[1].set_xlabel('Number of Bins')
    axs[1].legend()
    axs[1].grid(True)

    print()

    plt.suptitle('Parameter Estimation vs. Number of Bins')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()