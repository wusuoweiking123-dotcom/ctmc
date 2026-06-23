import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import minimize
import pandas as pd


# 1. Simulate CEV process data
def simulate_cev(sigma, beta, T, N, f0=1):
    dt = T / N
    f = np.zeros(N)
    f[0] = f0
    np.random.seed(21)

    for i in range(1, N):
        drift_term = -0.5 * sigma ** 2 * (f[i - 1] / f0) ** (2 * beta) * dt
        diffusion_term = sigma * (f[i - 1] / f0) ** beta * np.sqrt(dt) * np.random.randn()
        f[i] = f[i - 1] * np.exp(drift_term + diffusion_term)
        f[i] = max(f[i], 1e-10)
    return f, dt


# 2. Discretize data into bins (modified to support boundary extension)
def discretize_data(data, num_bins, boundary_multiplier=1.0):
    x_min = np.min(data)
    x_max = np.max(data) * boundary_multiplier  # Extend upper boundary by K times
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


# 4. Construct theoretical Q matrix(CEV process)
def theoretical_Q(grid, sigma, beta, f0):
    h = grid[1] - grid[0]
    N = len(grid)
    Q = np.zeros((N, N))

    for i, x in enumerate(grid):
        if x <= 0:
            continue
        common_factor = -(grid[i] / f0) ** (2 * beta) * grid[i] ** 2 * sigma ** 2

        if i > 0:
            if i < N - 1:
                denominator = grid[i - 1] * grid[i + 1] + grid[i] * grid[i - 1] - grid[i - 1] ** 2 - grid[i + 1] * grid[
                    i]
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


# 5. Log-likelihood function
def log_likelihood(Q, state_seq, dt):
    P = expm(Q * dt)
    eps = 1e-12
    loglik = 0.0

    for i in range(len(state_seq) - 1):
        s1, s2 = state_seq[i], state_seq[i + 1]
        prob = P[s1, s2]
        loglik += np.log(prob + eps)
    return -loglik


# 6. Fit parameters using likelihood
def fit_params_matrix(state_seq, bins, dt, f0):
    centers = (bins[:-1] + bins[1:]) / 2

    def loss(params):
        sigma, beta = params
        Q = theoretical_Q(centers, sigma, beta, f0)
        return log_likelihood(Q, state_seq, dt)

    res = minimize(
        loss,
        x0=[0.2, -0.5],
        bounds=[(0.01, 0.99), (-2.0, 1.0)]
    )
    return res.x


# 7. NEW: Boundary extension analysis function
def boundary_extension_analysis(data, dt, f0, sigma_true, beta_true, num_bins=5, k_values=None):
    """
    分析不同边界扩展倍数K对参数估计误差的影响

    Parameters:
    - data: CEV过程数据
    - dt: 时间步长
    - f0: 初始值
    - sigma_true, beta_true: 真实参数
    - num_bins: 状态数量
    - k_values: K倍数列表

    Returns:
    - results_df: 包含所有结果的DataFrame
    """
    if k_values is None:
        k_values = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]

    results = []

    print("开始边界扩展分析...")
    print("=" * 60)

    for k in k_values:
        print(f"\n分析边界扩展倍数 K = {k}")
        print("-" * 40)

        # 使用扩展边界离散化数据
        state_seq, bins = discretize_data(data, num_bins, boundary_multiplier=k)

        # 拟合参数
        try:
            est_sigma, est_beta = fit_params_matrix(state_seq, bins, dt, f0)

            # 计算误差
            sigma_error = abs(sigma_true - est_sigma)
            beta_error = abs(beta_true - est_beta)
            mae = (sigma_error + beta_error) / 2
            mse = (sigma_error ** 2 + beta_error ** 2) / 2

            # 边界信息
            x_min_original = np.min(data)
            x_max_original = np.max(data)
            x_max_extended = x_max_original * k

            result = {
                'K': k,
                'sigma_true': sigma_true,
                'beta_true': beta_true,
                'sigma_est': est_sigma,
                'beta_est': est_beta,
                'sigma_error': sigma_error,
                'beta_error': beta_error,
                'mae': mae,
                'mse': mse,
                'x_min': x_min_original,
                'x_max_original': x_max_original,
                'x_max_extended': x_max_extended,
                'num_bins': num_bins
            }

            results.append(result)

            print(f"真实参数: σ = {sigma_true}, β = {beta_true}")
            print(f"估计参数: σ̂ = {est_sigma:.4f}, β̂ = {est_beta:.4f}")
            print(f"参数误差: Δσ = {sigma_error:.4f}, Δβ = {beta_error:.4f}")
            print(f"MAE = {mae:.4f}, MSE = {mse:.4f}")
            print(f"边界范围: [{x_min_original:.4f}, {x_max_extended:.4f}]")

        except Exception as e:
            print(f"K = {k} 时估计失败: {e}")
            continue

    results_df = pd.DataFrame(results)
    return results_df


# 8. NEW: Visualization function for boundary analysis
def plot_boundary_analysis(results_df):
    """plot the result"""
    if results_df.empty:
        print("No result")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Error of parameter estimation vs K
    axes[0, 0].plot(results_df['K'], results_df['sigma_error'], 'bo-', label='σ error', linewidth=2, markersize=6)
    axes[0, 0].plot(results_df['K'], results_df['beta_error'], 'ro-', label='β error', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Boundary Extension Factor (K)')
    axes[0, 0].set_ylabel('Parameter Estimation Error')
    axes[0, 0].set_title('Parameter Errors vs Boundary Extension')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # MAE 和 MSE vs K
    axes[0, 1].plot(results_df['K'], results_df['mae'], 'go-', label='MAE', linewidth=2, markersize=6)
    axes[0, 1].plot(results_df['K'], results_df['mse'], 'mo-', label='MSE', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Boundary Extension Factor (K)')
    axes[0, 1].set_ylabel('Error Metric')
    axes[0, 1].set_title('MAE and MSE vs Boundary Extension')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 参数估计值 vs K
    axes[1, 0].plot(results_df['K'], results_df['sigma_est'], 'bo-', label='σ̂', linewidth=2, markersize=6)
    axes[1, 0].axhline(y=results_df['sigma_true'].iloc[0], color='b', linestyle='--', alpha=0.7, label='σ_true')
    axes[1, 0].plot(results_df['K'], results_df['beta_est'], 'ro-', label='β̂', linewidth=2, markersize=6)
    axes[1, 0].axhline(y=results_df['beta_true'].iloc[0], color='r', linestyle='--', alpha=0.7, label='β_true')
    axes[1, 0].set_xlabel('Boundary Extension Factor (K)')
    axes[1, 0].set_ylabel('Parameter Value')
    axes[1, 0].set_title('Parameter Estimates vs Boundary Extension')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 边界范围 vs K
    axes[1, 1].plot(results_df['K'], results_df['x_max_extended'], 'co-', label='Extended Upper Bound', linewidth=2,
                    markersize=6)
    axes[1, 1].axhline(y=results_df['x_max_original'].iloc[0], color='c', linestyle='--', alpha=0.7,
                       label='Original Upper Bound')
    axes[1, 1].set_xlabel('Boundary Extension Factor (K)')
    axes[1, 1].set_ylabel('Upper Boundary Value')
    axes[1, 1].set_title('Boundary Extension vs K')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Main function (enhanced)
def main():
    # True parameters
    sigma_true, beta_true = 0.2, -0.6
    T = 10
    N = 520
    f0 = 1

    # Simulate data
    data, dt = simulate_cev(sigma=sigma_true, beta=beta_true, T=T, N=N)

    print("原始功能演示:")
    print("=" * 50)

    # 原始功能：默认边界
    state_seq, bins = discretize_data(data, num_bins=5)
    print("Discrete state sequence:", state_seq[:10], "...")  # 只显示前10个
    print("State intervals:", bins)

    # Construct theoretical Q matrix (for reference, not used in new loss function)
    centers = (bins[:-1] + bins[1:]) / 2
    Q_theo = theoretical_Q(centers, sigma_true, beta_true, f0)
    print("Theoretical Q matrix shape:", Q_theo.shape)

    # Fit parameters using new loss function
    est_sigma, est_beta = fit_params_matrix(state_seq, bins, dt, f0)

    print(f"True parameters: σ = {sigma_true}, β = {beta_true}")
    print(f"Estimated parameters: σ̂ = {est_sigma:.4f}, β̂ = {est_beta:.4f}")

    # Parameter estimation errors
    param_true = np.array([sigma_true, beta_true])
    param_est = np.array([est_sigma, est_beta])
    mae = np.mean(np.abs(param_true - param_est))
    mse = np.mean((param_true - param_est) ** 2)

    print("\nError Analysis:")
    print(f"MAE (Mean Absolute Error): {mae:.6f}")
    print(f"MSE (Mean Squared Error): {mse:.6f}")

    # 新功能：边界扩展分析
    print("\n\n新功能：边界扩展分析")
    print("=" * 50)

    # 定义要测试的K值
    k_values = [1.0, 1.5, 2.0, 2.5, 3.0,3.5, 4.0,4.5, 5.0]

    # 执行边界扩展分析
    results_df = boundary_extension_analysis(
        data=data,
        dt=dt,
        f0=f0,
        sigma_true=sigma_true,
        beta_true=beta_true,
        num_bins=5,
        k_values=k_values
    )

    # 显示结果摘要
    if not results_df.empty:
        print("\n边界扩展分析结果摘要:")
        print("=" * 60)
        print(results_df[['K', 'sigma_est', 'beta_est', 'sigma_error', 'beta_error', 'mae', 'mse']].round(4))

        # 找到最佳K值
        best_mae_idx = results_df['mae'].idxmin()
        best_mse_idx = results_df['mse'].idxmin()

        print(f"\n最优结果:")
        print(f"最小MAE: K = {results_df.loc[best_mae_idx, 'K']}, MAE = {results_df.loc[best_mae_idx, 'mae']:.6f}")
        print(f"最小MSE: K = {results_df.loc[best_mse_idx, 'K']}, MSE = {results_df.loc[best_mse_idx, 'mse']:.6f}")

        # 绘制分析结果
        plot_boundary_analysis(results_df)

    # 原始绘图功能
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(data)) * dt, data, label='CEV Process', color='blue', alpha=0.6)

    # Map state indices to values in bins (使用原始边界)
    state_value = np.array([bins[s] for s in state_seq])
    plt.step(np.arange(len(data)) * dt, state_value, label='Markov Process', color='red', alpha=0.6)

    plt.title('CEV Process Trajectory with Discrete State Space (Original Boundary)')
    plt.xlabel('Time')
    plt.ylabel('F(t)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()