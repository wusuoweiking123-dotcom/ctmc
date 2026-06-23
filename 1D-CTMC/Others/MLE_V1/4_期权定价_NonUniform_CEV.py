import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import minimize


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


# 2. Discretize data into bins - 只添加一个简单的非均匀网格选项
def discretize_data(data, num_bins, use_nonuniform_grid=False, K=100):
    x_min, x_max = np.min(data), np.max(data)

    if use_nonuniform_grid:
        # 简单的非均匀网格：在K附近更密集
        # 左半部分
        left_end = K - 0.2 * K
        left_end = max(left_end, x_min)

        # 右半部分
        right_start = K + 0.2 * K
        right_start = min(right_start, x_max)

        # 分配bin数量
        n_left = num_bins // 4
        n_center = num_bins // 2
        n_right = num_bins - n_left - n_center

        # 创建bins
        bins_left = np.linspace(x_min, left_end, n_left + 1)
        bins_center = np.linspace(left_end, right_start, n_center + 1)
        bins_right = np.linspace(right_start, x_max, n_right + 1)

        # 组合bins，去除重复点
        bins = np.concatenate([bins_left[:-1], bins_center[:-1], bins_right])
    else:
        # 原来的均匀网格
        bins = np.linspace(x_min, x_max, num_bins + 1)

    indices = np.digitize(data, bins) - 1
    indices = np.clip(indices, 0, len(bins) - 2)
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


# 4. Construct theoretical Q matrix - 保持原来的逻辑，只做安全性改进
def theoretical_Q(grid, sigma, beta, f0):
    h = grid[1] - grid[0] if len(grid) > 1 else 1.0  # 安全性检查
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
def fit_params(state_seq, bins, dt, f0):
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


# 7. American option pricing using dynamic programming
def price_American_option(Q, centers, K, r, dt, M, option_type='put'):
    """
    使用动态规划算法定价美式期权
    Parameters:
    Q: 转移速率矩阵 (generator matrix)
    centers: 状态空间的中心点
    K: 行权价
    r: 无风险利率
    dt: 时间步长
    M: 时间步数
    option_type: 'put' 或 'call'
    """
    N = len(centers)

    # 转移概率矩阵 P(Δt) = exp(Q * Δt)
    P = expm(Q * dt)

    # 初始化价值矩阵 V[时间步, 状态]
    V = np.zeros((M + 1, N))

    # 即时行权价值函数 f(x)
    if option_type == 'put':
        exercise_value = np.maximum(K - centers, 0)
    else:
        exercise_value = np.maximum(centers - K, 0)

    # 边界条件：到期日的期权价值 V(T, X) = f(X)
    V[M, :] = exercise_value

    # 向后迭代：从到期日向前计算
    for i in range(M - 1, -1, -1):
        # 持有价值：e^(-r*dt) * P(dt) * V(t+1)
        continuation_value = np.exp(-r * dt) * P @ V[i + 1, :]

        # 美式期权价值：max(即时行权价值, 持有价值)
        V[i, :] = np.maximum(exercise_value, continuation_value)

    return V, exercise_value


def get_option_price_mean(option_values):
    return np.mean(option_values)


def interpolate_option_value(centers, option_values, target_price):
    """
    插值获取目标价格对应的期权价值
    """
    if target_price <= centers[0]:
        return option_values[0]
    if target_price >= centers[-1]:
        return option_values[-1]

    # 找到目标价格的左右邻居
    right_idx = np.searchsorted(centers, target_price)
    left_idx = right_idx - 1

    # 线性插值
    weight = (target_price - centers[left_idx]) / (centers[right_idx] - centers[left_idx])
    interpolated_value = option_values[left_idx] + weight * (option_values[right_idx] - option_values[left_idx])

    return interpolated_value


def get_closest_price_value(centers, option_values, target_price):
    """
    获取最接近目标价格的期权价值
    """
    closest_idx = np.argmin(np.abs(centers - target_price))
    return option_values[closest_idx]


def main():
    sigma_true = 0.14
    beta_true = -0.6
    T = 50
    M = 12500
    f0 = 100
    K = 100
    r = 0.05

    print("=== CEV美式期权定价算法 ===")
    print(f"真实参数: σ={sigma_true}, β={beta_true}")
    print(f"期货初始价格: {f0}, 行权价: {K}")

    # 步骤1：模拟CEV过程数据
    print("\n步骤1: 模拟CEV过程...")
    f, dt = simulate_cev(sigma_true, beta_true, T, M + 1, f0)
    print(f"模拟{len(f)}个价格点，时间步长: {dt:.4f}")

    # 步骤2：离散化数据 - 现在可以选择非均匀网格
    print("\n步骤2: 离散化状态空间...")
    num_bins = 800

    # 设置是否使用非均匀网格
    use_nonuniform = False  # 改为False可以使用原来的均匀网格

    state_indices, bins = discretize_data(f, num_bins, use_nonuniform_grid=use_nonuniform, K=K)
    centers = (bins[:-1] + bins[1:]) / 2
    print(f"状态空间范围: [{centers[0]:.2f}, {centers[-1]:.2f}]")
    print(f"状态数量: {len(centers)}")
    print(f"网格类型: {'非均匀网格' if use_nonuniform else '均匀网格'}")

    # 如果使用非均匀网格，显示网格密度信息
    if use_nonuniform:
        grid_spacing = np.diff(centers)
        k_idx = np.argmin(np.abs(centers - K))
        print(f"行权价K={K}附近的网格间距: {grid_spacing[max(0, k_idx - 5):min(len(grid_spacing), k_idx + 5)]}")
        print(f"最小网格间距: {np.min(grid_spacing):.4f}")
        print(f"最大网格间距: {np.max(grid_spacing):.4f}")

    # 步骤3：参数估计
    print("\n步骤3: 估计CEV参数...")
    sigma_fit, beta_fit = fit_params(state_indices, bins, dt, f0)
    print(f"拟合参数: σ={sigma_fit:.4f}, β={beta_fit:.4f}")
    print(f"参数误差: Δσ={abs(sigma_fit - sigma_true):.4f}, Δβ={abs(beta_fit - beta_true):.4f}")

    # 步骤4：构建CTMC转移速率矩阵
    print("\n步骤4: 构建CTMC转移速率矩阵...")
    Q = theoretical_Q(centers, sigma_fit, beta_fit, f0)
    print(f"转移速率矩阵Q的维度: {Q.shape}")

    # 检查Q矩阵的性质
    row_sums = np.sum(Q, axis=1)
    print(f"Q矩阵行和检查 (应该接近0): max|row_sum| = {np.max(np.abs(row_sums)):.6f}")

    # 步骤5：美式看跌期权定价
    print("\n步骤5: 美式看跌期权定价...")
    V_put, exercise_value_put = price_American_option(
        Q, centers, K, r, dt, M, option_type='put'
    )

    # 步骤6：美式看涨期权定价
    print("\n步骤6: 美式看涨期权定价...")
    V_call, exercise_value_call = price_American_option(
        Q, centers, K, r, dt, M, option_type='call'
    )

    # 步骤7：获取当前价格对应的期权价值 - 使用三种方法对比
    print(f"\n=== 期权价值提取方法对比 ===")

    # 方法1：取均值（你当前的方法）
    initial_put_value_mean = get_option_price_mean(V_put[0, :])
    initial_call_value_mean = get_option_price_mean(V_call[0, :])

    # 方法2：插值法
    initial_put_value_interp = interpolate_option_value(centers, V_put[0, :], f0)
    initial_call_value_interp = interpolate_option_value(centers, V_call[0, :], f0)

    # 方法3：最接近点
    initial_put_value_closest = get_closest_price_value(centers, V_put[0, :], f0)
    initial_call_value_closest = get_closest_price_value(centers, V_call[0, :], f0)

    print(f"美式看跌期权价格:")
    print(f"  均值方法: {initial_put_value_mean:.6f}")
    print(f"  插值方法: {initial_put_value_interp:.6f}")
    print(f"  最近点方法: {initial_put_value_closest:.6f}")

    print(f"美式看涨期权价格:")
    print(f"  均值方法: {initial_call_value_mean:.6f}")
    print(f"  插值方法: {initial_call_value_interp:.6f}")
    print(f"  最近点方法: {initial_call_value_closest:.6f}")

    # 理论价格对比
    vol_effective = sigma_true
    d1 = (np.log(f0 / K) + 0.5 * vol_effective ** 2 * T) / (vol_effective * np.sqrt(T))
    d2 = d1 - vol_effective * np.sqrt(T)

    european_put = max(K * np.exp(-r * T) * norm.cdf(-d2) - f0 * norm.cdf(-d1), 0)
    european_call = max(f0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), 0)

    print(f"\n=== 理论价格对比 ===")
    print(f"欧式看跌期权 (BS): {european_put:.6f}")
    print(f"欧式看涨期权 (BS): {european_call:.6f}")

    # 合理性检查
    print(f"\n=== 合理性检查 ===")
    methods = [
        ("均值", initial_call_value_mean, initial_put_value_mean),
        ("插值", initial_call_value_interp, initial_put_value_interp),
        ("最近点", initial_call_value_closest, initial_put_value_closest)
    ]

    for method_name, call_price, put_price in methods:
        call_reasonable = call_price >= european_call  # 美式 >= 欧式
        put_reasonable = put_price >= european_put  # 美式 >= 欧式

        print(f"{method_name}方法:")
        print(f"  看涨合理性: {'✅' if call_reasonable else '❌'} ({call_price:.3f} vs {european_call:.3f})")
        print(f"  看跌合理性: {'✅' if put_reasonable else '❌'} ({put_price:.3f} vs {european_put:.3f})")

    # 绘图
    plt.figure(figsize=(14, 12))

    # 子图1：模拟的期货价格路径
    plt.subplot(2, 2, 1)
    time_grid = np.linspace(0, T, len(f))
    plt.plot(time_grid, f, 'b-', linewidth=1)
    plt.axhline(y=K, color='r', linestyle='--', label=f'Strike K={K}')
    plt.axhline(y=f0, color='g', linestyle='--', label=f'S0={f0}')
    plt.xlabel('Time')
    plt.ylabel('Futures Price')
    plt.title('CEV Process Simulation')
    plt.legend()
    plt.grid(True)

    # 子图2：状态空间离散化
    plt.subplot(2, 2, 2)
    plt.hist(f, bins=30, alpha=0.7, density=True, label='Simulated Data')
    plt.axvline(x=f0, color='g', linestyle='--', label=f'S0={f0}')
    plt.axvline(x=K, color='r', linestyle='--', label=f'K={K}')

    # 显示网格点（每隔30个显示一个）
    for i, center in enumerate(centers):
        if i % 30 == 0:
            plt.axvline(x=center, color='orange', alpha=0.3, linewidth=0.5)

    plt.xlabel('Price')
    plt.ylabel('Density')
    grid_type_text = '非均匀' if use_nonuniform else '均匀'
    plt.title(f'状态空间离散化 ({grid_type_text}网格)')
    plt.legend()
    plt.grid(True)

    # 子图3：期权价值vs价格
    plt.subplot(2, 2, 3)
    plt.plot(centers, V_put[0, :], 'r-', linewidth=2, label='美式看跌')
    plt.plot(centers, V_call[0, :], 'b-', linewidth=2, label='美式看涨')
    plt.plot(centers, exercise_value_put, 'r--', alpha=0.5, label='看跌内在价值')
    plt.plot(centers, exercise_value_call, 'b--', alpha=0.5, label='看涨内在价值')
    plt.axvline(x=f0, color='g', linestyle=':', label=f'F0={f0}')
    plt.axvline(x=K, color='k', linestyle=':', label=f'K={K}')
    plt.xlabel('期货价格')
    plt.ylabel('期权价值')
    plt.title('期权价值 vs 期货价格')
    plt.legend()
    plt.grid(True)

    # 子图4：网格密度（如果是非均匀网格）
    plt.subplot(2, 2, 4)
    if use_nonuniform:
        grid_spacing = np.diff(centers)
        plt.plot(centers[:-1], grid_spacing, 'o-', markersize=2)
        plt.axvline(x=K, color='r', linestyle='--', label=f'K={K}')
        plt.xlabel('价格')
        plt.ylabel('网格间距')
        plt.title('非均匀网格密度分布')
        plt.legend()
        plt.grid(True)
    else:
        # 如果是均匀网格，显示期权Delta
        delta_put = np.gradient(V_put[0, :], centers)
        delta_call = np.gradient(V_call[0, :], centers)
        plt.plot(centers, delta_put, 'r-', linewidth=2, label='看跌Delta')
        plt.plot(centers, delta_call, 'b-', linewidth=2, label='看涨Delta')
        plt.axvline(x=f0, color='g', linestyle=':', label=f'F0={f0}')
        plt.axvline(x=K, color='k', linestyle=':', label=f'K={K}')
        plt.xlabel('期货价格')
        plt.ylabel('Delta')
        plt.title('期权Delta')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()