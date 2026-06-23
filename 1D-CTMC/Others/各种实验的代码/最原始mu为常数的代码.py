import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import minimize

# 1. 模拟 OU 过程数据
def simulate_ou(theta, sigma, mu = 0.0,  x0=0, T=20, N=500):
    dt = T / N
    x = np.zeros(N)
    x[0] = x0
    np.random.seed(51)
    for i in range(1, N):
        x[i] = x[i-1] + theta * (mu - x[i-1]) * dt + sigma*np.sqrt(dt)*np.random.randn()
    return x, dt

# 2. 将数据离散为状态（分箱）
def discretize_data(data, num_bins):
    x_min, x_max = np.min(data), np.max(data)
    bins = np.linspace(x_min, x_max, num_bins + 1)
    indices = np.digitize(data, bins) - 1
    indices = np.clip(indices, 0, num_bins - 1)
    return indices, bins

# 3. 统计 N_ij 和 T_i，构造经验 Q 矩阵
def build_Q_matrix(state_seq, num_states, dt):
    N_ij = np.zeros((num_states, num_states))
    T_i = np.zeros(num_states)

    for i in range(len(state_seq) - 1):
        cur_state = state_seq[i]
        next_state = state_seq[i+1]
        N_ij[cur_state, next_state] += dt
        T_i[cur_state] += dt

    Q = np.zeros_like(N_ij)
    for i in range(num_states):
        if T_i[i] > 0:
            Q[i] = N_ij[i] / T_i[i]
            Q[i, i] = -np.sum(Q[i]) + Q[i, i]
    return Q

# 4. 构造 OU 模型的 Q 理论矩阵（用于拟合）
def theoretical_Q(grid, theta, mu, sigma):
    h = grid[1] - grid[0]
    N = len(grid)
    Q = np.zeros((N, N))
    for i, x in enumerate(grid):
        drift = theta * (mu - x)
        diff = sigma**2

        if i > 0:
            Q[i, i-1] = max(diff / (2*h**2) - drift / (2*h), 0)
        if i < N - 1:
            Q[i, i+1] = max(diff / (2*h**2) + drift / (2*h), 0)
        Q[i, i] = -np.sum(Q[i])
    return Q

# 5. 计算 MLE 损失函数
def log_likelihood(Q, state_seq, dt):
    P = expm(Q * dt)
    eps = 1e-12
    loglik = 0.0
    for i in range(len(state_seq) - 1):
        s1, s2 = state_seq[i], state_seq[i+1]
        prob = P[s1, s2]
        loglik += np.log(prob + eps)
    return -loglik  # 负号是为了最小化

# 6. 拟合 OU 参数（最大化似然）
def fit_params(state_seq, bins, dt):
    centers = (bins[:-1] + bins[1:]) / 2

    def loss(params):
        theta, sigma = params
        Q = theoretical_Q(centers, theta, 0.0, sigma)
        return log_likelihood(Q, state_seq, dt)

    res = minimize(loss, x0=[0.5, 0.2], bounds=[(0.01, 5), (0.01, 3)])
    return res.x

# 主程序
def main():
    theta_true, sigma_true = 1.0 , 0.3
    # 生成模拟数据
    data, dt = simulate_ou(theta=theta_true, sigma=sigma_true)

    # 离散状态空间
    state_seq, bins = discretize_data(data, num_bins=30)
    print(f"States: {state_seq}")
    print(f"Bins: {bins}")

    # 构造经验 Q 矩阵
    #Q_empirical = build_Q_matrix(state_seq, num_states=len(bins) - 1, dt=dt)
    #print("经验 Q 矩阵：\n", Q_empirical)

    # 计算理论 Q 矩阵
    centers = (bins[:-1] + bins[1:]) / 2
    Q_theo = theoretical_Q(centers, theta_true, 1, sigma_true)
    print("理论 Q 矩阵：\n", Q_theo)

    # 拟合 OU 参数
    est_theta, est_sigma = fit_params(state_seq, bins, dt)

    print(f"真实参数: θ = {theta_true}, σ = {sigma_true}")
    print(f"估计参数: θ = {est_theta:.4f}, σ = {est_sigma:.4f}")

    # 绘制 OU 过程轨道与离散状态空间的状态序列
    plt.figure(figsize=(10, 6))

    # 绘制 OU 过程轨道
    plt.plot(np.arange(len(data)) * dt, data, label='OU Process', color='blue', alpha=0.6)

    # 计算每个 state_seq 对应的 bins 中的值
    state_value = np.zeros(len(data))
    bins0 = bins
    for i in range(len(state_seq)):
        state_value[i] = bins[state_seq[i]]  # 从 bins 中获取对应的值

    plt.step(np.arange(len(data)) * dt, state_value, label='Markov process', color='red', alpha=0.6)

    # 添加离散状态空间的标签
    plt.title('OU Process Trajectory with Discrete State Space')
    plt.xlabel('Time')
    plt.ylabel('x(t)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
