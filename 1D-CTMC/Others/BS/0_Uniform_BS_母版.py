import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


# 构造对数均匀的状态空间网格
def construct_log_grid(S_min, S_max, num_bins):
    log_grid = np.linspace(np.log(S_min), np.log(S_max), num_bins)
    grid = np.exp(log_grid)
    return grid


# 构造生成元矩阵 Q
def theoretical_Q(grid, sigma, r):
    h = grid[1] - grid[0]
    N = len(grid)
    Q = np.zeros((N, N))
    mu = r - 0.5 * sigma ** 2
    D = 0.5 * sigma ** 2

    for i in range(N):
        if i > 0:
            Q[i, i - 1] = D / h ** 2 - mu / (2 * h)
        if i < N - 1:
            Q[i, i + 1] = D / h ** 2 + mu / (2 * h)
        Q[i, i] = -np.sum(Q[i])  # 保证每行和为 0

    return Q


# 用 CTMC 定价欧式期权
def price_European_option(Q, centers, K, r, T, option_type='put'):
    P_T = expm(Q * T)

    if option_type == 'put':
        payoff = np.maximum(K - centers, 0)
    else:
        payoff = np.maximum(centers - K, 0)

    option_values = np.exp(-r * T) * P_T @ payoff
    return option_values


# 方法1：找最接近S0的网格点
def get_option_price_at_S0(grid, option_values, S0):
    s0_index = np.argmin(np.abs(grid - S0))
    return option_values[s0_index]


# 方法2：学长的方法 - 对所有状态取均值
def get_option_price_mean(option_values):
    return np.mean(option_values)


# 单次CTMC定价
def single_ctmc_price(S_min, S_max, sigma, r, T, S0, K, num_bins=1000, option_type='call', method='closest'):
    """
    method: 'closest' - 找最接近S0的网格点
            'mean' - 对所有状态取均值（学长的方法）
    """
    try:
        grid = construct_log_grid(S_min, S_max, num_bins)
        Q = theoretical_Q(grid, sigma, r)
        option_values = price_European_option(Q, grid, K, r, T, option_type)

        if method == 'mean':
            price_at_s0 = get_option_price_mean(option_values)
        else:  # method == 'closest'
            price_at_s0 = get_option_price_at_S0(grid, option_values, S0)

        return price_at_s0
    except:
        return np.inf


# 边界优化 - 支持两种方法
def find_optimal_boundaries(bs_benchmark, sigma, r, T, S0, K, num_bins=1000, option_type='call', method='closest'):
    S_min_values = np.arange(20, 81, 5)  # 20到80，步长5
    S_max_values = np.arange(150, 401, 10)  # 150到400，步长10

    best_error = np.inf
    best_bounds = None

    # 双层for循环
    for s_min in S_min_values:
        for s_max in S_max_values:
            ctmc_price = single_ctmc_price(s_min, s_max, sigma, r, T, S0, K, num_bins, option_type, method)

            if np.isfinite(ctmc_price):
                error = abs(ctmc_price - bs_benchmark)
                if error < best_error:
                    best_error = error
                    best_bounds = (s_min, s_max)

    return best_bounds, best_error


def main():
    # 参数设定
    sigma = 0.3
    r = 0.05
    T = 1
    S0 = 100
    K = 100
    num_bins = 1000
    bs_benchmark = 8.916037
    option_type = 'call'

    print("=== 两种方法对比 ===")
    print(f"BS基准价格: {bs_benchmark:.6f}")
    print()

    # 方法1：找最接近S0的网格点
    print("方法1：找最接近S0的网格点")
    optimal_bounds_1, min_error_1 = find_optimal_boundaries(bs_benchmark, sigma, r, T, S0, K, num_bins, option_type,
                                                            'closest')
    optimal_price_1 = single_ctmc_price(optimal_bounds_1[0], optimal_bounds_1[1], sigma, r, T, S0, K, num_bins,
                                        option_type, 'closest')

    print(f"最优边界: S_min = {optimal_bounds_1[0]}, S_max = {optimal_bounds_1[1]}")
    print(f"CTMC价格: {optimal_price_1:.6f}")
    print(f"绝对误差: {min_error_1:.6f}")
    print()

    # 方法2：取均值
    print("方法2：对所有状态取均值")
    optimal_bounds_2, min_error_2 = find_optimal_boundaries(bs_benchmark, sigma, r, T, S0, K, num_bins, option_type,
                                                            'mean')
    optimal_price_2 = single_ctmc_price(optimal_bounds_2[0], optimal_bounds_2[1], sigma, r, T, S0, K, num_bins,
                                        option_type, 'mean')

    print(f"最优边界: S_min = {optimal_bounds_2[0]}, S_max = {optimal_bounds_2[1]}")
    print(f"CTMC价格: {optimal_price_2:.6f}")
    print(f"绝对误差: {min_error_2:.6f}")
    print()

    # 对比
    print("=== 方法对比 ===")
    if min_error_1 < min_error_2:
        print(f"方法1更准确，误差小 {min_error_2 - min_error_1:.6f}")
    else:
        print(f"方法2更准确，误差小 {min_error_1 - min_error_2:.6f}")


if __name__ == "__main__":
    main()