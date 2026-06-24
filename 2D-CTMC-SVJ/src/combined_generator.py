# -*- coding: utf-8 -*-
"""
文件名: src/combined_generator.py
功能描述: 双层组合生成元模块
         将Layer 1（方差）和Layer 2（价格）的CTMC 组合为单一mN×mN 生成元矩阵
         基于 Mackay, Vachon & Cui (2023) Eq. (7.2) Proposition 7.1
作者: [Author]
创建日期: 2026-05-06
"""

import numpy as np
from scipy import sparse
from loguru import logger


def construct_combined_generator(Q_variance, G_price_list, use_sparse=True,
                                 variance_grid=None, model_params=None,
                                 price_grid=None):
    """
    构造双层CTMC 的组合生成元矩阵 G^(m,N)

    组合生成元是一个mN×mN 的分块矩阵(Eq. 7.2):

        G^(m,N) = | q_11*I_N + G_1   q_12*S_{12}  ...  q_1m*S_{1m}  |
                  | q_21*S_{21}     q_22*I_N+G_2  ...  q_2m*S_{2m}  |
                  | ...              ...           ...  ...           |
                  | q_m1*S_{m1}     q_m2*S_{m2}   ...  q_mm*I_N+G_m |

    当 rho≠0 时, 非对角块使用平移插值矩阵 S_{lj} 替代单位阵,
    以保证方差转移时 S = exp(X + rho*V/sigma_v) 保持不变。

    参数:
        Q_variance (np.ndarray): Layer 1 方差生成元矩阵，形状 (m, m)
        G_price_list (list): Layer 2 各体制生成元矩阵列表，每个形状(N, N)
        use_sparse (bool): 是否使用稀疏矩阵存储
        variance_grid (np.ndarray or None): 方差网格 (rho≠0 时需要)
        model_params (dict or None): 模型参数 (rho≠0 时需要)
        price_grid (np.ndarray or None): 对数价格网格 (rho≠0 时需要)
    返回值:
        np.ndarray or scipy.sparse.csr_matrix: 组合生成元，形状 (m*N, m*N)
    """
    m = Q_variance.shape[0]
    if isinstance(G_price_list, np.ndarray) and G_price_list.ndim == 3:
        N = G_price_list.shape[1]
    else:
        N = G_price_list[0].shape[0]
    total_states = m * N

    rho = (model_params or {}).get('rho', 0.0)
    use_shift = abs(rho) > 1e-14 and variance_grid is not None and price_grid is not None

    logger.info(f"Building combined generator: {m}×{N} = {total_states} total states"
                f" (shift={'ON' if use_shift else 'OFF'})")

    if use_sparse:
        return _build_combined_sparse(Q_variance, G_price_list, m, N, total_states,
                                       variance_grid, model_params, use_shift, price_grid)
    else:
        return _build_combined_dense(Q_variance, G_price_list, m, N, total_states,
                                      variance_grid, model_params, use_shift, price_grid)


def _build_shift_matrix(price_grid, dx, N):
    """
    构造平移插值矩阵: 将 price_grid 上的向量平移 dx

    线性插值, 边界处使用最近邻外推

    参数:
        price_grid (np.ndarray): (N,) 价格网格
        dx (float): 平移量
        N (int): 网格点数

    返回值:
        np.ndarray: (N, N) 稀疏平移矩阵
    """
    S = np.zeros((N, N))
    targets = price_grid + dx

    for n in range(N):
        t = targets[n]
        if t <= price_grid[0]:
            S[n, 0] = 1.0
        elif t >= price_grid[-1]:
            S[n, -1] = 1.0
        else:
            idx = np.searchsorted(price_grid, t) - 1
            idx = max(0, min(idx, N - 2))
            w = (t - price_grid[idx]) / (price_grid[idx + 1] - price_grid[idx])
            S[n, idx] = 1.0 - w
            S[n, idx + 1] = w
    return S


def _build_combined_dense(Q_variance, G_price_list, m, N, total_states,
                           variance_grid=None, model_params=None, use_shift=False,
                           price_grid=None):
    """
    构造稠密组合生成元矩阵

    参数:
        Q_variance (np.ndarray): Layer 1 生成元        G_price_list (list): Layer 2 生成元列表        m (int): 方差状态数
        N (int): 价格状态数
        total_states (int): 总状态数
        variance_grid, model_params, use_shift: 平移补偿参数

    返回值:
        np.ndarray: 稠密组合生成元    """
    G_combined = np.zeros((total_states, total_states))
    I_N = np.eye(N)

    rho_sv = 0.0
    if use_shift:
        rho_sv = model_params['rho'] / model_params['sigma_v']

    for l in range(m):
        row_start = l * N
        row_end = (l + 1) * N

        for j in range(m):
            col_start = j * N
            col_end = (j + 1) * N

            if l == j:
                G_combined[row_start:row_end, col_start:col_end] = (
                    Q_variance[l, l] * I_N + G_price_list[l]
                )
            else:
                q_val = Q_variance[l, j]
                if abs(q_val) < 1e-15:
                    continue
                if use_shift:
                    dx = rho_sv * (variance_grid[l] - variance_grid[j])
                    if abs(dx) > 1e-14:
                        S_shift = _build_shift_matrix(price_grid, dx, N)
                    else:
                        S_shift = I_N
                    G_combined[row_start:row_end, col_start:col_end] = q_val * S_shift
                else:
                    G_combined[row_start:row_end, col_start:col_end] = q_val * I_N

    _log_combined_info(G_combined)
    return G_combined


def _build_combined_sparse(Q_variance, G_price_list, m, N, total_states,
                            variance_grid=None, model_params=None, use_shift=False,
                            price_grid=None):
    """
    构造稀疏组合生成元矩阵

    利用的稀疏性
        - Q^(m) 三对角（每个方差状态只与相邻状态转移）
        - 每个 G_l 三对角        - 因此组合矩阵是分块三对角

    参数:
        Q_variance, G_price_list, m, N, total_states: 同上
        variance_grid, model_params, use_shift: 平移补偿参数

    返回值:
        scipy.sparse.csr_matrix: 稀疏组合生成元
    """
    rho_sv = 0.0
    if use_shift:
        rho_sv = model_params['rho'] / model_params['sigma_v']

    shift_cache = {}
    if use_shift:
        for l in range(m):
            for lp in range(max(0, l - 2), min(m, l + 3)):
                if lp != l and abs(Q_variance[l, lp]) > 1e-15:
                    dx = rho_sv * (variance_grid[l] - variance_grid[lp])
                    if abs(dx) > 1e-14:
                        shift_cache[(l, lp)] = _build_shift_matrix(price_grid, dx, N)

    rows, cols, vals = [], [], []

    for l in range(m):
        base_row = l * N

        for i in range(N):
            row_idx = base_row + i

            for j_idx, val in _get_nonzero_entries(
                G_price_list[l], i, Q_variance, l, N,
                variance_grid, rho_sv, use_shift, shift_cache
            ):
                rows.append(row_idx)
                cols.append(j_idx)
                vals.append(val)

    G_sparse = sparse.csr_matrix(
        (vals, (rows, cols)), shape=(total_states, total_states)
    )

    _log_combined_info(G_sparse.toarray())
    logger.info(f"Combined generator sparsity: {G_sparse.nnz}/{total_states**2} "
                f"({100 * G_sparse.nnz / total_states**2:.2f}%)")

    return G_sparse


def _get_nonzero_entries(G_l, i, Q_variance, l, N,
                         variance_grid=None, rho_sv=0.0, use_shift=False,
                         shift_cache=None):
    """
    获取组合生成元第 (l*N+i) 行的非零元素

    参数:
        G_l (np.ndarray): 当前体制的Layer 2 生成元
        i (int): 当前价格状态索引
        Q_variance (np.ndarray): Layer 1 生成元
        l (int): 当前方差体制索引
        N (int): 价格状态数
        variance_grid, rho_sv, use_shift, shift_cache: 平移补偿参数

    返回值:
        list of (col_index, value) tuples
    """
    entries = []

    for j in range(max(0, i - 1), min(N, i + 2)):
        diag_val = G_l[i, j] + (Q_variance[l, l] if i == j else 0)
        if abs(diag_val) > 1e-15:
            entries.append((l * N + j, diag_val))

    for l_prime in range(max(0, l - 2), min(Q_variance.shape[0], l + 3)):
        if l_prime != l and abs(Q_variance[l, l_prime]) > 1e-15:
            if use_shift and shift_cache and (l, l_prime) in shift_cache:
                S_mat = shift_cache[(l, l_prime)]
                for j in range(max(0, i - 1), min(N, i + 2)):
                    w = S_mat[i, j]
                    if abs(w) > 1e-15:
                        entries.append((l_prime * N + j, Q_variance[l, l_prime] * w))
            else:
                entries.append((l_prime * N + i, Q_variance[l, l_prime]))

    return entries


def _log_combined_info(G):
    """
    记录组合生成元的基本信息

    参数:
        G (np.ndarray): 组合生成元矩阵    """
    diag_max = np.max(np.diag(G))
    off_diag = G.copy()
    np.fill_diagonal(off_diag, 0)
    row_sums = np.abs(G.sum(axis=1))

    logger.info(
        f"Combined generator: shape={G.shape}, "
        f"diag_max={diag_max:.6f}, "
        f"off_diag_range=[{off_diag[off_diag != 0].min():.6f}, "
        f"{off_diag.max():.6f}], "
        f"max_row_sum_dev={row_sums.max():.2e}"
    )


def flatten_state_index(price_idx, variance_idx, N):
    """
    将二维状态索引映射为一维索引(Eq. 7.1)

    参数:
        price_idx (int): 价格状态索引(0-based)
        variance_idx (int): 方差状态索引(0-based)
        N (int): 价格状态总数

    返回值:
        int: 一维状态索引    """
    return variance_idx * N + price_idx


def unflatten_state_index(flat_idx, N):
    """
    将一维索引还原为二维状态索引
    参数:
        flat_idx (int): 一维状态索引        N (int): 价格状态总数

    返回值:
        tuple: (price_idx, variance_idx)
    """
    variance_idx = flat_idx // N
    price_idx = flat_idx % N
    return price_idx, variance_idx
