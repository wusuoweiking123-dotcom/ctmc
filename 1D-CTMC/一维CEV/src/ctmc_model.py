import numpy as np
from scipy.linalg import expm
from scipy.stats import expon

from loguru import logger


def construct_theoretical_q_matrix(grid, initial_price, params, model_type='cev'):
    """
    Construct the generator (transition-rate) matrix Q for the given model.

    :param grid: Price-state grid (centres of discretisation bins)
    :param initial_price: F_0 used in the local-vol scaling (x/F_0)^β
    :param params: Parameter vector
                   CEV:      [sigma, beta]
                   CEV-DEJD: [sigma, beta, lambda, p, eta1, eta2]
                   CEV-RS:   [sigma1, beta1, sigma2, beta2, lambda12, lambda21]
    :param model_type: 'cev' | 'cev_dejd' | 'cev_rs'
    :return: Q matrix  (N×N for CEV/CEV-DEJD, 2N×2N for CEV-RS)
    """
    if model_type == 'cev':
        return construct_cev_q_matrix(grid, initial_price, params)
    elif model_type == 'cev_dejd':
        return construct_cev_dejd_q_matrix(grid, initial_price, params)
    elif model_type == 'cev_rs':
        return construct_cev_rs_q_matrix(grid, initial_price, params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def construct_cev_q_matrix(grid, initial_price, params):
    sigma, beta = params
    h = grid[1] - grid[0]
    n_states = len(grid)
    q_matrix = np.zeros((n_states, n_states))

    for i, x in enumerate(grid):
        # 修复扩散项计算
        volatility_term = sigma * (x / initial_price) ** beta
        diffusion_sq = (volatility_term * x) ** 2
        drift = 0.0

        if i > 0:
            q_matrix[i, i - 1] = max(diffusion_sq / (2 * h ** 2) - drift / (2 * h), 0)
        if i < n_states - 1:
            q_matrix[i, i + 1] = max(diffusion_sq / (2 * h ** 2) + drift / (2 * h), 0)

        q_matrix[i, i] = -np.sum(q_matrix[i])

    logger.debug(f"CEV Q matrix: min={q_matrix.min():.6f}, max={q_matrix.max():.6f}")
    logger.debug(f"Q matrix row sums: {np.abs(q_matrix.sum(axis=1)).max():.2e}")

    return q_matrix


# ---------------------------------------------------------------------------
# CEV-DEJD Q matrix  (Eq. 13–14 in Lian & Song 2021)
# ---------------------------------------------------------------------------

def construct_cev_dejd_q_matrix(grid, initial_price, params):
    """
    Generator matrix for CEV with Kou (2002) double-exponential jump diffusion.

    Q = Λ_D + Λ_J

    Λ_D is the diffusion part (with jump-induced drift correction),
    Λ_J is the jump part derived from the Lévy measure of the DEJD model.
    """
    sigma, beta, lambda_jump, p, eta1, eta2 = params

    zeta = calculate_zeta(p, eta1, eta2)

    lambda_d = construct_diffusion_matrix(
        grid, initial_price, sigma, beta, lambda_jump, zeta
    )
    lambda_j = construct_jump_matrix(grid, initial_price, lambda_jump, p, eta1, eta2, beta)

    q_matrix = lambda_d + lambda_j

    logger.debug(f"CEV-DEJD Q matrix built: {q_matrix.shape}")
    logger.debug(f"Λ_D range: [{lambda_d.min():.6f}, {lambda_d.max():.6f}]")
    logger.debug(f"Λ_J range: [{lambda_j.min():.6f}, {lambda_j.max():.6f}]")

    return q_matrix


# ---------------------------------------------------------------------------
# CEV-RS Q matrix  (Eq. 18 in Lian & Song 2021)
# ---------------------------------------------------------------------------

def construct_cev_rs_q_matrix(grid, initial_price, params):
    """
    2N × 2N block generator matrix for the regime-switching CEV model.

    Q = | Λ₁ - λ₁₂ I    λ₁₂ I  |
        |   λ₂₁ I    Λ₂ - λ₂₁ I|

    States 0…N-1 correspond to regime 1, states N…2N-1 to regime 2.
    """
    sigma1, beta1, sigma2, beta2, lambda12, lambda21 = params
    n_states = len(grid)

    lambda1 = construct_cev_q_matrix(grid, initial_price, [sigma1, beta1])
    lambda2 = construct_cev_q_matrix(grid, initial_price, [sigma2, beta2])

    I_n = np.eye(n_states)

    q_matrix = np.block([
        [-lambda12 * I_n + lambda1,  lambda12 * I_n],
        [lambda21 * I_n,            -lambda21 * I_n + lambda2],
    ])

    logger.debug(f"CEV-RS Q matrix built: {q_matrix.shape}")
    return q_matrix


# ---------------------------------------------------------------------------
# Helpers for CEV-DEJD
# ---------------------------------------------------------------------------

def calculate_zeta(p, eta1, eta2):
    """
    Compensator ζ = E[e^Y] − 1 for the DEJD jump distribution.

    From Eq. (11):  ζ = p·η₁/(η₁−1) + (1−p)·η₂/(η₂+1) − 1

    Requires η₁ > 1 for the upward-jump MGF to exist.
    """
    if eta1 <= 1:
        logger.warning(f"η1={eta1} ≤ 1 — MGF does not exist; clamping to 1.01")
        eta1 = 1.01
    zeta = p * eta1 / (eta1 - 1) + (1 - p) * eta2 / (eta2 + 1) - 1
    logger.debug(f"Compensator ζ = {zeta:.6f}  (p={p}, η1={eta1}, η2={eta2})")
    return zeta


def construct_diffusion_matrix(grid, initial_price, sigma, beta, lambda_jump, zeta):
    """
    Diffusion sub-matrix Λ_D for the CEV-DEJD model (Eq. 14).

    The drift of the *diffusion* part includes the jump-risk correction
    −λ ζ x  that ensures the risk-neutral condition dF/F = 0 in expectation.
    """
    h = grid[1] - grid[0]
    n_states = len(grid)
    lambda_d = np.zeros((n_states, n_states))

    for i, x in enumerate(grid):
        drift = -lambda_jump * zeta * x
        diffusion_sq = (x / initial_price) ** (2 * beta) * x ** 2 * sigma ** 2

        if i > 0:
            lambda_d[i, i - 1] = max(
                diffusion_sq / (2 * h ** 2) - drift / (2 * h), 0
            )
        if i < n_states - 1:
            lambda_d[i, i + 1] = max(
                diffusion_sq / (2 * h ** 2) + drift / (2 * h), 0
            )

        lambda_d[i, i] = -np.sum(lambda_d[i, :])

    return lambda_d


def construct_jump_matrix(grid, initial_price, lambda_jump, p, eta1, eta2, beta):
    """
    Jump sub-matrix Λ_J for the CEV-DEJD model (Eq. 13 + Appendix A.2).

    Derived by integrating the Lévy measure
        ν(x, dy) = (x/S₀)^β λ [ p η₁ (y+1)^{−η₁−1} 1_{y>0}
                                + (1−p) η₂ (y+1)^{η₂−1} 1_{−1<y<0} ] dy
    over each bin interval [αᵢ(j−1), αᵢ(j)].

    -----------------------------------------------------------------------
    BUG FIX (Bug 4-a):  The **downward-jump** branch previously used
        (αᵢ(j)+1)^{−η₂}
    but the correct integral of η₂(y+1)^{η₂−1} dy is (y+1)^{η₂}, so the
    exponent must be **+η₂** (positive), not −η₂.

    BUG FIX (Bug 4-b):  The **mixed** case αᵢ(j−1) < 0 ≤ αᵢ(j) was
    entirely missing.  This occurs whenever a jump crosses the y=0 line
    (i.e. jump size exactly straddles the current price level).  The
    integral must be split at y=0 into downward and upward parts.
    -----------------------------------------------------------------------
    """
    n_states = len(grid)
    lambda_j = np.zeros((n_states, n_states))

    for i in range(n_states):
        # Boundary states: no jump contribution (Eq. 13, condition 1)
        if i == 0 or i == n_states - 1:
            continue

        common_factor = lambda_jump * (grid[i] / initial_price) ** beta

        for j in range(n_states):
            if i == j:
                continue   # diagonal filled at the end

            # αᵢ(j−1) and αᵢ(j) as defined below Eq. (13)
            alpha_jm1 = _alpha(i, j - 1, grid)   # lower edge of bin j
            alpha_j   = _alpha(i, j,     grid)   # upper edge of bin j

            # Skip the j=0 lower edge (undefined lower bin)
            if j == 0:
                continue

            # ---- Case 2: purely upward  (both α > 0) ----------------------
            if alpha_jm1 > 0 and alpha_j > 0:
                # Integral of p η₁ (y+1)^{−η₁−1} dy from α(j-1) to α(j)
                # = −p [(y+1)^{−η₁}]_{α(j−1)}^{α(j)}
                # = p [ (α(j−1)+1)^{−η₁} − (α(j)+1)^{−η₁} ]  > 0
                lambda_j[i, j] = common_factor * p * (
                    (alpha_jm1 + 1) ** (-eta1) - (alpha_j + 1) ** (-eta1)
                )

            # ---- Case 3: purely downward  (both α < 0) --------------------
            # BUG FIX: exponent is +η₂, not −η₂
            # Integral of (1−p) η₂ (y+1)^{η₂−1} dy from α(j−1) to α(j)
            # = (1−p) [(y+1)^{η₂}]_{α(j−1)}^{α(j)}
            # = (1−p) [ (α(j)+1)^{η₂} − (α(j−1)+1)^{η₂} ]  > 0
            # because α(j) > α(j−1) implies (α(j)+1) > (α(j−1)+1) in (0,1)
            elif alpha_jm1 < 0 and alpha_j < 0:
                lambda_j[i, j] = common_factor * (1 - p) * (
                    (alpha_j + 1) ** eta2 - (alpha_jm1 + 1) ** eta2
                )

            # ---- Case 4 (NEW): mixed — bin straddles y = 0 ----------------
            # α(j−1) < 0 ≤ α(j)
            # Split integral at y=0:
            #   downward part: (1−p) ∫_{α(j−1)}^{0} η₂(y+1)^{η₂−1}dy
            #                = (1−p)[1 − (α(j−1)+1)^{η₂}]
            #   upward part:    p ∫_{0}^{α(j)} η₁(y+1)^{−η₁−1}dy
            #                =  p[1 − (α(j)+1)^{−η₁}]
            elif alpha_jm1 < 0 <= alpha_j:
                down_part = (1 - p) * (1.0 - (alpha_jm1 + 1) ** eta2)
                up_part   = p * (1.0 - (alpha_j + 1) ** (-eta1))
                lambda_j[i, j] = common_factor * (down_part + up_part)

            # all-zero for the remaining edge case (alpha_jm1 >= 0 > alpha_j)
            # which cannot occur for a monotone α function and a well-ordered grid

    # Diagonal: row-sum = 0  (generator property)
    for i in range(1, n_states - 1):
        lambda_j[i, i] = -np.sum(lambda_j[i, :])

    logger.debug("Λ_J built successfully")
    return lambda_j


# Keep the old name as an alias so existing call sites don't break
construct_jump_matrix_exact = construct_jump_matrix


def _alpha(i, j, grid):
    """
    Bin-centre ratio  αᵢ(j) = ½[(xⱼ/xᵢ)−1] + ½[(xⱼ₊₁/xᵢ)−1]

    Returns ±∞ for out-of-range j to simplify boundary logic.
    """
    if j < 0:
        return float('-inf')
    if j >= len(grid) - 1:
        return float('+inf')

    x_i = grid[i]
    if x_i <= 0:
        logger.warning(f"Non-positive grid point x_i={x_i} at index {i}")
        return 0.0

    return 0.5 * (grid[j] / x_i - 1.0) + 0.5 * (grid[j + 1] / x_i - 1.0)


# Keep the public name expected by call sites
calculate_alpha = _alpha


def calculate_zeta_ref(p, eta1, eta2):
    """Alias kept for backward compatibility."""
    return calculate_zeta(p, eta1, eta2)


def calculate_double_exponential_density(y, p, eta1, eta2):
    """
    DEJD density f_Y(y) (Eq. 12).

    f_Y(y) = p η₁ e^{−η₁ y} 1_{y≥0} + (1−p) η₂ e^{η₂ y} 1_{y<0}
    """
    if y >= 0:
        return p * eta1 * np.exp(-eta1 * y)
    else:
        return (1 - p) * eta2 * np.exp(eta2 * y)


def calculate_transition_probability(q_matrix, time_step):
    """P = exp(Q Δt), clipped and row-normalised."""
    p_matrix = expm(q_matrix * time_step)
    p_matrix = np.clip(p_matrix, 0, 1)
    row_sums = p_matrix.sum(axis=1)
    p_matrix = p_matrix / row_sums[:, np.newaxis]
    logger.debug("Transition probability matrix calculated.")
    return p_matrix


def validate_q_matrix(q_matrix, model_type='cev'):
    """
    Check that Q satisfies the generator-matrix conditions:
      1. diagonal elements ≤ 0
      2. off-diagonal elements ≥ 0
      3. each row sums to 0
    """
    diagonal_valid     = np.all(np.diag(q_matrix) <= 0)
    off_diag           = q_matrix - np.diag(np.diag(q_matrix))
    off_diagonal_valid = np.all(off_diag >= -1e-10)   # small tolerance for numerics
    row_sums           = np.abs(q_matrix.sum(axis=1))
    row_sum_valid      = np.all(row_sums < 1e-8)

    is_valid = diagonal_valid and off_diagonal_valid and row_sum_valid

    if not is_valid:
        logger.warning(f"Q-matrix validation failed ({model_type})")
        if not diagonal_valid:
            logger.warning("  Some diagonal elements are positive.")
        if not off_diagonal_valid:
            logger.warning("  Some off-diagonal elements are negative.")
        if not row_sum_valid:
            logger.warning(
                f"  Max row-sum deviation: {row_sums.max():.2e}"
            )

    if model_type == 'cev_rs' and q_matrix.shape[0] % 2 != 0:
        logger.warning(
            f"CEV-RS Q matrix should have even dimension, got {q_matrix.shape[0]}"
        )
        is_valid = False

    return is_valid


def get_state_index(price, bins_centers):
    """Nearest-neighbour grid lookup (replaces np.digitize)."""
    return int(np.argmin(np.abs(bins_centers - price)))


def get_regime_state_index(price, bins_centers, regime):
    """
    Global state index in the 2N CEV-RS state space.

    Regime 1 → [0, N)
    Regime 2 → [N, 2N)
    """
    n_states = len(bins_centers)
    price_index = get_state_index(price, bins_centers)
    if regime == 1:
        return price_index
    elif regime == 2:
        return n_states + price_index
    else:
        raise ValueError(f"Invalid regime {regime}: must be 1 or 2.")


def extract_regime_q_matrices(q_matrix_full):
    """
    Recover (Λ₁, Λ₂, λ₁₂, λ₂₁) from the full 2N×2N CEV-RS generator.
    """
    n_total  = q_matrix_full.shape[0]
    n_states = n_total // 2

    top_left    = q_matrix_full[:n_states,  :n_states]
    top_right   = q_matrix_full[:n_states,   n_states:]
    bottom_left = q_matrix_full[n_states:,  :n_states]
    bottom_right= q_matrix_full[n_states:,   n_states:]

    lambda12 = np.mean(np.diag(top_right))
    lambda21 = np.mean(np.diag(bottom_left))

    I_n    = np.eye(n_states)
    lambda1 = top_left  + lambda12 * I_n
    lambda2 = bottom_right + lambda21 * I_n

    return lambda1, lambda2, lambda12, lambda21