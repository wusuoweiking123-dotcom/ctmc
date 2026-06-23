import numpy as np
from scipy.linalg import expm

from loguru import logger


def price_option(q_matrix, bins_centers, model_params, option_params, ctmc_params, model_type):
    """
    Using the dynamic programming method for option pricing

    :param q_matrix: Transition rate matrix Q
    :param bins_centers: Centers of the bins
    :param model_params: Model parameters dictionary
    :param option_params: Options parameters dictionary
    :param ctmc_params: CTMC parameters dictionary
    :param model_type: Model type ('cev', 'cev_dejd', or 'cev_rs')
    :return: (Option price, option value matrix)
             For CEV-RS: also returns early exercise boundaries for both regimes
    """
    if not validate_option_params(option_params, model_type):
        raise ValueError(f"Invalid option parameters for {model_type} model")

    if model_type == 'cev_rs':
        return price_option_cev_rs(
            q_matrix, bins_centers, model_params, option_params, ctmc_params
        )
    else:
        return price_option_standard(
            q_matrix, bins_centers, model_params, option_params, ctmc_params, model_type
        )


def price_option_standard(q_matrix, bins_centers, model_params, option_params, ctmc_params, model_type):
    """
    Standard option pricing for CEV and CEV-DEJD models (N-dimensional state space)
    """
    strike_price = option_params['K']
    risk_free_rate = option_params['r_f']
    option_type = option_params['option_type']
    option_style = option_params['option_am_eu']

    n_time_steps = ctmc_params['N_t']
    n_states = ctmc_params['N_s']
    time_step = ctmc_params['dt']

    # Calculate the transition probability matrix P = exp(Q * dt)
    p_matrix = expm(q_matrix * time_step)
    p_matrix = np.clip(p_matrix, 0, 1)
    row_sums = p_matrix.sum(axis=1)
    p_matrix = p_matrix / row_sums[:, np.newaxis]

    logger.debug("The transition probability matrix has been calculated successfully.")

    # Initialize the value matrix V[state, time]
    value_matrix = np.zeros((n_states, n_time_steps + 1))

    # Terminal payoff at expiration (boundary condition, Eq. 6 in paper)
    if option_type == 'put':
        exercise_values = np.maximum(strike_price - bins_centers, 0)
    else:
        exercise_values = np.maximum(bins_centers - strike_price, 0)

    value_matrix[:, n_time_steps] = exercise_values

    # Apply boundary conditions for jump diffusion at expiration
    if model_type == 'cev_dejd':
        value_matrix = apply_jump_boundary_conditions(
            value_matrix, bins_centers, option_params, n_time_steps
        )

    # Backward recursion: V(t_i, x) = max{ f(x), e^{-r*dt} * P * V(t_{i+1}, .) }
    # This implements Eq. (5) from the paper
    for t in range(n_time_steps - 1, -1, -1):
        continuation_values = np.exp(-risk_free_rate * time_step) * p_matrix @ value_matrix[:, t + 1]

        if option_style == 'eu':
            value_matrix[:, t] = continuation_values
        else:
            # American: compare immediate exercise with continuation
            # NOTE: use the same exercise_values for all t (payoff depends only on price, not time)
            value_matrix[:, t] = np.maximum(exercise_values, continuation_values)

    # -----------------------------------------------------------------------
    # BUG FIX (Bug 1): The backward recursion already stores the fully
    # optimal t=0 value in value_matrix[:, 0], including the early-exercise
    # comparison at t=0.  The previous code read
    #   option_price = p_matrix[idx, :] @ value_matrix[:, 1]
    # which recomputes only the *continuation* value from t=1, bypassing
    # the max{exercise, continuation} step at t=0 and therefore always
    # reporting European-style prices even for American options.
    # -----------------------------------------------------------------------
    current_price = model_params['F_t']

    # BUG FIX (Bug 3): np.digitize returns the *insertion index*, which
    # systematically over-shoots by one bin when bins_centers are grid
    # centres rather than bin edges.  Use nearest-neighbour search instead.
    current_state_index = int(np.argmin(np.abs(bins_centers - current_price)))

    option_price = value_matrix[current_state_index, 0]

    logger.info(
        f"{option_style.upper()} style {option_type} option pricing completed "
        f"using {model_type} model: {option_price:.6f}"
    )

    return option_price, value_matrix


def price_option_cev_rs(q_matrix, bins_centers, model_params, option_params, ctmc_params):
    """
    CEV-RS (Regime Switching) model option pricing with 2N-dimensional state space.

    State layout:
      states  0 … N-1  → regime 1
      states  N … 2N-1 → regime 2
    """
    strike_price = option_params['K']
    risk_free_rate = option_params['r_f']
    option_type = option_params['option_type']
    option_style = option_params['option_am_eu']

    n_time_steps = ctmc_params['N_t']
    n_price_states = len(bins_centers)
    time_step = ctmc_params['dt']
    initial_regime = model_params.get('initial_regime', 1)

    logger.info(
        f"CEV-RS pricing: {n_price_states} price states × 2 regimes = "
        f"{2 * n_price_states} total states"
    )
    logger.info(f"Initial regime: {initial_regime}")

    # 2N × 2N transition probability matrix
    p_matrix = expm(q_matrix * time_step)
    p_matrix = np.clip(p_matrix, 0, 1)
    row_sums = p_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1  # 避免除零
    p_matrix = p_matrix / row_sums[:, np.newaxis]

    logger.info(f"P matrix row sums: min={p_matrix.sum(axis=1).min():.6f}, max={p_matrix.sum(axis=1).max():.6f}")

    n_total_states = 2 * n_price_states
    value_matrix = np.zeros((n_total_states, n_time_steps + 1))

    if option_type == 'put':
        exercise_values = np.maximum(strike_price - bins_centers, 0)
    else:
        exercise_values = np.maximum(bins_centers - strike_price, 0)

    # Terminal values identical for both regimes (payoff depends only on price)
    value_matrix[:n_price_states, n_time_steps] = exercise_values
    value_matrix[n_price_states:, n_time_steps] = exercise_values

    exercise_values_2n = np.concatenate([exercise_values, exercise_values])

    exercise_boundary_regime1 = np.zeros(n_time_steps + 1)
    exercise_boundary_regime2 = np.zeros(n_time_steps + 1)

    # Backward recursion (same Eq. 5, but 2N-dimensional)
    for t in range(n_time_steps - 1, -1, -1):
        continuation_values = (
            np.exp(-risk_free_rate * time_step) * p_matrix @ value_matrix[:, t + 1]
        )

        if option_style == 'eu':
            value_matrix[:, t] = continuation_values
        else:
            value_matrix[:, t] = np.maximum(exercise_values_2n, continuation_values)

            exercise_boundary_regime1[t] = find_exercise_boundary(
                value_matrix[:n_price_states, t],
                exercise_values,
                continuation_values[:n_price_states],
                bins_centers,
                option_type,
            )
            exercise_boundary_regime2[t] = find_exercise_boundary(
                value_matrix[n_price_states:, t],
                exercise_values,
                continuation_values[n_price_states:],
                bins_centers,
                option_type,
            )

    exercise_boundary_regime1[n_time_steps] = strike_price
    exercise_boundary_regime2[n_time_steps] = strike_price

    # -----------------------------------------------------------------------
    # BUG FIX (Bug 1, same as standard): read from value_matrix[:, 0]
    # directly—the backward loop has already computed the full optimal value
    # at t=0 including the early-exercise decision.
    # BUG FIX (Bug 3): nearest-neighbour index, not np.digitize.
    # -----------------------------------------------------------------------
    current_price = model_params['F_t']
    logger.info(f"Current price: {current_price}, Grid range: [{bins_centers[0]:.2f}, {bins_centers[-1]:.2f}]")

    if current_price < bins_centers[0] or current_price > bins_centers[-1]:
        logger.warning(f"Current price outside grid range! Using nearest boundary.")

    price_index = int(np.argmin(np.abs(bins_centers - current_price)))
    logger.info(f"Price index: {price_index}, Total states: {n_total_states}")

    logger.info(f"Value matrix at t=0: min={value_matrix[:, 0].min():.6f}, max={value_matrix[:, 0].max():.6f}")

    if initial_regime == 1:
        current_state = price_index
    else:
        current_state = n_price_states + price_index

    option_price = value_matrix[current_state, 0]

    logger.info(
        f"{option_style.upper()} style {option_type} option pricing completed "
        f"using CEV-RS model: {option_price:.6f}"
    )
    logger.info(
        f"Exercise boundary at t=0: "
        f"Regime 1={exercise_boundary_regime1[0]:.2f}, "
        f"Regime 2={exercise_boundary_regime2[0]:.2f}"
    )

    result_dict = {
        'option_price': option_price,
        'value_matrix': value_matrix,
        'exercise_boundary_regime1': exercise_boundary_regime1,
        'exercise_boundary_regime2': exercise_boundary_regime2,
        'p_matrix': p_matrix,
    }

    return option_price, value_matrix, result_dict


def find_exercise_boundary(option_values, exercise_values, continuation_values,
                           bins_centers, option_type):
    """
    Find the early exercise boundary price for a single regime.

    Put:  highest price where immediate exercise is optimal (E > C).
    Call: lowest  price where immediate exercise is optimal (E > C).
    """
    n_states = len(bins_centers)
    exercise_optimal = exercise_values > continuation_values + 1e-10

    if not np.any(exercise_optimal):
        return bins_centers[0] if option_type == 'put' else bins_centers[-1]

    if option_type == 'put':
        exercise_indices = np.where(exercise_optimal)[0]
        if len(exercise_indices) > 0:
            boundary_index = exercise_indices[-1]
            if boundary_index < n_states - 1:
                e_diff = exercise_values[boundary_index] - continuation_values[boundary_index]
                e_diff_next = (exercise_values[boundary_index + 1]
                               - continuation_values[boundary_index + 1])
                if e_diff != e_diff_next:
                    weight = e_diff / (e_diff - e_diff_next)
                    boundary = (bins_centers[boundary_index]
                                + weight * (bins_centers[boundary_index + 1]
                                            - bins_centers[boundary_index]))
                else:
                    boundary = bins_centers[boundary_index]
            else:
                boundary = bins_centers[boundary_index]
            return boundary
        return bins_centers[0]
    else:
        exercise_indices = np.where(exercise_optimal)[0]
        if len(exercise_indices) > 0:
            boundary_index = exercise_indices[0]
            if boundary_index > 0:
                e_diff = exercise_values[boundary_index] - continuation_values[boundary_index]
                e_diff_prev = (exercise_values[boundary_index - 1]
                               - continuation_values[boundary_index - 1])
                if e_diff != e_diff_prev:
                    weight = -e_diff_prev / (e_diff - e_diff_prev)
                    boundary = (bins_centers[boundary_index - 1]
                                + weight * (bins_centers[boundary_index]
                                            - bins_centers[boundary_index - 1]))
                else:
                    boundary = bins_centers[boundary_index]
            else:
                boundary = bins_centers[boundary_index]
            return boundary
        return bins_centers[-1]


def calculate_implied_volatility(market_price, model_params, option_params,
                                 ctmc_params, model_type='cev'):
    """Placeholder for implied volatility calculation."""
    logger.warning(
        f"Implied volatility calculation not yet implemented for {model_type} model"
    )
    return None


def calculate_greeks(q_matrix, bins_centers, model_params, option_params,
                     ctmc_params, model_type='cev'):
    """Placeholder for Greeks calculation."""
    greeks = {'delta': None, 'gamma': None, 'theta': None, 'vega': None, 'rho': None}
    if model_type == 'cev_dejd':
        greeks.update({'lambda_sensitivity': None, 'jump_risk': None})
    if model_type == 'cev_rs':
        greeks.update({
            'delta_regime1': None,
            'delta_regime2': None,
            'regime_sensitivity': None,
        })
    logger.warning(f"Greeks calculation not yet implemented for {model_type} model")
    return greeks


def validate_option_params(option_params, model_type='cev'):
    """Validate the option parameter dictionary."""
    required_keys = ['K', 'r_f', 'toT', 'option_type', 'option_am_eu']
    for key in required_keys:
        if key not in option_params:
            logger.error(f"Missing required parameter: {key}")
            return False

    if option_params['K'] <= 0:
        logger.error("Strike price must be positive")
        return False
    if option_params['toT'] <= 0:
        logger.error("Time to maturity must be positive")
        return False
    if option_params['option_type'] not in ['call', 'put']:
        logger.error("Option type must be 'call' or 'put'")
        return False
    if option_params['option_am_eu'] not in ['am', 'eu']:
        logger.error("Option style must be 'am' or 'eu'")
        return False

    if model_type in ('cev_dejd', 'cev_rs'):
        if option_params['r_f'] < 0:
            logger.error(f"Risk-free rate should be non-negative for {model_type} model")
            return False
        if option_params['toT'] > (2 if model_type == 'cev_dejd' else 3):
            logger.warning(f"Very long maturity may cause numerical issues in {model_type} model")

    return True


def calculate_stable_transition_matrix(q_matrix, time_step, max_norm=10):
    """
    Calculate transition matrix with enhanced numerical stability.
    Uses sub-stepping when ||Q*dt|| is large.
    """
    matrix_norm = np.linalg.norm(q_matrix * time_step)

    if matrix_norm > max_norm:
        logger.warning(
            f"Large matrix norm detected: {matrix_norm:.2f}, using time step reduction"
        )
        n_substeps = int(np.ceil(matrix_norm / max_norm))
        sub_time_step = time_step / n_substeps

        p_matrix = np.eye(q_matrix.shape[0])
        p_substep = expm(q_matrix * sub_time_step)
        for _ in range(n_substeps):
            p_matrix = p_matrix @ p_substep

        p_matrix = np.clip(p_matrix, 0, 1)
        row_sums = p_matrix.sum(axis=1)
        p_matrix = p_matrix / row_sums[:, np.newaxis]
        return p_matrix
    else:
        p_matrix = expm(q_matrix * time_step)
        p_matrix = np.clip(p_matrix, 0, 1)
        row_sums = p_matrix.sum(axis=1)
        p_matrix = p_matrix / row_sums[:, np.newaxis]
        return p_matrix


def validate_transition_matrix(p_matrix, tolerance=1e-6):
    """Validate that the transition probability matrix is well-formed."""
    if np.any(p_matrix < -tolerance):
        logger.warning("Transition matrix contains negative probabilities")
        return False
    row_sums = np.sum(p_matrix, axis=1)
    if not np.allclose(row_sums, 1.0, atol=tolerance):
        logger.warning(
            f"Transition matrix row sums deviate from 1: "
            f"max deviation = {np.max(np.abs(row_sums - 1))}"
        )
        return False
    if not np.all(np.isfinite(p_matrix)):
        logger.warning("Transition matrix contains non-finite values")
        return False
    return True


def monitor_pricing_convergence(value_matrix, time_step, tolerance=1e-6):
    """Monitor convergence of the pricing algorithm."""
    if time_step < 2:
        return {'converged': True, 'max_change': 0}

    current_values = value_matrix[:, time_step]
    previous_values = value_matrix[:, time_step + 1]

    valid_mask = previous_values > tolerance
    relative_changes = np.zeros_like(current_values)
    relative_changes[valid_mask] = np.abs(
        (current_values[valid_mask] - previous_values[valid_mask])
        / previous_values[valid_mask]
    )

    max_change = np.max(relative_changes)
    return {
        'converged': max_change < tolerance,
        'max_change': max_change,
        'mean_change': np.mean(relative_changes),
        'num_large_changes': np.sum(relative_changes > tolerance),
    }


def analyze_option_value_profile(value_matrix, bins_centers, option_params, model_type):
    """Analyze the option value profile for insights."""
    strike_price = option_params['K']

    if model_type == 'cev_rs':
        n_price_states = len(bins_centers)
        initial_values_r1 = value_matrix[:n_price_states, 0]
        initial_values_r2 = value_matrix[n_price_states:, 0]
        final_values = value_matrix[:n_price_states, -1]

        # BUG FIX (Bug 3): nearest-neighbour
        strike_index = int(np.argmin(np.abs(bins_centers - strike_price)))

        analysis = {
            'model_type': model_type,
            'strike_index': strike_index,
            'at_strike_initial_value_r1': initial_values_r1[strike_index],
            'at_strike_initial_value_r2': initial_values_r2[strike_index],
            'at_strike_intrinsic_value': final_values[strike_index],
            'max_time_value_r1': np.max(initial_values_r1 - final_values),
            'max_time_value_r2': np.max(initial_values_r2 - final_values),
            'regime_value_difference': np.max(np.abs(initial_values_r1 - initial_values_r2)),
        }
        logger.info(f"Option value analysis (CEV-RS):")
        logger.info(f"  Max time value Regime 1: {analysis['max_time_value_r1']:.4f}")
        logger.info(f"  Max time value Regime 2: {analysis['max_time_value_r2']:.4f}")
        logger.info(f"  Max regime difference: {analysis['regime_value_difference']:.4f}")
    else:
        # BUG FIX (Bug 3)
        strike_index = int(np.argmin(np.abs(bins_centers - strike_price)))

        initial_values = value_matrix[:, 0]
        final_values = value_matrix[:, -1]

        analysis = {
            'model_type': model_type,
            'strike_index': strike_index,
            'at_strike_initial_value': initial_values[strike_index],
            'at_strike_intrinsic_value': final_values[strike_index],
            'max_time_value': np.max(initial_values - final_values),
            'total_value_range': np.max(initial_values) - np.min(initial_values),
            'early_exercise_region': None,
        }

        if option_params['option_am_eu'] == 'am':
            early_exercise_mask = np.abs(initial_values - final_values) < 1e-6
            analysis['early_exercise_region'] = np.sum(early_exercise_mask)

        logger.info(
            f"Option value analysis ({model_type}): "
            f"max time value = {analysis['max_time_value']:.4f}"
        )

    return analysis


def apply_jump_boundary_conditions(value_matrix, bins_centers, option_params, n_time_steps):
    """
    Apply boundary conditions for jump diffusion models at each time step.
    Called *once* after the terminal payoff is set, before backward recursion.
    """
    strike_price = option_params['K']
    option_type = option_params['option_type']

    for t in range(n_time_steps + 1):
        if option_type == 'put':
            value_matrix[0, t] = max(strike_price - bins_centers[0], 0)
            value_matrix[-1, t] = 0.0
        else:
            value_matrix[0, t] = 0.0
            time_remaining = (n_time_steps - t) * option_params['toT'] / n_time_steps
            discounted_strike = strike_price * np.exp(-option_params['r_f'] * time_remaining)
            value_matrix[-1, t] = max(bins_centers[-1] - discounted_strike, 0)

    return value_matrix


def interpolate_boundary_price(current_price, bins_centers, option_values, transition_probs):
    """Linear interpolation when current price is near state-space boundaries."""
    if current_price <= bins_centers[0]:
        return option_values[0]
    if current_price >= bins_centers[-1]:
        return option_values[-1]

    idx = np.searchsorted(bins_centers, current_price)
    idx = np.clip(idx, 1, len(bins_centers) - 1)

    weight = ((current_price - bins_centers[idx - 1])
              / (bins_centers[idx] - bins_centers[idx - 1]))
    return (1 - weight) * option_values[idx - 1] + weight * option_values[idx]


def compare_model_prices(price_cev, price_cev_dejd, price_cev_rs, option_params):
    """Compare option prices across CEV, CEV-DEJD and CEV-RS models."""
    comparison = {
        'cev_price': price_cev,
        'cev_dejd_price': price_cev_dejd,
        'cev_rs_price': price_cev_rs,
        'dejd_vs_cev_diff': price_cev_dejd - price_cev,
        'rs_vs_cev_diff': price_cev_rs - price_cev,
        'rs_vs_dejd_diff': price_cev_rs - price_cev_dejd,
    }
    if price_cev > 0:
        comparison['dejd_vs_cev_pct'] = (price_cev_dejd - price_cev) / price_cev * 100
        comparison['rs_vs_cev_pct'] = (price_cev_rs - price_cev) / price_cev * 100

    logger.info("Model comparison:")
    logger.info(f"  CEV:      {price_cev:.6f}")
    logger.info(f"  CEV-DEJD: {price_cev_dejd:.6f} (diff: {comparison['dejd_vs_cev_diff']:+.6f})")
    logger.info(f"  CEV-RS:   {price_cev_rs:.6f} (diff: {comparison['rs_vs_cev_diff']:+.6f})")

    return comparison


def extract_regime_values(value_matrix, bins_centers):
    """Extract per-regime value matrices from the combined CEV-RS matrix."""
    n_price_states = len(bins_centers)
    return value_matrix[:n_price_states, :], value_matrix[n_price_states:, :]