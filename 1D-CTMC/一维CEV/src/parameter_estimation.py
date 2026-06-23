import copy
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.linalg import expm
from loguru import logger

# Grid resolution settings
_CALIB_NUM_BINS = 100
_CALIB_N_T = 100


def estimate_parameters_from_options(option_data, model_params, option_params,
                                     ctmc_params, initial_guess, model_type,
                                     target_date=None):
    from src.ctmc_model import construct_theoretical_q_matrix
    from src.option_pricing import price_option
    from utility.data_processor import set_state_space_range, discretize_data

    daily_data = option_data[option_data['Date'] == target_date]
    logger.info(f"Calibration starting for {target_date} ({len(daily_data)} options)")

    calib_ctmc = copy.deepcopy(ctmc_params)
    calib_ctmc['num_bins'] = _CALIB_NUM_BINS
    calib_ctmc['N_s'] = _CALIB_NUM_BINS
    calib_ctmc['N_t'] = _CALIB_N_T

    def _price_row(row, params, ctmc_p):
        local_model = copy.deepcopy(model_params)
        local_option = copy.deepcopy(option_params)
        local_ctmc = copy.deepcopy(ctmc_p)

        local_option['K'] = row['K']
        local_option['r_f'] = row['Rf']
        local_option['toT'] = row['TAU']
        local_model['F_t'] = row['F_t']

        if pd.isna(row['TAU']) or row['TAU'] <= 0: return None, None
        local_ctmc['dt'] = row['TAU'] / local_ctmc['N_t']

        x_lower, x_upper = set_state_space_range(local_ctmc['F'], row['K'], local_option['option_type'],
                                                 model_type=model_type)
        _, bins = discretize_data(local_ctmc['F'], local_ctmc['num_bins'], x_lower, x_upper)
        bins_centers = (bins[:-1] + bins[1:]) / 2

        q = construct_theoretical_q_matrix(bins_centers, local_model['F_0'], params, model_type)
        res = price_option(q, bins_centers, local_model, local_option, local_ctmc, model_type)
        return res[0], row['market_price']

    def objective(params):
        if not validate_parameters(params, model_type): return 1e10
        total_loss, n = 0.0, 0
        for _, row in daily_data.iterrows():
            try:
                m_p, mkt_p = _price_row(row, params, calib_ctmc)
                if m_p is not None and mkt_p > 0 and m_p > 0:
                    total_loss += (np.log(m_p / mkt_p)) ** 2
                    n += 1
            except:
                continue
        return total_loss / max(n, 1)

    from commonConfig import OPTIMIZER_PARAMS
    bounds = OPTIMIZER_PARAMS[model_type]['bounds']
    if model_type == 'cev_rs':
        return _multistart_optimize(objective, initial_guess, bounds, model_type, [0])
    return opt.minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds).x


def calculate_daily_pricing_performance(daily_data, optimal_params, model_params, option_params, ctmc_params,
                                        model_type):
    from src.ctmc_model import construct_theoretical_q_matrix
    from src.option_pricing import price_option
    from utility.data_processor import set_state_space_range, discretize_data

    results = []
    for _, row in daily_data.iterrows():
        try:
            if pd.isna(row['TAU']) or row['TAU'] <= 0: continue

            # --- CRITICAL FIX: Use local_model and local_option instead of global templates ---
            local_model = copy.deepcopy(model_params)
            local_option = copy.deepcopy(option_params)
            local_ctmc = copy.deepcopy(ctmc_params)

            local_option['K'] = row['K']
            local_option['r_f'] = row['Rf']
            local_option['toT'] = row['TAU']
            local_model['F_t'] = row['F_t']
            local_ctmc['dt'] = row['TAU'] / local_ctmc['N_t']

            x_lower, x_upper = set_state_space_range(local_ctmc['F'], row['K'], local_option['option_type'],
                                                     model_type=model_type)
            _, bins = discretize_data(local_ctmc['F'], local_ctmc['N_s'], x_lower, x_upper)
            bins_centers = (bins[:-1] + bins[1:]) / 2

            q = construct_theoretical_q_matrix(bins_centers, local_model['F_0'], optimal_params, model_type)
            # Pass correctly updated local_model/local_option
            pricing_res = price_option(q, bins_centers, local_model, local_option, local_ctmc, model_type)

            results.append({
                'Date': row['Date'], 'K': row['K'], 'TAU': round(row['TAU'], 4),
                'Mkt_Price': round(row['market_price'], 4), 'Model_Price': round(pricing_res[0], 4),
                'RAE_%': round(abs(pricing_res[0] - row['market_price']) / row['market_price'] * 100, 2)
            })
        except:
            continue
    return pd.DataFrame(results)


def get_parameter_names(model_type):
    names = {'cev': ['sigma', 'beta'], 'cev_dejd': ['sigma', 'beta', 'lambda', 'p', 'eta1', 'eta2'],
             'cev_rs': ['sigma1', 'beta1', 'sigma2', 'beta2', 'l12', 'l21']}
    return names.get(model_type, [])


def validate_parameters(params, model_type):
    if model_type == 'cev': return params[0] > 0
    if model_type in ['cev_dejd', 'cev_rs']: return params[0] > 0 and params[2] > 0
    return True


def _multistart_optimize(objective, initial_guess, bounds, model_type, counter):
    n_starts, rng = 3, np.random.default_rng(42)  # Reduced n_starts for faster terminal testing
    start_points = [np.array(initial_guess)]
    for _ in range(n_starts - 1):
        p = initial_guess * (1.0 + rng.uniform(-0.2, 0.2, len(initial_guess)))
        start_points.append(np.clip(p, [b[0] for b in bounds], [b[1] for b in bounds]))

    best_p, best_v = initial_guess, np.inf
    for x0 in start_points:
        try:
            res = opt.minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
            if res.fun < best_v: best_v, best_p = res.fun, res.x
        except:
            continue
    return best_p


# ===========================================================================
# Likelihood functions
# ===========================================================================

def log_likelihood_futures(q_matrix, state_sequence, time_step):
    """Negative log-likelihood for a 1-D CTMC observed at discrete times."""
    p_matrix = expm(q_matrix * time_step)
    eps = 1e-12
    ll  = 0.0
    for i in range(len(state_sequence) - 1):
        ll += np.log(p_matrix[state_sequence[i], state_sequence[i + 1]] + eps)
    return -ll


def log_likelihood_futures_rs(q_matrix, state_sequence, time_step,
                               n_price_states, initial_regime=1):
    """
    Negative log-likelihood for the CEV-RS 2-D CTMC.
    Regime state is marginalised via forward filtering.
    """
    p_matrix    = expm(q_matrix * time_step)
    eps         = 1e-12
    ll          = 0.0
    regime_prob = np.zeros(2)
    regime_prob[initial_regime - 1] = 1.0

    for i in range(len(state_sequence) - 1):
        cs, ns = state_sequence[i], state_sequence[i + 1]
        total  = 0.0
        new_rp = np.zeros(2)

        for rf in range(2):
            if regime_prob[rf] < eps:
                continue
            for rt in range(2):
                prob          = p_matrix[rf * n_price_states + cs,
                                         rt * n_price_states + ns]
                contrib       = regime_prob[rf] * prob
                total        += contrib
                new_rp[rt]   += contrib

        if total > eps:
            regime_prob  = new_rp / total
            ll          += np.log(total + eps)
        else:
            ll += np.log(eps)

    return -ll


# ===========================================================================
# Internal helpers
# ===========================================================================

def _report_rae(daily_data, params, ctmc_p, price_fn, label="RAE"):
    """
    Compute and log the Relative Absolute Error (RAE) — Eq. (20) in paper.

        RAE = (1/N) * sum( |P_model - P_market| / P_market )
    """
    total_rae, n = 0.0, 0
    for _, row in daily_data.iterrows():
        try:
            mp, mkp = price_fn(row, params, ctmc_p)
            if mkp > 0:
                total_rae += abs(mp - mkp) / mkp
                n         += 1
        except Exception:
            pass

    if n > 0:
        logger.info(f"{label} | {n} options | RAE = {total_rae / n * 100:.2f}%")
    else:
        logger.warning(f"{label}: no valid observations.")


def _get_bounds(model_type):
    if model_type == 'cev':
        return [(0.01, 0.99), (-5.0, 5.0)]
    elif model_type == 'cev_dejd':
        return [
            (0.01, 0.99),   # sigma
            (-5.0,  5.0),   # beta
            (0.01,  2.0),   # lambda
            (0.01,  0.99),  # p
            (1.1,  10.0),   # eta1  (>1 for MGF convergence)
            (0.1,  10.0),   # eta2
        ]
    elif model_type == 'cev_rs':
        return [
            (0.05,  0.99),  # sigma1  — raised from 0.01 to prevent near-zero vol
            (-10.0, 10.0),  # beta1
            (0.05,  0.99),  # sigma2  — raised from 0.01 to prevent near-zero vol
            (-10.0, 10.0),  # beta2
            (0.01,  20.0),  # lambda12
            (0.01,  20.0),  # lambda21
        ]
    raise ValueError(f"Unsupported model type: {model_type}")


# ===========================================================================
# Validation & logging
# ===========================================================================

def validate_parameters(params, model_type='cev'):
    """Lightweight sanity check before building the Q matrix."""
    if model_type == 'cev':
        if len(params) != 2:
            return False
        sigma, beta = params
        if not (0 < sigma < 1): return False
        if abs(beta) > 10:      return False

    elif model_type == 'cev_dejd':
        if len(params) != 6:
            return False
        sigma, beta, lam, p, eta1, eta2 = params
        if not (0 < sigma < 1):   return False
        if abs(beta) > 10:        return False
        if not (0 < lam <= 2):    return False
        if not (0 < p < 1):       return False
        if eta1 <= 1:             return False
        if not (0 < eta2 <= 10):  return False

    elif model_type == 'cev_rs':
        if len(params) != 6:
            return False
        s1, b1, s2, b2, l12, l21 = params
        if not (0 < s1 < 1):      return False
        if not (0 < s2 < 1):      return False
        if abs(b1) > 10:          return False
        if abs(b2) > 10:          return False
        if not (0 < l12 <= 20):   return False
        if not (0 < l21 <= 20):   return False
    else:
        return False

    return True


def log_estimation_results(params, model_type):
    if model_type == 'cev':
        logger.info(f"CEV params: sigma={params[0]:.4f}, beta={params[1]:.4f}")
    elif model_type == 'cev_dejd':
        logger.info(
            f"CEV-DEJD params: sigma={params[0]:.4f}, beta={params[1]:.4f}, "
            f"lambda={params[2]:.4f}, p={params[3]:.4f}, "
            f"eta1={params[4]:.4f}, eta2={params[5]:.4f}"
        )
    elif model_type == 'cev_rs':
        logger.info(
            f"CEV-RS params: "
            f"sigma1={params[0]:.4f}, beta1={params[1]:.4f} | "
            f"sigma2={params[2]:.4f}, beta2={params[3]:.4f} | "
            f"lambda12={params[4]:.4f}, lambda21={params[5]:.4f}"
        )


def get_parameter_names(model_type='cev'):
    if model_type == 'cev':
        return ['sigma', 'beta']
    elif model_type == 'cev_dejd':
        return ['sigma', 'beta', 'lambda', 'p', 'eta1', 'eta2']
    elif model_type == 'cev_rs':
        return ['sigma1', 'beta1', 'sigma2', 'beta2', 'lambda12', 'lambda21']
    raise ValueError(f"Unsupported model type: {model_type}")


def get_initial_guess(model_type='cev'):
    """Starting values near Table 6 of Lian & Song (2021)."""
    if model_type == 'cev':
        return np.array([0.2, -0.5])
    elif model_type == 'cev_dejd':
        return np.array([0.2, -0.5, 0.1, 0.5, 2.0, 2.0])
    elif model_type == 'cev_rs':
        return np.array([0.16, 1.0, 0.16, -1.0, 1.0, 7.0])
    raise ValueError(f"Unsupported model type: {model_type}")


# ===========================================================================
# Regime analysis utilities (CEV-RS)
# ===========================================================================

def calculate_stationary_distribution(lambda12, lambda21):
    """pi1 = lambda21/(lambda12+lambda21),  pi2 = lambda12/(lambda12+lambda21)"""
    total = lambda12 + lambda21
    if total <= 0:
        return 0.5, 0.5
    pi1, pi2 = lambda21 / total, lambda12 / total
    logger.info(f"Stationary distribution: pi1={pi1:.4f}, pi2={pi2:.4f}")
    return pi1, pi2


def calculate_expected_regime_duration(lambda12, lambda21):
    """E[T1]=1/lambda12,  E[T2]=1/lambda21  (years)"""
    if lambda12 <= 0 or lambda21 <= 0:
        return np.inf, np.inf
    e1, e2 = 1.0 / lambda12, 1.0 / lambda21
    logger.info(f"Expected durations: E[T1]={e1:.4f} yr, E[T2]={e2:.4f} yr")
    return e1, e2


def analyze_regime_parameters(params):
    """Economic interpretation of CEV-RS estimates."""
    s1, b1, s2, b2, l12, l21 = params

    pi1, pi2 = calculate_stationary_distribution(l12, l21)
    ed1, ed2 = calculate_expected_regime_duration(l12, l21)

    analysis = {
        'regime1_volatility':        s1,
        'regime2_volatility':        s2,
        'regime1_elasticity':        b1,
        'regime2_elasticity':        b2,
        'regime1_leverage_type':     'inverse' if b1 > 0 else 'normal',
        'regime2_leverage_type':     'inverse' if b2 > 0 else 'normal',
        'lambda12':                  l12,
        'lambda21':                  l21,
        'stationary_prob_regime1':   pi1,
        'stationary_prob_regime2':   pi2,
        'expected_duration_regime1': ed1,
        'expected_duration_regime2': ed2,
        'dominant_regime':           1 if pi1 > pi2 else 2,
    }
    analysis['dominant_leverage_type'] = (
        analysis['regime1_leverage_type'] if pi1 > pi2
        else analysis['regime2_leverage_type']
    )

    logger.info(
        f"Regime 1: sigma={s1:.4f}, beta={b1:.4f} "
        f"({analysis['regime1_leverage_type']} leverage)"
    )
    logger.info(
        f"Regime 2: sigma={s2:.4f}, beta={b2:.4f} "
        f"({analysis['regime2_leverage_type']} leverage)"
    )
    logger.info(
        f"Dominant: Regime {analysis['dominant_regime']} | "
        f"pi1={pi1:.4f}, pi2={pi2:.4f}"
    )
    return analysis