# Default parameters of the model
DEFAULT_MODEL_PARAMS = {
    'F_0': None,      # Initial futures price (read from the data)
    'F_t': None,      # Current futures price

    # ===== CEV model parameters =====
    'sigma': 0.2,     # Initial volatility value
    'beta': -0.5,     # The initial values of the beta parameters of the CEV model

    # ===== CEV-DEJD additional parameters =====
    'lambda': 0.1,    # Poisson process intensity λ
    'p': 0.5,         # Probability of upward jump (0 ≤ p ≤ 1)
    'eta1': 2.0,      # Upward jump parameter η1 > 0
    'eta2': 2.0,      # Downward jump parameter η2 > 0

    # ===== CEV-RS (Regime Switching) parameters =====
    'sigma1': 0.16,   # Regime 1 volatility
    'beta1': 1.0,     # Regime 1 elasticity parameter (positive = inverse leverage)
    'sigma2': 0.16,   # Regime 2 volatility
    'beta2': -1.0,    # Regime 2 elasticity parameter (negative = normal leverage)
    'lambda12': 1.0,  # Transition rate from regime 1 to regime 2
    'lambda21': 7.0,  # Transition rate from regime 2 to regime 1
    'initial_regime': 1  # Initial regime state (1 or 2)
}

# Model selection configuration
MODEL_CONFIG = {
    'model_type': 'cev_rs',  # Options: 'cev', 'cev_dejd', 'cev_rs'
}

# Default parameters of options
DEFAULT_OPTION_PARAMS = {
    'r_f': None,              # Risk-free rate
    'K': None,                    # Strike price
    'toT': None,          # Maturity date (in years)
    'option_type': 'put',        # Options type: 'call' or 'put'
    'option_am_eu': 'am'         # Options style: 'am'(American) or 'eu'(European)
}

# Default parameters of CTMC
DEFAULT_CTMC_PARAMS = {
    'N': None,                   # The number of futures observations (calculated from the data)
    'N_t': 1000,                 # Time steps
    'N_s': 500,                  # The number of bins in the state space
    'dt': None,                  # Time step (calculated from toT/N_t)
    'x_lower': None,             # Lower bound of the state space
    'x_upper': None,             # Upper bound of the state space
    'num_bins': 500,             # The number of discretized bins
    'state_seq': [],             # State sequences
    'bins': [],                  # Bin boundary
    'F': None                    # Futures price data array
}

# Optimizer parameters - separated by model type
OPTIMIZER_PARAMS = {
    'cev': {
        'method': 'minimize',
        'bounds': [(0.01, 0.99), (-10.0, 10.0)],  # (sigma, beta) bounds
        'tol': 1e-8,
        'options': {'disp': True}
    },
    'cev_dejd': {
        'method': 'minimize',
        'bounds': [
            (0.01, 0.99),    # sigma bounds
            (-10.0, 10.0),   # beta bounds
            (0.01, 2.0),     # lambda bounds
            (0.01, 0.99),    # p bounds
            (0.1, 10.0),     # eta1 bounds
            (0.1, 10.0)      # eta2 bounds
        ],
        'tol': 1e-8,
        'options': {'disp': True}
    },
    'cev_rs': {
        'method': 'minimize',
        'bounds': [
            (0.01, 0.99),    # sigma1 bounds
            (-10.0, 10.0),   # beta1 bounds
            (0.01, 0.99),    # sigma2 bounds
            (-10.0, 10.0),   # beta2 bounds
            (0.01, 20.0),    # lambda12 bounds
            (0.01, 20.0)     # lambda21 bounds
        ],
        'tol': 1e-8,
        'options': {'disp': True}
    }
}

# File path configuration
FILE_PATHS = {
    'futures_data': 'input/Futures_au2412.csv',
    'options_data': 'input/Futures_put_options_au2412.csv',
    'result_dir': 'result/',
    'log_dir': 'log/',
    'data_dir': 'data/'
}

# Data filtering criteria
OPTION_FILTER_CRITERIA = {
    'tau_min': 15 / 365,         # Minimum maturity time (days/year)
    'tau_max': 180 / 365,        # Maximum maturity time (days/year)
    'moneyness_min': 0.81,       # Minimum moneyness
    'moneyness_max': 1.02,       # Maximum moneyness
    'volume_min': 100,           # Minimum transaction volumes
    'price_min': 0.50            # Minimum market price
}

# Adjustment parameters of the state space
STATE_SPACE_PARAMS = {
    'cev': {
        'delta_upper': 0.05,     # Upper bound adjustment ratio
        'delta_lower': 0.02      # Lower bound adjustment ratio
    },
    'cev_dejd': {
        'delta_upper': 0.15,     # Larger range for jump diffusion
        'delta_lower': 0.10
    },
    'cev_rs': {
        'delta_upper': 0.10,     # Moderate expansion for regime switching
        'delta_lower': 0.08
    }
}

# Log configuration
LOG_CONFIG = {
    'level': 'DEBUG',
    'format': '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level}</level> | <cyan>{message}</cyan>',
    'rotation': '10 MB'
}

# Model descriptions for documentation
MODEL_DESCRIPTIONS = {
    'cev': {
        'name': 'Constant Elasticity of Variance (CEV)',
        'parameters': ['sigma (volatility)', 'beta (elasticity)'],
        'description': 'Basic CEV model with local volatility depending on price level.'
    },
    'cev_dejd': {
        'name': 'CEV with Double Exponential Jump Diffusion (CEV-DEJD)',
        'parameters': ['sigma', 'beta', 'lambda (jump intensity)', 'p (up probability)',
                       'eta1 (up decay)', 'eta2 (down decay)'],
        'description': 'CEV model extended with Kou-style double exponential jumps.'
    },
    'cev_rs': {
        'name': 'CEV with Regime Switching (CEV-RS)',
        'parameters': ['sigma1', 'beta1', 'sigma2', 'beta2', 'lambda12', 'lambda21'],
        'description': 'CEV model with Markov regime switching between two states. '
                       'Captures structural changes like bulls/bears markets.'
    }
}