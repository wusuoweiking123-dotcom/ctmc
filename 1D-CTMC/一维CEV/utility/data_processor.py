import numpy as np
import pandas as pd

from loguru import logger


def discretize_data(data, num_bins, x_lower, x_upper):
    """
    Discretize the continuous data into a specified number of bins.

    :param data: Input data array
    :param num_bins: Number of bins
    :param x_lower: Date lower bound
    :param x_upper: Data upper bound
    :return: (indices, bins) - State index sequence and binning boundaries

    Example:
        >>> data = np.array([100, 105, 103])
        >>> indices, bins = discretize_data(data, 10, 95, 110)
    """

    # Expand the boundaries to ensure that all data is within the range
    x_min = x_lower - 1
    x_max = x_upper + 1

    # Create equally spaced bins
    bins = np.linspace(x_min, x_max, num_bins + 1)

    # Distribute the data into the boxes
    indices = np.digitize(data, bins) - 1

    # Ensure that the index is within the valid range
    indices = np.clip(indices, 0, num_bins - 1)

    logger.debug(f"Data discretization completed: {len(data)} data points -> {num_bins} states")

    return indices, bins


def set_state_space_range(futures_data, strike_price, option_type, model_type='cev',
                          delta_upper=None, delta_lower=None):
    """
    Set the state space range based on the type of option and the setting status of the exercise price
    Different models may require different state space ranges

    :param futures_data: options price data
    :param strike_price: strike price
    :param option_type: options type ('call' or 'put')
    :param model_type: 'cev', 'cev_dejd', or 'cev_rs'
    :param delta_upper: Upper bound expansion ratio (if None, use model defaults)
    :param delta_lower: Lower bound expansion ratio (if None, use model defaults)
    :return: (x_lower, x_upper) - Upper and lower bounds of the state space
    """
    max_price = np.max(futures_data)
    min_price = np.min(futures_data)

    # Set default deltas based on model type if not provided
    if delta_upper is None or delta_lower is None:
        if model_type == 'cev':
            delta_upper = delta_upper or 0.05
            delta_lower = delta_lower or 0.02
        elif model_type == 'cev_dejd':
            # Larger range for jump diffusion to accommodate jumps
            delta_upper = delta_upper or 0.15
            delta_lower = delta_lower or 0.10
            logger.info(f"CEV-DEJD model: Using expanded state space for jump accommodation")
        elif model_type == 'cev_rs':
            # Moderate expansion for regime switching
            delta_upper = delta_upper or 0.10
            delta_lower = delta_lower or 0.08
            logger.info(f"CEV-RS model: Using expanded state space for regime dynamics")
        else:
            delta_upper = delta_upper or 0.05
            delta_lower = delta_lower or 0.02

    if option_type == 'call':
        if strike_price > max_price:
            x_upper = strike_price * (1 + delta_upper)
            x_lower = min_price * (1 - delta_lower)
        else:
            x_upper = max_price * (1 + delta_upper)
            x_lower = min_price * (1 - delta_lower)
    else:  # put
        if strike_price < min_price:
            x_upper = max_price * (1 + delta_upper)
            x_lower = strike_price * (1 - delta_lower)
        else:
            x_upper = max_price * (1 + delta_upper)
            x_lower = min_price * (1 - delta_lower)

    # Additional adjustment for jump diffusion models: ensure minimum range
    if model_type == 'cev_dejd':
        price_range = x_upper - x_lower
        min_range = max_price * 0.5  # Minimum 50% of max price as range
        if price_range < min_range:
            center = (x_upper + x_lower) / 2
            x_upper = center + min_range / 2
            x_lower = center - min_range / 2
            # Ensure x_lower is positive
            if x_lower <= 0:
                x_lower = max_price * 0.1
                x_upper = x_lower + min_range

    # Additional adjustment for regime switching models
    if model_type == 'cev_rs':
        # Ensure sufficient range for potential regime-driven volatility changes
        price_range = x_upper - x_lower
        min_range = max_price * 0.4  # Minimum 40% of max price as range
        if price_range < min_range:
            center = (x_upper + x_lower) / 2
            x_upper = center + min_range / 2
            x_lower = center - min_range / 2
            if x_lower <= 0:
                x_lower = max_price * 0.1
                x_upper = x_lower + min_range

    # Ensure x_lower is always positive
    x_lower = max(x_lower, max_price * 0.05)  # At least 5% of max price

    logger.info(f"State space range for {model_type}: [{x_lower:.2f}, {x_upper:.2f}]")

    return x_lower, x_upper


def filter_option_data(option_data, filter_criteria):
    """
    Filter the option data based on the specified conditions

    :param option_data: DataFrame(original option data)
    :param filter_criteria: Filter condition dictionary
    :return: The filtered option data
    """
    # Copy the data to avoid modifying the original data
    filtered_data = option_data.copy()

    # 1. Expiry Date Filtering
    filtered_data = filtered_data[
        (filtered_data["TAU"] >= filter_criteria['tau_min']) &
        (filtered_data["TAU"] <= filter_criteria['tau_max'])
        ]

    # 2. Moneyness Filtering
    filtered_data = filtered_data[
        (filtered_data["MONEYNESS"] > filter_criteria['moneyness_min']) &
        (filtered_data["MONEYNESS"] <= filter_criteria['moneyness_max'])
        ]

    # 3. Transaction Volumes Filtering
    filtered_data = filtered_data[
        filtered_data["volume"] > filter_criteria['volume_min']
        ]

    # 4. Market Prices Filtering
    filtered_data = filtered_data[
        filtered_data["market_price"] > filter_criteria['price_min']
        ]

    logger.info(f"Filter the options data: {len(option_data)} -> {len(filtered_data)} records")

    # Extract the required columns
    result_columns = ['Date', 'K', 'TAU', 'Rf', 'market_price', 'MONEYNESS', 'F_t']
    return filtered_data[result_columns]


def calculate_relative_error(model_price, market_price):
    """
    Calculate the relative error between the model price and the market price

    :param model_price: The price calculated by the model
    :param market_price: Actual market price
    :return: Relative error
    """
    if market_price == 0:
        logger.warning("The market price is 0, so the relative error cannot be calculated.")
        return np.inf

    return (model_price - market_price) / market_price


def analyze_jump_characteristics(futures_data, threshold_pct=0.02):
    """
    Analyze potential jump characteristics in futures data
    This helps in setting appropriate parameters for CEV-DEJD model

    :param futures_data: Array of futures prices
    :param threshold_pct: Threshold percentage to identify potential jumps
    :return: Dictionary with jump statistics
    """
    returns = np.diff(np.log(futures_data))
    threshold = np.std(returns) * 2  # 2 standard deviations

    # Identify potential jumps
    large_moves = np.abs(returns) > threshold
    positive_jumps = returns > threshold
    negative_jumps = returns < -threshold

    jump_stats = {
        'total_observations': len(returns),
        'potential_jumps': np.sum(large_moves),
        'jump_frequency': np.sum(large_moves) / len(returns),
        'positive_jumps': np.sum(positive_jumps),
        'negative_jumps': np.sum(negative_jumps),
        'jump_asymmetry': np.sum(positive_jumps) / max(np.sum(large_moves), 1),
        'avg_positive_jump': np.mean(returns[positive_jumps]) if np.sum(positive_jumps) > 0 else 0,
        'avg_negative_jump': np.mean(returns[negative_jumps]) if np.sum(negative_jumps) > 0 else 0,
        'max_positive_jump': np.max(returns) if len(returns) > 0 else 0,
        'max_negative_jump': np.min(returns) if len(returns) > 0 else 0
    }

    logger.info(
        f"Jump analysis: {jump_stats['potential_jumps']} potential jumps out of {jump_stats['total_observations']} observations")
    logger.info(f"Jump frequency: {jump_stats['jump_frequency']:.4f}")

    return jump_stats


def analyze_regime_characteristics(futures_data, window_size=20):
    """
    Analyze potential regime-switching characteristics in futures data
    This helps in setting appropriate parameters for CEV-RS model

    :param futures_data: Array of futures prices
    :param window_size: Rolling window size for volatility calculation
    :return: Dictionary with regime statistics
    """
    returns = np.diff(np.log(futures_data))

    if len(returns) < window_size * 2:
        logger.warning("Insufficient data for regime analysis")
        return None

    # Calculate rolling volatility
    rolling_vol = pd.Series(returns).rolling(window=window_size).std() * np.sqrt(252)
    rolling_vol = rolling_vol.dropna().values

    # Identify high and low volatility regimes
    vol_median = np.median(rolling_vol)
    high_vol_regime = rolling_vol > vol_median
    low_vol_regime = rolling_vol <= vol_median

    # Calculate regime statistics
    regime_stats = {
        'total_periods': len(rolling_vol),
        'high_vol_periods': np.sum(high_vol_regime),
        'low_vol_periods': np.sum(low_vol_regime),
        'high_vol_proportion': np.sum(high_vol_regime) / len(rolling_vol),
        'avg_high_vol': np.mean(rolling_vol[high_vol_regime]),
        'avg_low_vol': np.mean(rolling_vol[low_vol_regime]),
        'vol_ratio': np.mean(rolling_vol[high_vol_regime]) / np.mean(rolling_vol[low_vol_regime]) if np.mean(rolling_vol[low_vol_regime]) > 0 else np.inf
    }

    # Estimate transition frequencies
    transitions_to_high = 0
    transitions_to_low = 0
    for i in range(1, len(high_vol_regime)):
        if high_vol_regime[i] and not high_vol_regime[i - 1]:
            transitions_to_high += 1
        elif not high_vol_regime[i] and high_vol_regime[i - 1]:
            transitions_to_low += 1

    # Convert to annualized transition rates (approximate)
    periods_per_year = 252 / window_size
    regime_stats['estimated_lambda12'] = transitions_to_high / max(regime_stats['low_vol_periods'], 1) * periods_per_year
    regime_stats['estimated_lambda21'] = transitions_to_low / max(regime_stats['high_vol_periods'], 1) * periods_per_year

    logger.info(f"Regime analysis:")
    logger.info(f"  High volatility periods: {regime_stats['high_vol_proportion']:.2%}")
    logger.info(f"  Average high vol: {regime_stats['avg_high_vol']:.4f}")
    logger.info(f"  Average low vol: {regime_stats['avg_low_vol']:.4f}")
    logger.info(f"  Estimated λ12: {regime_stats['estimated_lambda12']:.4f}")
    logger.info(f"  Estimated λ21: {regime_stats['estimated_lambda21']:.4f}")

    return regime_stats


def analyze_leverage_effect(futures_data, window_size=20):
    """
    Analyze the leverage effect in futures data
    Positive correlation between returns and volatility changes = inverse leverage
    Negative correlation = normal leverage

    :param futures_data: Array of futures prices
    :param window_size: Rolling window size
    :return: Dictionary with leverage analysis
    """
    returns = np.diff(np.log(futures_data))

    if len(returns) < window_size * 2:
        logger.warning("Insufficient data for leverage analysis")
        return None

    # Calculate rolling volatility
    rolling_vol = pd.Series(returns).rolling(window=window_size).std()
    vol_changes = rolling_vol.diff().dropna().values

    # Align returns with volatility changes
    aligned_returns = returns[window_size:]
    if len(aligned_returns) > len(vol_changes):
        aligned_returns = aligned_returns[:len(vol_changes)]
    elif len(vol_changes) > len(aligned_returns):
        vol_changes = vol_changes[:len(aligned_returns)]

    # Calculate correlation
    valid_mask = ~(np.isnan(aligned_returns) | np.isnan(vol_changes))
    if np.sum(valid_mask) < 10:
        logger.warning("Insufficient valid data for correlation calculation")
        return None

    correlation = np.corrcoef(aligned_returns[valid_mask], vol_changes[valid_mask])[0, 1]

    leverage_analysis = {
        'correlation': correlation,
        'leverage_type': 'inverse' if correlation > 0 else 'normal',
        'strength': abs(correlation),
        'interpretation': 'Strong' if abs(correlation) > 0.3 else 'Moderate' if abs(correlation) > 0.1 else 'Weak'
    }

    logger.info(f"Leverage analysis:")
    logger.info(f"  Return-volatility correlation: {correlation:.4f}")
    logger.info(f"  Leverage type: {leverage_analysis['leverage_type']}")
    logger.info(f"  Strength: {leverage_analysis['interpretation']}")

    return leverage_analysis


def suggest_state_space_params(futures_data, model_type='cev'):
    """
    Suggest optimal state space parameters based on data characteristics

    :param futures_data: Array of futures prices
    :param model_type: 'cev', 'cev_dejd', or 'cev_rs'
    :return: Dictionary with suggested parameters
    """
    if model_type == 'cev_dejd':
        jump_stats = analyze_jump_characteristics(futures_data)

        # Adjust parameters based on jump characteristics
        if jump_stats['jump_frequency'] > 0.05:  # High jump frequency
            delta_upper = 0.20
            delta_lower = 0.15
            num_bins = 600  # More bins for jump diffusion
        elif jump_stats['jump_frequency'] > 0.02:  # Medium jump frequency
            delta_upper = 0.15
            delta_lower = 0.10
            num_bins = 550
        else:  # Low jump frequency
            delta_upper = 0.10
            delta_lower = 0.08
            num_bins = 500

        suggestions = {
            'delta_upper': delta_upper,
            'delta_lower': delta_lower,
            'num_bins': num_bins,
            'jump_stats': jump_stats
        }

    elif model_type == 'cev_rs':
        regime_stats = analyze_regime_characteristics(futures_data)
        leverage_analysis = analyze_leverage_effect(futures_data)

        # Adjust parameters based on regime characteristics
        if regime_stats is not None:
            vol_ratio = regime_stats.get('vol_ratio', 1.5)
            if vol_ratio > 2.0:  # High volatility contrast between regimes
                delta_upper = 0.15
                delta_lower = 0.12
                num_bins = 550
            elif vol_ratio > 1.5:  # Moderate contrast
                delta_upper = 0.10
                delta_lower = 0.08
                num_bins = 500
            else:  # Low contrast
                delta_upper = 0.08
                delta_lower = 0.06
                num_bins = 450
        else:
            delta_upper = 0.10
            delta_lower = 0.08
            num_bins = 500

        suggestions = {
            'delta_upper': delta_upper,
            'delta_lower': delta_lower,
            'num_bins': num_bins,
            'regime_stats': regime_stats,
            'leverage_analysis': leverage_analysis
        }

    else:
        # Original CEV suggestions
        suggestions = {
            'delta_upper': 0.05,
            'delta_lower': 0.02,
            'num_bins': 500,
            'jump_stats': None
        }

    logger.info(f"State space suggestions for {model_type}: delta_upper={suggestions['delta_upper']}, "
                f"delta_lower={suggestions['delta_lower']}, num_bins={suggestions['num_bins']}")

    return suggestions


def prepare_data_for_calibration(futures_data, option_data, model_type='cev'):
    """
    Prepare data for model calibration

    :param futures_data: Futures price data
    :param option_data: Option market data
    :param model_type: Model type
    :return: Dictionary with prepared data
    """
    # Analyze data characteristics
    suggestions = suggest_state_space_params(futures_data, model_type)

    # Prepare the return dictionary
    prepared_data = {
        'futures_data': np.asarray(futures_data),
        'option_data': option_data,
        'suggested_params': suggestions,
        'model_type': model_type
    }

    # Add model-specific analysis
    if model_type == 'cev_rs':
        prepared_data['regime_analysis'] = analyze_regime_characteristics(futures_data)
        prepared_data['leverage_analysis'] = analyze_leverage_effect(futures_data)
    elif model_type == 'cev_dejd':
        prepared_data['jump_analysis'] = analyze_jump_characteristics(futures_data)

    return prepared_data