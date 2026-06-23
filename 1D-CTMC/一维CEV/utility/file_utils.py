import os
import pandas as pd
import numpy as np
from datetime import datetime

from loguru import logger


def read_futures_data(filepath):
    """
    read futures data file

    :param filepath: file path
    :return: DataFrame including the date and the closing price

    Example:
        >>> data = read_futures_data('input/Futures_au2412.csv')
    """
    try:
        futures_info = pd.read_csv(filepath)
        logger.info(f"Successfully retrieved futures data: {filepath}")

        # Extract the required columns
        futures_price_data = futures_info[['Date', 'close']]
        return futures_price_data

    except FileNotFoundError:
        logger.error(f"File doesn't exist: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Failed to read futures data: {str(e)}")
        raise


def read_option_data(filepath):
    """
    read options data file

    :param filepath: file path
    :return: DataFrame(original options data)
    """
    try:
        option_info = pd.read_csv(filepath)
        logger.info(f"Successfully retrieved options data: {filepath}")
        return option_info

    except FileNotFoundError:
        logger.error(f"File doesn't exist: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Failed to read options data: {str(e)}")
        raise


def save_results(data, filename, result_dir='result', model_type='cev', timestamp=True):
    """
    Store the result data in the specified directory with model type identification

    :param data: The data to be saved (DataFrame or numpy array)
    :param filename: base filename (without extension)
    :param result_dir: result directory
    :param model_type: model type ('cev', 'cev_dejd', or 'cev_rs') for filename distinction
    :param timestamp: whether to add timestamp to filename
    """
    # Ensure that the directory exists
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        logger.info(f"Created result directory: {result_dir}")

    # Modify filename to include model type
    base_name, ext = os.path.splitext(filename)
    if not ext:
        ext = '.csv'

    # Add model type and optional timestamp
    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_filename = f"{base_name}_{model_type}_{timestamp_str}{ext}"
    else:
        final_filename = f"{base_name}_{model_type}{ext}"

    filepath = os.path.join(result_dir, final_filename)

    try:
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        elif isinstance(data, np.ndarray):
            pd.DataFrame(data).to_csv(filepath, index=False)
        elif isinstance(data, dict):
            pd.DataFrame(data).to_csv(filepath, index=False)
        else:
            # Convert to DataFrame
            pd.DataFrame(data).to_csv(filepath, index=False)

        logger.info(f"Results saved to: {filepath}")

    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        raise


def save_parameter_results(params, param_names, model_type, result_dir='result',
                           additional_info=None):
    """
    Save parameter estimation results with metadata

    :param params: Estimated parameter values
    :param param_names: Parameter names
    :param model_type: Model type
    :param result_dir: Result directory
    :param additional_info: Additional information to save
    """
    # Create parameter DataFrame
    param_df = pd.DataFrame({
        'parameter': param_names,
        'value': params,
        'model_type': [model_type] * len(param_names)
    })

    # Add additional information if provided
    if additional_info:
        for key, value in additional_info.items():
            param_df[key] = [value] * len(param_names)

    # Add timestamp
    param_df['estimation_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Add model-specific interpretation for CEV-RS
    if model_type == 'cev_rs' and len(params) == 6:
        interpretations = [
            'Regime 1 volatility',
            'Regime 1 elasticity (>0: inverse leverage)',
            'Regime 2 volatility',
            'Regime 2 elasticity (<0: normal leverage)',
            'Transition rate 1→2',
            'Transition rate 2→1'
        ]
        param_df['interpretation'] = interpretations

    filename = f"estimated_parameters_{model_type}.csv"
    save_results(param_df, filename, result_dir, model_type, timestamp=True)

    return param_df


def save_model_comparison(comparison_results, result_dir='result'):
    """
    Save model comparison results

    :param comparison_results: Dictionary with comparison data
    :param result_dir: Result directory
    """
    # Convert comparison results to DataFrame
    comparison_df = pd.DataFrame([comparison_results])
    comparison_df['comparison_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    filename = "model_comparison_results.csv"
    filepath = os.path.join(result_dir, filename)

    # Append to existing file or create new one
    if os.path.exists(filepath):
        existing_df = pd.read_csv(filepath)
        combined_df = pd.concat([existing_df, comparison_df], ignore_index=True)
        combined_df.to_csv(filepath, index=False)
    else:
        comparison_df.to_csv(filepath, index=False)

    logger.info(f"Model comparison results saved to: {filepath}")


def save_option_pricing_results(put_price, call_price, model_params, option_params,
                                model_type, result_dir='result'):
    """
    Save comprehensive option pricing results

    :param put_price: Put option price
    :param call_price: Call option price
    :param model_params: Model parameters used
    :param option_params: Option parameters used
    :param model_type: Model type
    :param result_dir: Result directory
    """
    # Create results dictionary
    results = {
        'model_type': model_type,
        'put_price': put_price,
        'call_price': call_price,
        'strike_price': option_params['K'],
        'time_to_maturity': option_params['toT'],
        'risk_free_rate': option_params['r_f'],
        'option_style': option_params['option_am_eu'],
        'current_futures_price': model_params['F_t'],
        'initial_futures_price': model_params['F_0'],
        'pricing_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Add model-specific parameters
    if model_type == 'cev':
        results.update({
            'sigma': model_params.get('sigma'),
            'beta': model_params.get('beta')
        })
    elif model_type == 'cev_dejd':
        results.update({
            'sigma': model_params.get('sigma'),
            'beta': model_params.get('beta'),
            'lambda': model_params.get('lambda'),
            'p': model_params.get('p'),
            'eta1': model_params.get('eta1'),
            'eta2': model_params.get('eta2')
        })
    elif model_type == 'cev_rs':
        results.update({
            'sigma1': model_params.get('sigma1'),
            'beta1': model_params.get('beta1'),
            'sigma2': model_params.get('sigma2'),
            'beta2': model_params.get('beta2'),
            'lambda12': model_params.get('lambda12'),
            'lambda21': model_params.get('lambda21'),
            'initial_regime': model_params.get('initial_regime', 1)
        })

    # Save to file
    filename = f"option_pricing_results_{model_type}.csv"
    filepath = os.path.join(result_dir, filename)

    results_df = pd.DataFrame([results])

    # Append to existing file or create new one
    if os.path.exists(filepath):
        existing_df = pd.read_csv(filepath)
        combined_df = pd.concat([existing_df, results_df], ignore_index=True)
        combined_df.to_csv(filepath, index=False)
    else:
        results_df.to_csv(filepath, index=False)

    logger.info(f"Option pricing results saved to: {filepath}")


def save_exercise_boundary_results(boundaries, bins_centers, model_type, option_params,
                                   result_dir='result'):
    """
    Save early exercise boundary results (especially useful for CEV-RS)

    :param boundaries: Dictionary with boundary data
                       For CEV-RS: {'regime1': array, 'regime2': array}
    :param bins_centers: Price grid centers
    :param model_type: Model type
    :param option_params: Option parameters
    :param result_dir: Result directory
    """
    # Ensure directory exists
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    if model_type == 'cev_rs':
        # Save both regime boundaries
        n_time_steps = len(boundaries.get('regime1', []))
        time_grid = np.linspace(0, option_params['toT'], n_time_steps)

        boundary_df = pd.DataFrame({
            'time': time_grid,
            'boundary_regime1': boundaries.get('regime1', np.zeros(n_time_steps)),
            'boundary_regime2': boundaries.get('regime2', np.zeros(n_time_steps))
        })

        filename = f"exercise_boundary_{model_type}_{timestamp_str}.csv"
        filepath = os.path.join(result_dir, filename)
        boundary_df.to_csv(filepath, index=False)

        logger.info(f"CEV-RS exercise boundaries saved to: {filepath}")

    else:
        # Single boundary for CEV and CEV-DEJD
        if isinstance(boundaries, np.ndarray):
            n_time_steps = len(boundaries)
            time_grid = np.linspace(0, option_params['toT'], n_time_steps)

            boundary_df = pd.DataFrame({
                'time': time_grid,
                'boundary': boundaries
            })

            filename = f"exercise_boundary_{model_type}_{timestamp_str}.csv"
            filepath = os.path.join(result_dir, filename)
            boundary_df.to_csv(filepath, index=False)

            logger.info(f"Exercise boundary saved to: {filepath}")


def save_regime_analysis_results(regime_params, regime_stats, result_dir='result'):
    """
    Save regime analysis results for CEV-RS model

    :param regime_params: Dictionary with regime parameters
    :param regime_stats: Dictionary with regime statistics
    :param result_dir: Result directory
    """
    # Ensure directory exists
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Combine parameters and statistics
    combined_results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **regime_params,
        **regime_stats
    }

    # Convert to DataFrame
    results_df = pd.DataFrame([combined_results])

    filename = f"regime_analysis_cev_rs_{timestamp_str}.csv"
    filepath = os.path.join(result_dir, filename)
    results_df.to_csv(filepath, index=False)

    logger.info(f"Regime analysis results saved to: {filepath}")


def create_directories(directories):
    """
    Create the directory structure required for the project

    :param directories: Directory path list
    """
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")


def check_file_exists(filepath):
    """
    Check if the file exists

    :param filepath: file path
    :return: bool - whether the file exists
    """
    exists = os.path.exists(filepath)
    if not exists:
        logger.warning(f"File doesn't exist: {filepath}")
    return exists


def backup_results(result_dir='result', backup_dir='backup'):
    """
    Create backup of existing results before new runs

    :param result_dir: Source result directory
    :param backup_dir: Backup directory
    """
    if not os.path.exists(result_dir):
        return

    # Create backup directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    specific_backup_dir = os.path.join(backup_dir, f"backup_{timestamp}")

    if not os.path.exists(specific_backup_dir):
        os.makedirs(specific_backup_dir)

    # Copy all files from result_dir to backup
    import shutil
    for filename in os.listdir(result_dir):
        src_path = os.path.join(result_dir, filename)
        dst_path = os.path.join(specific_backup_dir, filename)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)

    logger.info(f"Results backed up to: {specific_backup_dir}")


def load_previous_results(model_type, result_dir='result'):
    """
    Load previous results for the specified model type

    :param model_type: Model type to load results for
    :param result_dir: Result directory
    :return: Dictionary with loaded results or None if not found
    """
    results = {}

    # Try to load parameter results
    param_file = os.path.join(result_dir, f"estimated_parameters_{model_type}.csv")
    if os.path.exists(param_file):
        results['parameters'] = pd.read_csv(param_file)
        logger.info(f"Loaded previous parameter results for {model_type}")

    # Try to load pricing results
    pricing_file = os.path.join(result_dir, f"option_pricing_results_{model_type}.csv")
    if os.path.exists(pricing_file):
        results['pricing'] = pd.read_csv(pricing_file)
        logger.info(f"Loaded previous pricing results for {model_type}")

    # For CEV-RS, also try to load exercise boundary
    if model_type == 'cev_rs':
        boundary_files = [f for f in os.listdir(result_dir) if f.startswith('exercise_boundary_cev_rs')]
        if boundary_files:
            # Load the most recent one
            boundary_files.sort(reverse=True)
            boundary_file = os.path.join(result_dir, boundary_files[0])
            results['exercise_boundaries'] = pd.read_csv(boundary_file)
            logger.info(f"Loaded previous exercise boundaries for CEV-RS")

    return results if results else None


def clean_old_results(result_dir='result', days_old=30):
    """
    Clean up old result files

    :param result_dir: Result directory
    :param days_old: Number of days after which files are considered old
    """
    if not os.path.exists(result_dir):
        return

    import time
    current_time = time.time()
    cutoff_time = current_time - (days_old * 24 * 60 * 60)

    removed_count = 0
    for filename in os.listdir(result_dir):
        filepath = os.path.join(result_dir, filename)
        if os.path.isfile(filepath):
            file_time = os.path.getmtime(filepath)
            if file_time < cutoff_time:
                os.remove(filepath)
                removed_count += 1
                logger.info(f"Removed old result file: {filename}")

    if removed_count > 0:
        logger.info(f"Cleaned up {removed_count} old result files")
    else:
        logger.info("No old result files to clean up")


def export_results_to_excel(model_results, output_path='result/model_summary.xlsx'):
    """
    Export all model results to a single Excel file with multiple sheets

    :param model_results: Dictionary with results from multiple models
    :param output_path: Output Excel file path
    """
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for model_type, results in model_results.items():
                # Write parameters
                if 'parameters' in results:
                    results['parameters'].to_excel(writer, sheet_name=f'{model_type}_params', index=False)

                # Write pricing results
                if 'pricing' in results:
                    results['pricing'].to_excel(writer, sheet_name=f'{model_type}_pricing', index=False)

                # Write exercise boundaries for CEV-RS
                if model_type == 'cev_rs' and 'exercise_boundaries' in results:
                    results['exercise_boundaries'].to_excel(writer, sheet_name='cev_rs_boundaries', index=False)

        logger.info(f"Results exported to Excel: {output_path}")

    except Exception as e:
        logger.error(f"Failed to export to Excel: {str(e)}")
        raise