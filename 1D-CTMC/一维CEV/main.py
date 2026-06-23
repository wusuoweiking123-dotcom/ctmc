import sys
import copy
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

from commonConfig import *
from utility.file_utils import read_futures_data, read_option_data, create_directories, save_results
from utility.data_processor import filter_option_data
from src.parameter_estimation import estimate_parameters_from_options, calculate_daily_pricing_performance, \
    get_parameter_names, get_initial_guess


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    create_directories([FILE_PATHS['log_dir'], FILE_PATHS['result_dir']])

    model_type = MODEL_CONFIG['model_type']

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"{FILE_PATHS['log_dir']}run_{model_type}_{timestamp_str}.log"
    logger.add(log_file_path, level="INFO", format=LOG_CONFIG['format'])

    logger.info(f"--- CTMC Terminal Report System | Model: {model_type.upper()} ---")

    try:
        f_data = read_futures_data(FILE_PATHS['futures_data'])
        o_data_raw = read_option_data(FILE_PATHS['options_data'])
        o_data = filter_option_data(o_data_raw, OPTION_FILTER_CRITERIA)

        m_params = copy.deepcopy(DEFAULT_MODEL_PARAMS)
        m_params['F_0'] = f_data['close'][0]
        c_params = copy.deepcopy(DEFAULT_CTMC_PARAMS)
        c_params['F'] = np.asarray(f_data['close'][:])
        c_params['N'] = len(c_params['F']) - 1

        target_dates = [20240612, 20240620]  # Modify dates as needed
        all_results = []

        for date in target_dates:
            logger.info(f"Calculating for Date: {date}...")
            opt_p = estimate_parameters_from_options(o_data, m_params, DEFAULT_OPTION_PARAMS, c_params,
                                                     get_initial_guess(model_type), model_type, date)

            # Generate Report for current date
            day_data = o_data[o_data['Date'] == date]
            perf_df = calculate_daily_pricing_performance(day_data, opt_p, m_params, DEFAULT_OPTION_PARAMS, c_params,
                                                          model_type)

            if not perf_df.empty:
                logger.info(f"\n[ Pricing Report for {date} ]\n{perf_df.to_string(index=False)}")
                all_results.append(perf_df)

            names = get_parameter_names(model_type)
            optimal_params_dict = dict(zip(names, np.round(opt_p, 4)))
            logger.info(f"Optimal Parameters: {optimal_params_dict}\n")

        if all_results:
            final = pd.concat(all_results)

            summary_text = (
                f"\n{'=' * 50}\n"
                f"FINAL BATCH SUMMARY ({model_type.upper()})\n"
                f"Total Options: {len(final)}\n"
                f"Global Mean RAE: {final['RAE_%'].mean():.2f}%\n"
                f"{'=' * 50}"
            )
            logger.info(summary_text)

            save_results(
                data=final,
                filename="batch_pricing_report",
                result_dir=FILE_PATHS['result_dir'],
                model_type=model_type,
                timestamp=True
            )

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()