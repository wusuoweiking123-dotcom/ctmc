import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.linalg import expm
#from scipy.optimize import minimize
import scipy.optimize as opt

from loguru import logger


# 1. Discretize data into bins
def discretize_data(data, num_bins, x_lower, x_upper):
    x_min = x_lower - 1  #np.min(data) -1
    x_max = x_upper + 1  # np.max(data) +1
    bins = np.linspace(x_min, x_max, num_bins + 1) #状态值
    #pd.DataFrame(bins).to_csv('bins.csv')
    indices = np.digitize(data, bins) - 1
    indices = np.clip(indices, 0, num_bins - 1)
    return indices, bins


# 2. Construct theoretical Q matrix
def theoretical_Q(grid, F0, params):
    sigma, beta = params

    h = grid[1] - grid[0]
    N = len(grid)
    Q = np.zeros((N, N))

    for i, x in enumerate(grid):
        drift = 0
        diff2 = (x / F0) ** (2 * beta) * (x ** 2) * (sigma ** 2)
        if i > 0:
            Q[i, i - 1] = max(diff2 / (2 * h ** 2) - drift / (2 * h), 0)
        if i < N - 1:
            Q[i, i + 1] = max(diff2 / (2 * h ** 2) + drift / (2 * h), 0)
        Q[i, i] = -np.sum(Q[i])
    return Q


# 3. Log-likelihood function
def log_likelihood_futures(Q, state_seq, dt):
    P = expm(Q * dt)
    eps = 1e-12
    loglik = 0.0

    for i in range(len(state_seq) - 1):
        s1, s2 = state_seq[i], state_seq[i + 1]  # F(t+1)|F(t)
        prob = P[s1, s2]                         # prob(F(t+1)|F(t))
        loglik += np.log(prob + eps)
    return -loglik

# 参数拟合：MLE
def fit_params_options(option_data, model_parameters, option_parameters, CTMC_parameters, x0):
    # Step 1: all parameters
    #bins = CTMC_parameters['bins']

    # Step: construct Q
    # print("\n步骤4: 构建CTMC转移速率矩阵...")
    F_0 = model_parameters['F_0']
    num_bins = CTMC_parameters['num_bins']
    F_data = CTMC_parameters['F']

    #bins_centers = (bins[:-1] + bins[1:]) / 2
    #print(f"状态空间范围: [{bins_centers[0]:.2f}, {bins_centers[-1]:.2f}]")
    #print(f"状态数量: {len(bins_centers)}")
    #Q_mat = theoretical_Q(bins_centers, F_0, x0)
    # print(f"转移速率矩阵Q的维度: {Q_fit.shape}")

    # num_y = option_data.shape[0]
    date_list = option_data['Date'].unique()
    MLE_option = np.array([[0.0]], dtype=np.float_)

    for t in range(len(date_list)):
        date_t = date_list[t]

        if date_t == 20240603:  # the two options used in numerical demo
            Mkt_data_t = option_data.loc[option_data['Date'] == date_t, :]
            NK_T = Mkt_data_t.shape[0]  # number of options at time t

            for kt in range(NK_T):
                K_t = Mkt_data_t['K'].values[kt]
                # adjust the range for state space
                x_lower_k, x_upper_k = SetStateSpaceRange(CTMC_parameters['F'], K_t, option_parameters['option_type'])
                CTMC_parameters['x_lower'] = x_lower_k
                CTMC_parameters['x_upper'] = x_upper_k

                # 步骤1：读取Excel中的真实CEV过程数据
                # print("\n步骤1: 读取Excel中的CEV过程数据...")
                # x_lower = CTMC_parameters['x_lower']
                # x_upper = CTMC_parameters['x_upper']

                # 步骤2：离散化数据
                # print("\n步骤2: 离散化状态空间...")
                state_seq, bins = discretize_data(F_data, num_bins, x_lower_k, x_upper_k)
                CTMC_parameters['state_seq'] = state_seq
                CTMC_parameters['bins'] = bins

                bins_centers = (bins[:-1] + bins[1:]) / 2
                Q_mat = theoretical_Q(bins_centers, F_0, x0)

                # pricing
                option_parameters['K'] = K_t
                option_parameters['r_f'] = Mkt_data_t['Rf'].values[kt]
                option_parameters['F_t'] = Mkt_data_t['F_t'].values[kt]
                model_price_t, _  = option_pricing(Q_mat, bins_centers, model_parameters, option_parameters, CTMC_parameters)
                market_price_t = Mkt_data_t['market_price'].values[kt]
                re_error = (model_price_t - market_price_t)/market_price_t
                MLE_option[0][0] = MLE_option + (re_error) ** 2  # loss function

    return MLE_option[0][0]


# 4. Fit parameters using likelihood
def fit_params_futures(state_seq, bins, dt, F0, x0):

    def loss_func(params):
        Q = theoretical_Q(bins, F0, params)
        return log_likelihood_futures(Q, state_seq, dt)

    # Test
    # Q = theoretical_Q(bins, F0, x0)
    # log_re = log_likelihood_futures(Q, state_seq, dt)

    bnds = [(0.01, 0.99), (-5.0, 5.0)]
    res = opt.minimize(loss_func, x0, bounds=bnds, tol=1e-8, options={'disp': True})
    #res = opt.minimize(loss_func, x0, bounds=bnds, constraints=constraints, tol=1e-8, options={'disp': True})

    print('the optimal estimates: sigma='+str(res.x[0])+', beta='+str(res.x[1]))

    return res.x


# 5. American option pricing using dynamic programming
def option_pricing(Q, bins_centers, model_parameters, option_parameters, CTMC_parameters):
    #price_American_option(Q_fit, bins_centers, K, r_f, toT, dt, N, option_am_eu, option_type='call')

    # option info
    K = option_parameters['K']
    r_f = option_parameters['r_f']
    toT = option_parameters['toT']
    option_type = option_parameters['option_type']
    option_am_eu = option_parameters['option_am_eu']

    # CTMC setup
    N_t =  CTMC_parameters['N_t']
    N_s = CTMC_parameters['N_s']
    dt = CTMC_parameters['dt']

    # setup
    #N = len(bins_centers)
    P = expm(Q * dt)

    # valuation
    V = np.zeros((N_s, N_t+1))
    # terminal payoff
    if option_type == 'put':
        exercise_value = np.maximum(K - bins_centers, 0)
    else:
        exercise_value = np.maximum(bins_centers - K, 0)

    V[:, N_t] = exercise_value

    for i in range(N_t-1, -1, -1):  # e.g. range(10, -1, -1): 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
        continuation_value = np.exp(-r_f * dt) * P @ V[:, i + 1]  #　E[V(t+1)|F_t]
        if option_am_eu == 'eu':  # European type
             V[:,i] = continuation_value
        else:             #'am'  # American type
             # 若想定价美式期权，则替换上行代码为：
             V[:,i] = np.maximum(exercise_value, continuation_value)

    # work out the option price at t = 0
    F_t = model_parameters['F_t']
    Ft_index = np.digitize(F_t, bins_centers)
    V_t = P[Ft_index,:] @  V[:,1]

    # save data
    #pd.DataFrame(P).to_csv('P.csv')
    #pd.DataFrame(V).to_csv('V.csv')

    return V_t, V

def get_option_price_mean(option_values):
    return np.mean(option_values)

def main_empirical_test(option_data, model_parameters, option_parameters, CTMC_parameters, x0):
    # Test
    #mle_err = fit_params_options(option_data, model_parameters, option_parameters, CTMC_parameters, x0)

    #===========================================================
    def loss_func(params):
        mle_err =  fit_params_options(option_data, model_parameters, option_parameters, CTMC_parameters, params)
        return mle_err

    bnds = [(0.01, 0.99), (-10.0, 10.0)]
    res = opt.minimize(loss_func, x0, bounds=bnds, tol=1e-8, options={'disp': True})
    # res = opt.minimize(loss_func, x0, bounds=bnds, constraints=constraints, tol=1e-8, options={'disp': True})

    print('the optimal estimates: sigma=' + str(res.x[0]) + ', beta=' + str(res.x[1]))

    return res.x

def main_numerical_test(model_parameters, option_parameters, CTMC_parameters, x0):

    # Step 1: all parameters
    # 步骤1：读取Excel中的真实CEV过程数据
    print("\n步骤1: 读取Excel中的CEV过程数据...")
    F_0 = model_parameters['F_0']

    F_data = CTMC_paramters['F']
    x_lower = CTMC_paramters['x_lower']
    x_upper = CTMC_paramters['x_upper']
    num_bins = CTMC_parameters['num_bins']
    #dt = CTMC_parameters['dt']
    #N = CTMC_parameters['N']

    # 步骤2：离散化数据
    print("\n步骤2: 离散化状态空间...")
    state_seq, bins = discretize_data(F_data, num_bins, x_lower, x_upper)
    print(state_seq)
    bins_centers = (bins[:-1] + bins[1:]) / 2
    print(f"状态空间范围: [{bins_centers[0]:.2f}, {bins_centers[-1]:.2f}]")
    print(f"状态数量: {len(bins_centers)}")

    # 步骤3：参数估计
    print("\n步骤3: 估计CEV参数...")
    opt_paras = fit_params_futures(state_seq, bins, 1/252, F_0, x0)  # for futures data: dt = 1/252
    sigma_fit, beta_fit =  opt_paras
    print(f"拟合参数: σ={sigma_fit:.4f}, β={beta_fit:.4f}")

    # update model parmeters
    model_parameters['sigma'] = sigma_fit
    model_parameters['beta'] = beta_fit

    #print(f"参数误差: error_σ={np.sqrt((sigma_fit - sigma_true)**2) / abs(sigma_true):.4f}, error_β={np.sqrt((beta_fit - beta_true)**2) / abs(beta_true):.4f}")

    # 步骤4：构建CTMC转移速率矩阵
    print("\n步骤4: 构建CTMC转移速率矩阵...")
    Q_fit = theoretical_Q(bins_centers, F_0, opt_paras)
    print(f"转移速率矩阵Q的维度: {Q_fit.shape}")

    # 步骤5：美式看跌期权定价
    print("=== CEV美式期权定价算法 ===")
    print(f"期货初始价格: {F_0}, 行权价: {K}")

    # Option Pricing part
    print("\n步骤5: 美式看跌期权定价...")
    option_parameters['option_type'] = 'put'
    V_put_0, V_put = option_pricing(Q_fit, bins_centers, model_parameters, option_parameters, CTMC_parameters)
    #V_put = price_American_option(Q_fit, bins_centers, K, r_f, toT, dt, N, option_am_eu, option_type='put')
    initial_put_value_mean = V_put_0   #get_option_price_mean(V_put[:,0])
    print(f"看跌期权价格: {initial_put_value_mean:.6f}")

    # 步骤6：美式看涨期权定价
    print("\n步骤6: 美式看涨期权定价...")
    option_parameters['option_type'] = 'call'
    V_call_0, V_call = option_pricing(Q_fit, bins_centers, model_parameters, option_parameters, CTMC_parameters)
    #V_call = price_American_option(Q_fit, bins_centers, K, r_f, toT, dt, N, option_am_eu, option_type='call')

    # 方法：取均值
    initial_call_value_mean = V_call_0  #get_option_price_mean(V_call[:,0])
    print(f"看涨期权价格: {initial_call_value_mean:.6f}")

    return V_put_0, V_call_0

# utility
def read_option_data(filepath):
    option_info = pd.read_csv(filepath)
    # conduct option filtering (can be converted into a function)
    # To collect ATM (at-the-money) options:
    # 1. Tau (time-to-maturity) greater than 15 and less than 60 days
    test_data_1 = option_info[(option_info["TAU"] >= 15 / 365) & (option_info["TAU"] <= 180 / 365)]
    # 2. moneyness (K/S(t)) within [0.95, 1.05]
    test_data_2 = test_data_1[(test_data_1["MONEYNESS"] > 0.81) & (test_data_1["MONEYNESS"] <= 1.02)]
    # 3. trading volume (Volume) > 0
    test_data_3 = test_data_2[test_data_2["volume"] > 100]
    # 4. open interest (OI) high then 100
    # 5. market price higher than 1.0
    option_data = test_data_3[test_data_3["market_price"] > 0.50]
    # 6. descriptive statiatics
    logger.debug('the filtered sample size=' + str(option_data.shape))  # Test
    option_price = option_data[['Date', 'K', 'TAU', 'Rf', 'market_price', 'MONEYNESS', 'F_t']]
    return option_price

def SetStateSpaceRange(F, K, option_type):

    delta_upper = 0.05
    delta_lower = 0.02
    MAX_F = np.max(F)
    MIN_F = np.min(F)

    if option_type == 'call':
        if K > MAX_F:
            x_upper = K * (1 + delta_upper)
            x_lower = MIN_F * (1 - delta_lower)
        else:
            x_upper = MAX_F * (1 + delta_upper)
            x_lower = MIN_F * (1 - delta_lower)
    else:  # put
        if K < MIN_F:
            x_upper = MAX_F * (1 + delta_upper)
            x_lower = K * (1 - delta_lower)
        else:
            x_upper = MAX_F * (1 + delta_upper)
            x_lower = MIN_F * (1 - delta_lower)

    print('\n ==== state space=[' + str(x_lower) + ',' + str(x_upper) + '] with K=' + str(K) + '=======\n')

    return x_lower, x_upper

if __name__ == "__main__":
    #main()

    # read data
    futures_filepath = '一维CEV/input/Futures_au2412.csv'
    futures_info = pd.read_csv(futures_filepath)
    futures_price_data = futures_info[['Date', 'close']]

    # model paras
    F_0 = futures_price_data['close'][0]
    sigma = 0.2  # initial values
    beta = -0.5
    model_parameters = {
        'F_0':F_0,
        'F_t':555.22, #F_0,
        'sigma':sigma,
        'beeta':beta
    }

    # option paras
    K = 464 #520 #464 #520      # strike price
    r_f = 0.01953  # risk-free rate
    toT = 0.479452055  # time to maturity of the option
    option_type = 'put'  # call = 1 / put = 0
    option_am_eu  = 'am'   # American = 1 / European = 0
    option_parameters = {
        'r_f': r_f,
        'K':   K,
        'toT': toT,
        'option_type': option_type,
        'option_am_eu':    option_am_eu
    }

    # CTMC setup
    F = np.asarray(futures_price_data['close'][:])
    #F = df.iloc[:, 7].dropna().to_numpy()  # 第H列，索引7
    N_t = 1000 #len(F) - 1  # number of time steps
    N = len(F) - 1  # number of states
    #dt = toT/N_t  #1 / 252  # T的选择到底是多少？？？
    #print(f"读取{len(F)}个价格点，时间步长 dt: {dt:.4f}")

    #
    x_lower, x_upper = SetStateSpaceRange(F, K, option_type)

    # x_lower = 450
    # x_upper = 700

    num_bins = 500
    CTMC_paramters = {
        'F': F,
        'N': N,  # frequency of futures observations
        'N_t': N_t,
        'N_s': num_bins,
        'dt': toT/N_t,
        'x_lower': x_lower,
        'x_upper': x_upper,
        'num_bins': num_bins,
        'state_seq': [],
        'bins': []
    }

    # =================================== CASE I ==============================================
    # numerical Test
    # Strategy: 1) estimate <sigma, beta> from futures prices
    #           2) use the optimal parameters in 1) to price American/European call/put options
    #===========================================================================================
    # x0= [0.2, -0.5]
    # option_numerical_put_price, option_numerical_call_price = main_numerical_test(model_parameters, option_parameters, CTMC_paramters, x0)
    # print('option_am_eu='+option_parameters['option_am_eu'])
    # print('with the numerical put price='  + str(option_numerical_put_price))
    # print('with the numerical call price=' + str(option_numerical_call_price))

    # =================================== CASE II ==============================================
    # empirical Test
    # Strategy: 1) estimate <sigma, beta> from option prices quoted in market
    #           2) the pricing errors are minimized
    # ===========================================================================================

    futures_options_filepath = '一维CEV/input/Futures_put_options_au2412.csv'
    option_data = read_option_data(futures_options_filepath)

    # estimate model parameter from option prices
    x0 = [0.1477, 0.3615]
    opt_parameters = main_empirical_test(option_data, model_parameters, option_parameters, CTMC_paramters, x0)

    print('the demo for CTMC algorithm is done.')










