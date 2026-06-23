import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates
from matplotlib.dates import date2num

# 读取数据
futures_options_filepath = r'/Users/a2022/Desktop/CTMC V2/input/Futures_put_options_au2412.csv'
df = pd.read_csv(futures_options_filepath)


# Black-Scholes看跌期权公式
def black_scholes_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def implied_volatility_put(market_price, S, K, T, r):
    def objective_function(sigma):
        return black_scholes_put(S, K, T, r, sigma) - market_price

    try:
        if market_price <= 0 or T <= 0:
            return np.nan
        return brentq(objective_function, 0.001, 5.0, xtol=1e-6)
    except:
        return np.nan


# 计算隐含波动率
df['Date'] = pd.to_datetime(df['Date'].astype(str))
df['Implied_volatility'] = [implied_volatility_put(row['market_price'], row['F_t'], row['K'], row['tau'], row['Rf'])
                            for _, row in df.iterrows()]

# 移除无效数据
df = df.dropna(subset=['Implied_volatility'])


# Gaussian kernel function
def gaussian_kernel(x, y, h1, h2):
    return (1 / (2 * np.pi)) * np.exp(-0.5 * (x ** 2 / h1)) * np.exp(-0.5 * (y ** 2 / h2))


# Nadaraya-Watson estimator
def nadaraya_watson_estimation(data, m, date, h1, h2):
    numerator = np.sum(
        data['Implied_volatility'] * gaussian_kernel(m - data['Moneyness'], date - date2num(data['Date']), h1, h2))
    denominator = np.sum(gaussian_kernel(m - data['Moneyness'], date - date2num(data['Date']), h1, h2))
    return numerator / denominator if denominator > 1e-10 else np.nan


# Construct smooth surface
def smooth_surface(data, grid_size=50, h1=0.01, h2=1):
    m_range = np.linspace(data['Moneyness'].min(), data['Moneyness'].max(), grid_size)
    date_range = np.linspace(date2num(data['Date']).min(), date2num(data['Date']).max(), grid_size)
    smooth_surface = np.zeros((grid_size, grid_size))

    for i, m in enumerate(m_range):
        for j, date in enumerate(date_range):
            smooth_surface[i, j] = nadaraya_watson_estimation(data, m, date, h1, h2)

    return m_range, date_range, smooth_surface


# Generate smooth surface
m_range, date_range, smooth_surface_data = smooth_surface(df)

# Draw 3D surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

M, DATE = np.meshgrid(m_range, date_range)
surf = ax.plot_surface(DATE, M, smooth_surface_data.T, cmap='viridis', alpha=0.8)

# 修正日期轴设置 - 7个月跨度，显示年份
ax.w_xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # 每月显示一次
ax.w_xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=8)

ax.set_xlabel('Date')
ax.set_ylabel('Moneyness')
ax.set_zlabel('Implied Volatility')
ax.set_title('Implied Volatility Surface of Options on Gold Futures')

plt.colorbar(surf, shrink=0.5, aspect=5)
plt.show()