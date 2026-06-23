import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 读取数据
file_path = r'/Users/a2022/Desktop/CTMC V2/input/Futures_au2412.csv'
df = pd.read_csv(file_path)

# 转换日期格式
df['Date'] = pd.to_datetime(df['Date'].astype(str))

# 创建图表
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['close'], linewidth=2, color='steelblue')

# 添加网格线
plt.grid(True, alpha=0.5, linestyle='-', linewidth=0.8)

# 设置标题和标签
plt.title('Gold Futures Prices', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Prices',fontsize=12)
plt.show()
