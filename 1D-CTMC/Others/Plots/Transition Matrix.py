import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import random

# 读取数据
P = pd.read_csv('/Users/a2022/Desktop/CTMC V2/doc/P.csv', index_col=0).values


plt.figure(figsize=(8, 8))
plt.spy(P, markersize=1)
plt.title("Spy plot of Transition Matrix (non-zero entries)")
plt.xlabel("To state")
plt.ylabel("From state")
plt.show()



# 网络图
# 设定一个阈值，只显示转移概率大于该阈值的边
threshold = 0.1

# 创建有向图
G = nx.DiGraph()

# 添加边（只保留大概率转移）
for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        if P[i, j] > threshold:
            G.add_edge(i, j, weight=P[i, j])

# 绘制网络
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G, seed=42)  # 力导向布局
edges = nx.draw_networkx_edges(G, pos, alpha=0.3)
nodes = nx.draw_networkx_nodes(G, pos, node_size=30, node_color="blue")
plt.title(f"Transition Network (edges > {threshold})")
plt.axis("off")
plt.show()

# 创建有向图（换成 G_sub）
G_sub = nx.DiGraph()

# 添加边（只保留大概率转移）
for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        if P[i, j] > threshold:
            G_sub.add_edge(i, j, weight=P[i, j])

# 绘制网络
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G_sub, seed=42)  # 力导向布局

# 绘制节点和边
nx.draw_networkx_nodes(G_sub, pos, node_size=30, node_color="blue")
nx.draw_networkx_edges(G_sub, pos, alpha=0.3, arrowsize=5)

# ----------- (1) 随机挑选一些节点做标注 ------------
sample_nodes = random.sample(list(G_sub.nodes()), 5)  # 随机挑 5 个节点
nx.draw_networkx_labels(G_sub, pos, labels={n: f"State {n}" for n in sample_nodes},
                        font_size=8, font_color="red")

# ----------- (2) 在部分边上显示转移概率权重 ------------
sample_edges = random.sample(list(G_sub.edges(data=True)), 5)  # 随机挑 5 条边
edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in sample_edges}
nx.draw_networkx_edge_labels(G_sub, pos, edge_labels=edge_labels,
                             font_size=7, font_color="green")

# ----------- (3) 在图外加文字说明框 ------------
textstr = (
    "Network Representation of Transition Matrix\n"
    "• Blue nodes = discrete states\n"
    "• Edges = transitions with probability > 0.1\n"
    "• Red labels = example states\n"
    "• Green numbers = transition probabilities"
)
plt.gcf().text(0.02, 0.02, textstr, fontsize=10, va="bottom", ha="left",
               bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"))

plt.title(f"Transition Network (edges > {threshold})", fontsize=14, fontweight="bold")
plt.axis("off")
plt.show()











# 设置图表样式
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
 
# 方法1：完整热力图（推荐用于海报）
plt.figure(figsize=(10, 8))
plt.imshow(P, cmap='Reds', aspect='auto')
plt.colorbar(label='Transition Probability')
plt.title('State Transition Probability Matrix P (500×500)', fontsize=14, fontweight='bold')
plt.xlabel('Target State')
plt.ylabel('Current State')

# 只显示部分刻度避免拥挤
ticks = np.arange(0, 500, 100)
plt.xticks(ticks, [f'{i}' for i in ticks])
plt.yticks(ticks, [f'{i}' for i in ticks])

plt.tight_layout()
plt.savefig('transition_matrix_P.png', dpi=300, bbox_inches='tight')
plt.show()

# 方法2：对角线附近放大（展示稀疏特性）- 修正版
plt.figure(figsize=(8, 6))
center = 250
window = 50
start, end = center-window, center+window
P_zoom = P[start:end, start:end]

plt.imshow(P_zoom, cmap='Blues', aspect='auto')
plt.colorbar(label='Transition Probability')

# 修正坐标轴 - 显示真实状态编号
# 设置坐标刻度位置和标签
tick_positions = np.arange(0, end-start+1, 10)  # 每10个状态显示一个刻度
true_states = np.arange(start, end+1, 10)       # 对应的真实状态编号

plt.xticks(tick_positions, true_states)
plt.yticks(tick_positions, true_states)

plt.title(f'Local View: States {start}-{end}', fontsize=14, fontweight='bold')
plt.xlabel('Target State')
plt.ylabel('Current State')
plt.tight_layout()
plt.savefig('transition_matrix_zoom.png', dpi=300, bbox_inches='tight')
plt.show()

