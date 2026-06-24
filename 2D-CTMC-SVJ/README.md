# 2D-CTMC-SVJ 期权定价系统

**作者**: [Author]
**创建日期**: 2026-05-06
**Python**: 3.13+

基于二维连续时间马尔可夫链 (2D-CTMC) 近似的随机波动率跳跃扩散 (SVJ) 模型期权定价框架。

## 项目简介

本项目实现了 Mackay, Vachon & Cui (2023) 提出的二维 CTMC 近似方法，用于对 Heston 随机波动率 (SV) 模型和 Bates SVJ 模型下的欧式/美式期权进行高效定价。

核心思路：
- **Layer 1**: 将 CIR 方差过程离散化为 m 状态 CTMC，构造方差生成元 Q
- **Layer 2**: 对每个方差体制构造价格过程生成元 G_l
- **组合**: 构造 mN × mN 的二维生成元，通过矩阵指数进行时间演化
- **分裂**: 支持 Lie-Trotter 和 Strang splitting 进行方差/价格维度的时间步进

## 环境配置

```bash
pip install -r requirements.txt
```

### 依赖说明

| 包 | 版本要求 | 用途 |
|---|---|---|
| numpy | ≥ 1.24 | 矩阵运算核心 |
| scipy | ≥ 1.10 | 优化器 (L-BFGS-B)、数值积分 |
| pandas | ≥ 2.0 | 数据处理 |
| loguru | ≥ 0.7 | 日志系统 |
| matplotlib | ≥ 3.7 | 可视化 (可选) |
| pytest | ≥ 7.0 | 单元测试 |

## 快速开始

```bash
# 运行完整验证流程 (Phase 0-3)
python main.py

# 运行单元测试
python -m pytest test/ -v

# 收敛研究
python convergence_study.py --mode grid       # 网格收敛
python convergence_study.py --mode adaptive    # 自适应网格
python convergence_study.py --mode nsteps      # 时间步数收敛

# 多日批量校准
python batch_calibrate.py
```

## 项目结构

```
2D-CTMC-SVJ/
├── main.py                      # 主入口: Phase 0-3 完整验证
├── commonConfig.py              # 全局配置 (模型参数、网格、优化器)
├── batch_calibrate.py           # 多日批量校准脚本
├── convergence_study.py         # 收敛研究工具
│
├── src/                         # 核心模块
│   ├── grid_construction.py     # sinh/uniform/自适应 网格构建
│   ├── layer1_variance.py       # CIR 方差生成元 Q + 5点模板
│   ├── layer2_price.py          # 价格体制生成元 G_l + 漂移修正
│   ├── combined_generator.py    # mN×mN 组合生成元
│   ├── jump_generator.py        # 跳跃生成元 (Merton/Kou)
│   ├── option_pricing.py        # 欧式定价 (向量化 + Strang splitting)
│   ├── american_pricing.py      # 美式定价 (向量化)
│   ├── tensor_pricing.py        # 张量定价 (实验性)
│   ├── heston_analytical.py     # Heston 闭式解 + 批量定价
│   ├── svj_analytical.py        # SVJ 半闭式解 (Bates) + 批量定价
│   ├── calibration.py           # 两阶段校准 (向量化目标函数)
│   └── data_loader.py           # 数据加载
│
├── utility/                     # 工具函数
│   ├── file_utils.py            # 文件 I/O
│   └── visualization.py         # 绘图 (网格、误差、收敛)
│
├── test/                        # 测试
│   └── test_2d_ctmc.py          # 31 个单元测试
│
├── scripts/                     # 诊断和实验脚本
│   ├── diagnostic_bias.py       # 系统性偏差诊断
│   ├── diagnostic_correlation.py# 相关性影响诊断
│   ├── convergence_focused.py   # 集中收敛研究
│   ├── quick_convergence.py     # 快速收敛测试
│   ├── test_shift_fix.py        # X-shift 修复验证
│   └── test_tensor.py           # 张量定价验证
│
├── input/                       # 输入数据
│   ├── Futures_au2412.csv       # 黄金期货日数据 (161天)
│   └── Futures_put_options_au2412.csv  # Put期权数据 (758行)
│
├── result/                      # 运行结果输出
├── Reference/                   # 参考文献
├── doc/                         # 文档
│   ├── PROGRESS.md              # 详细进度记录
│   └── theory.md                # 理论笔记
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 运行流程说明

`main.py` 包含 4 个阶段：

| Phase | 功能 | 说明 |
|---|---|---|
| Phase 0 | 数据加载与校准 | 模拟数据或真实市场数据 (au2412)，两阶段校准 Heston → SVJ |
| Phase 1 | Heston 欧式验证 | CTMC vs 闭式解，对比 Lie/Strang splitting |
| Phase 2 | SVJ 欧式验证 | CTMC-SVJ vs Bates 半闭式解 |
| Phase 3 | 美式期权定价 | SV/SVJ 模型下的美式 Put 定价 + 提前行权溢价 |

## 当前精度 (40×200 网格, n_steps=400)

| K | CTMC_SVJ | SVJ 解析 | 相对误差 |
|---|---|---|---|
| 95 | 4.2211 | 3.9901 | 5.8% |
| 100 | 6.1165 | 6.2289 | **1.8%** |
| 105 | 8.5161 | 9.0639 | -6.0% |

ATM 附近精度良好，翼部 (deep OTM/ITM) 误差较大。
网格收敛研究表明误差不随网格增密单调下降，根因为 CIR 方差生成元系统性偏差（详见 `doc/PROGRESS.md`）。

## 已完成成果

- 31 个单元测试全部通过
- 校准向量化加速 15.1x
- au2412 黄金期货 69 日批量校准 (SVJ RMSE 中位数 0.185)
- Heston/SVJ put-call parity 验证通过
- 期货模式 (`underlying_type='futures'`) 验证通过

## 下一步研究方向

详见 `doc/PROGRESS.md` 的"待完成事项"部分。核心方向：

1. **方差生成元质量改善** — 网格收敛研究已排除分辨率不足，根因在 CIR 生成元系统性偏差
2. **对比 1D-CTMC 实现** — 确认 `D:\ESG Projects\CTMC\1D-CTMC` 的生成元离散化方法一致性
3. **新数据接入** — Wind 终端高频数据 (tick期权 + 分钟期货)
4. **3/2 模型扩展** — `commonConfig.py` 中已有参数配置
5. **Kou 跳跃扩散** — `jump_generator.py` 已支持，待验证

## 参考文献

- Mackay, D., Vachon, A., & Cui, Z. (2023). *Continuous-Time Markov Chain Approximation for Two-Dimensional Stochastic Processes*
- 项目 `Reference/` 目录中的论文
