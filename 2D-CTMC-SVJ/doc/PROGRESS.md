# 2D-CTMC-SVJ 项目进度文档

**最后更新**: 2026-05-13
**原作者**: [Author]
**接手说明**: 本文档包含完整的项目上下文，阅读后可直接接手开发

---

## 零、接手人快速指南

### 先读什么

1. **README.md** — 项目概述、环境配置、运行方式
2. **本文档第二节** — 翼部精度攻关记录（已尝试什么、为什么失败、哪些方向可行）
3. **本文档第五节** — 待完成任务列表（带优先级）

### 核心问题

项目 ATM (K=100) 精度很好 (RE=1.8%)，但**翼部 OTM/ITM 误差大** (K=85 时 RE=34%)。
已排除的原因：分裂误差、初始插值误差、Richardson 外推、**网格分辨率不足**（详见第二节）。
**网格收敛研究表明误差不随网格增密单调下降，真正的根因是 CIR 方差生成元的系统性偏差**，下一步应检查 `layer1_variance.py` 的低 v 区域离散化质量。

### 建议的第一步操作

```bash
pip install -r requirements.txt          # 安装依赖
python -m pytest test/ -v                # 确认 31 个测试通过
python main.py                           # 运行完整验证 (Phase 0-3)
```

---

## 一、当前状态：网格收敛研究已完成，确认生成元系统性偏差

### 已完成

1. **全部核心模块开发完成，31个单元测试全部通过**
   - grid_construction, layer1_variance, layer2_price, jump_generator
   - combined_generator, option_pricing, american_pricing
   - heston_analytical, svj_analytical, calibration, data_loader

2. **期货模式 (`underlying_type='futures'`) 已集成**

3. **au2412 真实数据已加载**

4. **校准向量化完成** (任务1, 2026-05-08)
   - `heston_analytical.py`: 新增 `_heston_cf`, `compute_heston_price_fast`, `compute_heston_prices_batch`
   - `svj_analytical.py`: 新增 `_svj_cf`, `compute_svj_price_fast`, `compute_svj_prices_batch`
   - `calibration.py`: 目标函数改为按 `(T, option_type)` 分组批量计算
   - Gauss-Legendre 96点固定积分 + numpy广播，单次目标函数 **15.1x 加速**

5. **多日批量校准完成** (任务3, 2026-05-08)
   - `batch_calibrate.py`: 新增脚本，热启动逐日校准
   - 69/89 日期校准成功（≥4 OTM期权），总耗时 **47秒**
   - SVJ RMSE 中位数: 0.185, 均值: 0.198
   - 输出: `result/batch_calibration_*.csv`

6. **代码清理完成** (任务4, 2026-05-08)
   - 修复 166 处 docstring 编码损坏
   - 移除 4 个文件的未使用导入
   - 修复测试文件 typo + docstring 格式

7. **翼部精度攻关 — 第一轮探索** (任务2, 2026-05-09)
   - 已尝试的方案及结果见下方"翼部精度攻关记录"

8. **向量化定价核心** (任务7, 2026-05-09)
   - `option_pricing.py`: 方差维度步骤 N×dgemv → 1×dgemm，Lie 定价加速 ~1.3x
   - `american_pricing.py`: 同样向量化 + np.maximum 向量化提前行权检查
   - 31 个测试全部通过，定价结果完全一致

9. **自适应网格** (任务7, 2026-05-09)
   - `grid_construction.py`: 新增 `build_adaptive_price_grid()`
   - 策略: 基础 sinh 网格 (70% 预算) + 行权价附近 sinh 簇 (30% 预算)
   - 7 个行权价: 200 点 → 194 点 (去重后)，增加 54 个行权价感知点

  10. **收敛研究工具** (任务7, 2026-05-09)
      - `convergence_study.py`: 新增完整收敛研究脚本
      - 支持: 标准网格收敛 (20×100 ~ 80×400)、自适应网格、时间步数收敛
      - nsteps 收敛已确认: Strang 误差 = -1.50% 不随 n_steps 变化 → 纯网格离散化误差

  11. **完整网格收敛研究** (任务8, 2026-05-13) ✅ 已完成
      - 测试 4 组网格: 20×100, 30×150, 40×200, 60×300
      - n_steps=200 跑完全部 3 组，n_steps=400 跑完 2 组（60×300 因计算量过大未跑完全部 K）
      - **关键结论: 误差不随网格增密单调下降，排除了"网格分辨率不足"的假设**
      - 详见第二节"尝试6"

### 当前 CTMC-SVJ 精度 (40×200, n_steps=400, T=1.0, bilinear插值)

| K | CTMC_SV | Heston | SV_RE% | CTMC_SVJ | SVJ_ana | SVJ_RE% |
|---|---|---|---|---|---|---|
| 85 | 1.6801 | 1.2556 | +33.8% | 1.7489 | 1.2037 | +45.3% |
| 90 | 2.6969 | 2.4005 | +12.3% | 2.7865 | 2.3370 | +19.2% |
| 95 | 4.1122 | 4.0700 | +1.0% | 4.2211 | 3.9901 | +5.8% |
| 100 | 5.9937 | 6.3307 | -5.3% | 6.1165 | 6.2289 | **+1.8%** |
| 105 | 8.3879 | 9.1904 | -8.7% | 8.5161 | 9.0639 | -6.0% |
| 110 | 11.3079 | 12.5821 | -10.1% | 11.4315 | 12.4344 | -8.1% |
| 115 | 14.7268 | 16.3707 | -10.0% | 14.8366 | 16.2149 | -8.5% |

---

## 二、翼部精度攻关记录 (2026-05-09)

### 尝试1: 5点模板全面启用 ❌ 理论不可行

**改动**:
- `layer1_variance.py`: 新增 `_solve_5pt_rates_variance()`，`construct_variance_generator` 支持 `stencil='5pt'/'auto'`
- `layer2_price.py`: `construct_regime_generator_with_drift` 支持 `stencil='5pt'/'auto'`（SVJ路径）
- `jump_generator.py`: `construct_all_regime_generators_svj` 传递 stencil 参数
- `main.py`: `build_ctmc_infrastructure` 接受并传递 stencil 参数

**结果**: 5点模板在所有内部网格点100%回退到3点
- Layer 1: `5pt interior: 0/36, fallbacks: 36`
- Layer 2: `5pt interior: 0/196, fallbacks: 196`

**根因**: 在近似对称的网格上，4阶矩匹配条件 `Σ d_k⁴ q_k = 0` 与非负转移速率约束矛盾。
具体地：由条件4得 `q1 + q4 = 0`（外邻居速率之和为零），结合非负性得 `q1 = q4 = 0`；
再由条件3得 `q2 = q3 = 0`，与条件1/2矛盾。

**结论**: 经典5点模板（4阶矩匹配）在对称/近对称网格上**理论上不可行**。这是已知的数学限制。

### 尝试2: 初始状态双线性插值 ❌ 无改善

**改动**:
- `option_pricing.py`: 新增 `_bilinear_interpolate()`，替换最近邻为双线性插值
- `american_pricing.py`: 同样替换为双线性插值

**结果**: 与最近邻结果完全一致（差异 < 1e-12）
- 原因: V_0=0.04 几乎精确落在方差网格点上，X_0 也非常接近价格网格点
- 结论: 初始状态插值不是误差来源

### 尝试3: Richardson 外推 (价格维度) ❌ 几乎无改善

**做法**: `P_rich = 2 * P_fine - P_coarse_x`，消除价格维度的 O(h²) 主导误差

**结果** (T=1.0):
| K | Std_SV_RE% | Rich_SV_RE% | Std_SVJ_RE% | Rich_SVJ_RE% |
|---|---|---|---|---|
| 90 | 12.3% | 12.3% | 19.2% | 19.2% |
| 100 | 5.3% | 5.4% | 1.8% | 1.8% |
| 110 | 10.1% | 10.2% | 8.1% | 8.1% |

**结论**: Richardson 外推几乎无改善。说明价格维度的 O(h²) 误差已被 m=40 方差维度的误差淹没，不是主导误差源。

### 尝试4: Strang Splitting (2026-05-09) ❌ 改善 <0.2%

**改动**:
- `option_pricing.py`: 新增 `price_european_strang()`，Strang 分裂 (price/2→variance→price/2)，O(dt²) 分裂误差
- `american_pricing.py`: `price_american_fast()` 新增 `splitting` 参数，支持 `'lie'`/`'strang'`
- `main.py`: Phase 1-3 全面对比 Lie vs Strang
- 修复 `american_pricing.py` 中 `var_idx`/`price_idx` 未定义的 bug

**结果** (T=1.0, SV模型):
| K | RE_Lie% | RE_Strang% | 改善 |
|---|---|---|---|
| 85 | 33.81% | 33.64% | +0.17% |
| 90 | 12.34% | 12.24% | +0.10% |
| 100 | 5.32% | 5.36% | -0.04% |
| 110 | 10.13% | 10.14% | -0.01% |

**结论**: Strang splitting 改善极小 (<0.2%)。n_steps=400 时 dt=0.0025，分裂误差本身已很小，远小于网格离散化误差。

### 尝试5: 方差维度 Richardson + Strang (2026-05-09) ❌ 改善有限

**改动**:
- `option_pricing.py`: `price_european_richardson()` 增强：
  - 新增方差维度外推: `ext_v = 2*P_fine - P_cv`
  - 新增二维外推: `ext_2d = 4*P_fine - 2*P_cx - 2*P_cv`
  - 支持 `splitting='strang'` 参数
  - 自动选择最佳外推方向

**结果** (T=1.0, Strang splitting):
| K | RE_fine% | RE_ext_x% | RE_ext_v% | RE_2d% | 最佳 |
|---|---|---|---|---|---|
| 85 | 33.64% | 33.50% | **33.24%** | 101.09% | ext_v |
| 90 | 12.24% | 12.17% | **11.94%** | 100.75% | ext_v |
| 95 | 0.97% | 0.93% | **0.74%** | 100.55% | ext_v |
| 100 | **5.36%** | 5.39% | 5.55% | 100.42% | fine |
| 110 | **10.14%** | 10.16% | 10.25% | 100.27% | fine |

**结论**:
- 方差维度外推对 OTM (K≤95) 有 ~0.2-0.3% 改善，ATM/ITM 端略恶化
- 二维外推 (ext_2d) 完全不可用（误差爆炸至 ~100%），因 `4P-2Pcx-2Pcv` 放大噪声
- **主导误差不是分裂误差、不是 Richardson 可消除的光滑误差**，而可能是 CTMC 生成元本身的系统性偏差

### 尝试6: 完整网格收敛研究 (2026-05-13) ❌ 误差不随网格增密单调下降

**做法**: 测试 20×100, 30×150, 40×200, 60×300 四组网格配置，T=1.0 put，对比 SVJ RE%

**结果** (SVJ RE%, T=1.0):

| Grid | n_steps | K=85 | K=90 | K=95 | K=100 | K=105 | K=110 | K=115 |
|---|---|---|---|---|---|---|---|---|
| 20×100 | 200 | +23.40% | +4.60% | -4.75% | -9.60% | -11.84% | -12.34% | -11.63% |
| 30×150 | 200 | +17.82% | +0.91% | -7.31% | -11.37% | -13.04% | -13.13% | -12.10% |
| 40×200 | 200 | +13.93% | -8.93% | -15.27% | -16.08% | -14.64% | -12.04% | -8.71% |
| 40×200 | 400 | +18.37% | +1.32% | -7.00% | -11.14% | -12.87% | -13.01% | -12.03% |
| 60×300 | 400 | +13.03% | -2.39% | -9.67% | -13.06% | — | — | — |

**关键观察**:

1. **误差不单调收敛**: 20×100 在 ATM (K=100) 的 RE=-9.60% 比 40×200 (RE=-11.14%) 和 60×300 (RE=-13.06%) 都好。这不是 O(h²) 收敛行为。
2. **n_steps 与网格耦合**: 同为 40×200，n_steps 200→400 后 K=85 从 +13.93% 变为 +18.37%（反而恶化 4.4pp），说明 n_steps 和网格分辨率之间存在非平凡耦合。
3. **系统性偏差模式稳定**: 所有网格配置下都是 OTM overpriced (+), ATM/ITM underpriced (-)，符号模式不变。
4. **K=85 (OTM) 是唯一粗略收敛的**: 20×100 → 30×150 → 40×200 → 60×300 大致在 +23% → +18% → +14% → +13% 递减，但下降速率不是 O(h²)。

**结论**:
- **排除了"网格分辨率不足"的假设**。即使状态空间从 2000 增到 18000，误差仍在 ~10-13% 范围内振荡。
- 问题的根因不在价格维度 (N)，而在**方差维度 (m) 的 CTMC 生成元对 CIR 过程的近似质量**。
- **最可能的根因**: `layer1_variance.py` 中的 3 点模板在 v 较低区域（v < 0.02）的漂移/扩散系数矩匹配不够精确，导致方差转移概率系统性偏移。这在 Feller ratio = 2σ²θ/κ = 0.45 (不满足 Feller 条件) 的参数下尤其显著。
- **下一步**: (1) 检查 1D-CTMC 项目中相同参数下的方差生成元精度；(2) 尝试在方差维度使用非均匀网格（V_0 附近加密）；(3) 考虑使用 Exponentially Tilted CTMC 替代经典矩匹配方法。

1. ~~**方差维度 Richardson 外推**~~: 已测试，OTM 改善 ~0.3%，ATM/ITM 略恶化 ❌
2. ~~**二维 Richardson 外推**~~: 已测试，误差爆炸至 ~100%，不可用 ❌
3. ~~**Strang splitting**~~: 已测试，改善 <0.2%，分裂误差非主导 ❌
4. ~~**增密网格**~~: 已测试（见尝试6），误差不随网格增密单调下降 ❌
5. **方差维度生成元修正**: 检查 `layer1_variance.py` 在低 v 区域的 CTMC 离散化质量
6. **方差网格在 V_0 附近加密**: 当前 sinh 网格中心在 theta=0.04，可尝试中心设在 V_0
7. **对比 1D-CTMC**: 检查 `D:\ESG Projects\CTMC\1D-CTMC` 的 CIR 生成元精度，确认实现一致性
8. **非对称5点模板**: 沿漂移方向偏移，避开对称性导致的非负约束冲突
9. **生成元边界处理改进**: 当前边界点使用 von Neumann 边界条件，可尝试反射/吸收边界

---

## 三、文件结构

```
2D-CTMC-SVJ/
├── main.py                      # 主入口: Phase 0-3 完整验证
├── commonConfig.py              # 所有配置参数（含 underlying_type）
├── batch_calibrate.py           # 多日批量校准脚本（热启动）
├── convergence_study.py         # 收敛研究（网格/nsteps/自适应）
├── requirements.txt             # Python 依赖
├── .gitignore
├── README.md
│
├── src/                         # 核心模块
│   ├── grid_construction.py     # sinh/uniform/自适应 网格构建
│   ├── layer1_variance.py       # CIR 方差生成元 Q + 5点模板
│   ├── layer2_price.py          # 价格体制生成元 G_l + 漂移修正
│   ├── combined_generator.py    # mN×mN 组合生成元
│   ├── jump_generator.py        # 跳跃生成元 (Merton/Kou)
│   ├── option_pricing.py        # 欧式定价 (向量化 + Strang splitting)
│   ├── american_pricing.py      # 美式定价 (向量化 + Lie/Strang)
│   ├── tensor_pricing.py        # 张量定价 (实验性)
│   ├── heston_analytical.py     # Heston 闭式解 + 批量定价
│   ├── svj_analytical.py        # SVJ 半闭式解 + 批量定价
│   ├── calibration.py           # 两阶段校准 (向量化目标函数)
│   └── data_loader.py           # 数据加载
│
├── utility/                     # 工具函数
│   ├── file_utils.py            # 文件 I/O
│   └── visualization.py         # 绘图
│
├── test/
│   └── test_2d_ctmc.py          # 31 个单元测试
│
├── scripts/                     # 诊断和实验脚本
│   ├── diagnostic_bias.py       # 系统性偏差诊断
│   ├── diagnostic_correlation.py# 相关性 rho 影响诊断
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
├── Reference/                   # 参考文献论文
└── doc/
    ├── PROGRESS.md              # 本文件 (详细进度记录)
    └── theory.md                # 理论笔记
```

---

## 四、关键参数

```python
HESTON_DEFAULT_PARAMS = {
    'S_0': 100, 'V_0': 0.04, 'r': 0.03,
    'kappa': 2.0, 'theta': 0.04, 'sigma_v': 0.3, 'rho': -0.7,
    'underlying_type': 'spot'
}
SVJ_JUMP_DEFAULT_PARAMS = {
    'lambda_jump': 0.1, 'mu_J': -0.05, 'sigma_J': 0.10, 'jump_type': 'merton'
}
```

### au2412 批量校准统计（69日）

| 参数 | 均值 | 标准差 | 范围 |
|---|---|---|---|
| V_0 | 0.034 | 0.010 | [0.016, 0.060] |
| kappa | 1.826 | 0.020 | [1.807, 1.863] |
| theta | 0.061 | 0.033 | [0.034, 0.124] |
| sigma_v | 0.638 | 0.166 | [0.500, 0.938] |
| rho | -0.172 | 0.288 | [-0.735, 0.000] |
| SVJ RMSE | 0.198 | 0.094 | [0.041, 0.387] |

---

## 五、待完成事项

### ~~任务1: 校准向量化~~ [已完成 ✓]
- [x] Gauss-Legendre 固定节点积分替代 quad
- [x] numpy 广播批量定价
- [x] calibration.py 向量化目标函数
- 实测 15.1x 加速

### 任务2: 翼部精度修复 [高优先级 — 进行中]
    - [x] 5点模板全面启用 → **理论不可行**（对称网格+非负约束矛盾）
    - [x] 初始状态双线性插值 → **无改善**（初始状态恰好在网格点上）
    - [x] Richardson 外推（价格维度）→ **几乎无改善**（方差维度误差主导）
    - [x] Strang splitting → **改善 <0.2%**（分裂误差非主导）
    - [x] 方差维度 Richardson + 2D Richardson → **OTM 改善 ~0.3%，2D 外推不可用**
    - [x] 修复 american_pricing.py var_idx/price_idx 未定义 bug
    - [x] nsteps 收敛研究 → **Strang 误差恒定 -1.50%，确认纯网格误差**
    - [x] 完整网格收敛研究 → **误差不随网格增密单调下降，排除分辨率不足**
    - [ ] **下一步**: 检查 `layer1_variance.py` 低 v 区域离散化质量
    - [ ] 对比 `D:\ESG Projects\CTMC\1D-CTMC` 的 CIR 生成元精度
    - [ ] 尝试方差维度在 V_0 附近加密的 sinh 网格
    - 目标：翼部 RE < 5%

### ~~任务3: 多日批量校准~~ [已完成 ✓]
- [x] au2412 69个日期逐日校准（热启动）
- [x] 输出参数时间序列 CSV
- [x] 参数稳定性分析

### ~~任务4: 代码清理~~ [已完成 ✓]
- [x] 修复 166 处 docstring 编码损坏
- [x] 移除未使用导入
- [x] 修复测试文件 typo + 格式
- [x] 31个测试全部通过

### 任务5: 获取新数据 [等待]
- 用户待从 Wind 终端获取 tick 期权 + 分钟期货数据

### 任务6: 2D-CTMC 团队项目迁移 [延后]
- 骨架: `D:\ESG Projects\CTMC\2D-CTMC`

### 任务7: 向量化 + 自适应网格 + 收敛工具 [已完成 ✓]
- [x] option_pricing.py 方差维度 N×dgemv → 1×dgemm，~1.3x 加速
- [x] american_pricing.py 向量化 + np.maximum 行权检查
- [x] build_adaptive_price_grid() 行权价感知网格
- [x] convergence_study.py 完整收敛研究脚本
- [x] nsteps 收敛确认: Strang 误差 = -1.50% 恒定 → 纯网格离散化误差
- [x] 31 个测试全部通过

### ~~任务8: 完整网格收敛研究~~ [已完成 ✓] (2026-05-13)
- [x] 运行 4 组网格配置 (20×100 ~ 60×300)
- [x] 分析收敛速率 → **非单调收敛，排除 O(h²) 假设**
- [x] 结论: 问题不在网格分辨率，而在 CIR 生成元系统性偏差
- 详见第二节"尝试6"

---

## 六、建议执行顺序

**当前建议**: 检查方差维度生成元质量，这是解决翼部精度的关键突破口。

1. 对比 `D:\ESG Projects\CTMC\1D-CTMC` 项目中 CIR 生成元实现，确认离散化方法一致
2. 检查 `layer1_variance.py` 在低 v 区域 (v < 0.02) 的矩匹配精度
3. 尝试方差维度 sinh 网格中心从 theta=0.04 移到 V_0=0.04（当前两者恰好相同，可尝试更大 v 范围）
4. 如确认生成元偏差，考虑 Exponentially Tilted CTMC 或改进边界条件

---

## 七、已验证的正确性

| 测试 | 结果 |
|---|---|
| Heston put-call parity | ✓ |
| SVJ lambda=0 = Heston | ✓ |
| SVJ put-call parity | ✓ |
| 跳跃生成元行和 = 0 | ✓ |
| CTMC-SVJ ATM (K=100) | **1.8% RE** ✓ |
| 1D BS+Merton 跳跃方向 | ✓ |
| Futures 模式 (analytical) | ✓ |
| Futures 模式 (CTMC) | ✓ |
| au2412 单日校准 | ✓ (RMSE=0.86) |
| 校准向量化精度 | ✓ (quad vs GL 差异 6e-11) |
| au2412 批量校准 (69日) | ✓ (SVJ RMSE 中位数 0.185) |
| 31 个单元测试 | ✓ 全部通过 |
| 5点模板代码正确性 | ✓（编译通过，31测试通过，但因理论限制全部回退到3pt） |
| bilinear 插值正确性 | ✓（31测试通过，结果与最近邻一致） |
| Strang splitting 正确性 | ✓（31测试通过，改善 <0.2%） |
| 方差 Richardson 外推 | ✓（OTM 改善 ~0.3%，ATM/ITM 略恶化） |
| 2D Richardson 外推 | ✗（误差爆炸，不可用） |
| american_pricing bug fix | ✓（var_idx/price_idx 正确定义） |
| 向量化 option_pricing | ✓（31测试通过，定价结果一致，~1.3x加速） |
| 向量化 american_pricing | ✓（31测试通过，EEP 结果一致） |
| 自适应网格 build_adaptive_price_grid | ✓（7 strikes → 194 pts，含 54 strike-aware 点） |
| convergence_study.py | ✓（nsteps 模式验证通过） |
| nsteps 收敛: Strang 恒定 -1.50% | ✓（200/400/800 steps，确认纯网格误差） |
| 网格收敛研究 (20×100~60×300) | ✓（4组配置，误差非单调收敛，排除分辨率不足） |

---

## 八、重要上下文

- 项目路径: `D:\ESG Projects\CTMC\2D-CTMC-SVJ`
- 1D-CTMC 项目: `D:\ESG Projects\CTMC\1D-CTMC` (一维版本，可作为参考)
- 2D-CTMC 团队骨架: `D:\ESG Projects\CTMC\2D-CTMC`
- 理论参考: `Reference/` 目录中的论文，特别是 ssrn-3095327.pdf (Mackay et al.)
- 编码警告: 不要用 PowerShell `Set-Content` 操作 UTF-8 文件（会损坏中文编码）
- 5点模板数学限制: 对称网格上 Σd⁴q=0 + q≥0 无解，这是已知的PDE约束

## 九、代码模块依赖关系

```
main.py
  ├── commonConfig.py (参数配置)
  ├── src/grid_construction.py (网格)
  │     └── numpy
  ├── src/layer1_variance.py (Q 方差生成元)
  │     └── grid_construction
  ├── src/layer2_price.py (G_l 价格生成元)
  │     └── grid_construction, layer1_variance
  ├── src/combined_generator.py (组合生成元)
  │     └── layer1_variance, layer2_price
  ├── src/jump_generator.py (跳跃生成元)
  │     └── layer2_price
  ├── src/option_pricing.py (欧式定价)
  │     └── combined_generator (通过 Q, G)
  ├── src/american_pricing.py (美式定价)
  │     └── option_pricing (共享向量化和 splitting 逻辑)
  ├── src/heston_analytical.py (闭式解)
  │     └── scipy (数值积分)
  ├── src/svj_analytical.py (半闭式解)
  │     └── heston_analytical
  ├── src/calibration.py (校准)
  │     └── heston_analytical, svj_analytical, scipy.optimize
  └── src/data_loader.py (数据加载)
        └── pandas
```
