# Two-Dimensional Continuous-Time Markov Chain Approximation for Stochastic Volatility Models with Jumps

## 1. Introduction

### 1.1 Problem Statement

Pricing derivatives under stochastic volatility models — particularly those incorporating jumps — remains a central challenge in quantitative finance. The Heston (1993) model and its extension with Merton-type jumps, known as the Stochastic Volatility with Jumps (SVJ) or Bates (1996) model, are among the most widely used frameworks in both academic research and industry practice.

For **European** options, semi-closed-form solutions exist via Fourier inversion (Heston, 1993; Bates, 1996). However, for **American** options — which admit early exercise — no closed-form solution is available. Standard numerical approaches include:

- **Finite Difference Methods (FDM):** Solve the associated free-boundary PDE on a 2D grid. Suffers from numerical diffusion and requires careful treatment of boundary conditions.
- **Monte Carlo (MC) simulation:** Flexible but introduces stochastic noise. The Longstaff-Schwartz (2001) method for American options requires regression-based continuation value estimation, introducing additional bias.
- **Continuous-Time Markov Chain (CTMC) approximation:** Approximates the underlying diffusion process with a finite-state CTMC, reducing the pricing problem to matrix exponentiation.

This document presents the **Two-Dimensional CTMC (2D-CTMC)** framework based on Mackay, Vachon, and Cui (2023), extended to handle jump-diffusion processes. The method decomposes the 2D problem into two interacting one-dimensional CTMC layers via a decorrelation transformation, achieving computational efficiency while maintaining accuracy.

### 1.2 Key Advantages

1. **Deterministic pricing:** No simulation noise, unlike Monte Carlo methods.
2. **American option pricing with early exercise boundary:** The backward induction naturally produces the optimal exercise boundary $S^*(t, V)$ as a function of both time and variance level.
3. **Modular architecture:** The dual-layer decomposition separates variance dynamics (Layer 1) from price dynamics (Layer 2), enabling independent grid construction and generator design.
4. **Regime-dependent jump integration:** Jump components are added to each variance regime independently, preserving the block structure of the combined generator.

---

## 2. Model Specification

### 2.1 Heston Stochastic Volatility Model

The Heston model specifies the joint dynamics of the asset price $S_t$ and its instantaneous variance $V_t$ under the risk-neutral measure $\mathbb{Q}$:

$$
\frac{dS_t}{S_t} = (r - q)\,dt + \sqrt{V_t}\,dW_t^{(1)}
$$

$$
dV_t = \kappa(\theta - V_t)\,dt + \sigma_v\sqrt{V_t}\,dW_t^{(2)}
$$

where:
- $r$ is the risk-free rate,
- $q$ is the dividend yield (set to zero in this implementation),
- $\kappa > 0$ is the mean-reversion speed of variance,
- $\theta > 0$ is the long-run variance level,
- $\sigma_v > 0$ is the volatility of variance (vol-of-vol),
- $\text{Corr}(dW_t^{(1)}, dW_t^{(2)}) = \rho \in [-1, 1]$ is the leverage correlation.

**Parameters:** $\Theta_{\text{Heston}} = \{V_0, \kappa, \theta, \sigma_v, \rho\}$

### 2.2 SVJ (Bates) Model: Adding Jumps

The SVJ model extends Heston by adding a compound Poisson jump process to the asset price dynamics:

$$
\frac{dS_t}{S_t} = (r - \lambda \bar{k})\,dt + \sqrt{V_t}\,dW_t^{(1)} + dJ_t
$$

where:
- $J_t = \sum_{i=1}^{N_t} (e^{Z_i} - 1)$ is a compound Poisson process,
- $N_t \sim \text{Poisson}(\lambda t)$ with intensity $\lambda$,
- $Z_i \sim \mathcal{N}(\mu_J, \sigma_J^2)$ are i.i.d. log-jump sizes,
- $\bar{k} = \mathbb{E}[e^Z - 1] = e^{\mu_J + \sigma_J^2/2} - 1$ is the mean jump compensator.

The variance process $V_t$ remains unchanged from the Heston model. **Jumps affect only the asset price, not the variance.**

**Additional parameters:** $\Theta_{\text{Jump}} = \{\lambda, \mu_J, \sigma_J\}$

---

## 3. Decorrelation Transformation

### 3.1 Motivation

The 2D process $(S_t, V_t)$ is coupled through the correlation $\rho$. Direct CTMC approximation of the joint process would require constructing a generator on the full 2D state space of size $m \times N$, leading to a matrix of dimension $mN \times mN$ — computationally prohibitive.

The key insight from Mackay et al. (2023) is to apply a **decorrelation transformation** that decomposes the 2D process into two independent components, enabling a layer-by-layer construction.

### 3.2 Auxiliary Variable Definition

Define the **auxiliary process** $X_t$ via the transformation:

$$
X_t = \ln S_t - \rho \cdot \gamma(V_t)
$$

where $\gamma(v)$ is the **decorrelation function** defined by:

$$
\gamma(v) = \int_0^v \frac{\sigma_S(u)}{\sigma_V(u)}\,du
$$

For the Heston model, $\sigma_S(v) = \sqrt{v}$ and $\sigma_V(v) = \sigma_v\sqrt{v}$, giving:

$$
\gamma(v) = \int_0^v \frac{\sqrt{u}}{\sigma_v\sqrt{u}}\,du = \frac{v}{\sigma_v}
$$

and its derivative:

$$
\gamma'(v) = \frac{1}{\sigma_v}
$$

### 3.3 Independence of $(X_t, V_t)$

Applying Itô's lemma to $X_t = \ln S_t - \rho\gamma(V_t)$:

$$
dX_t = \mu_X(X_t, V_t)\,dt + \sigma_X(V_t)\,dW_t^*
$$

where $W_t^*$ is a Brownian motion **independent** of $W_t^{(2)}$ (which drives $V_t$), with:

$$
\sigma_X(v) = \sqrt{1 - \rho^2}\,\sqrt{v}
$$

$$
\mu_X(x, v) = r - \frac{v}{2} - \rho\,\psi(v)
$$

$$
\psi(v) = \frac{\kappa(\theta - v)}{\sigma_v}
$$

The independence of $X_t$ and $V_t$ (in the sense of their driving Brownian motions being independent) is the foundation of the dual-layer CTMC framework.

### 3.4 Price Recovery

The original price is recovered from the auxiliary variable via:

$$
S_t = \exp\big(X_t + \rho\,\gamma(V_t)\big) = \exp\!\Big(X_t + \frac{\rho\,V_t}{\sigma_v}\Big)
$$

---

## 4. Layer 1: Variance Process CTMC

### 4.1 CIR Process and Local Consistency

The variance process follows a Cox-Ingersoll-Ross (CIR) process:

$$
dV_t = \underbrace{\kappa(\theta - V_t)}_{\mu_V(V_t)}\,dt + \underbrace{\sigma_v\sqrt{V_t}}_{\text{diffusion}}\,dW_t^{(2)}
$$

We approximate $V_t$ by a finite-state CTMC $V_t^{(m)}$ on a grid $\mathcal{V} = \{v_1, v_2, \ldots, v_m\}$.

### 4.2 Generator Construction

The generator matrix $Q^{(m)} \in \mathbb{R}^{m \times m}$ is constructed using **local consistency conditions** (Eq. 5.3 in Mackay et al., 2023). The key idea: at each grid point $v_i$, the CTMC's instantaneous drift and variance must match those of the CIR process.

For **interior states** $i = 2, \ldots, m-1$, with $\delta_{i-1} = v_i - v_{i-1}$ and $\delta_i = v_{i+1} - v_i$:

$$
q_{i,i-1} = \frac{\sigma_V^2(v_i) - \delta_i \cdot \mu_V(v_i)}{\delta_{i-1}(\delta_{i-1} + \delta_i)}
$$

$$
q_{i,i+1} = \frac{\sigma_V^2(v_i) + \delta_{i-1} \cdot \mu_V(v_i)}{\delta_i(\delta_{i-1} + \delta_i)}
$$

$$
q_{i,i} = -(q_{i,i-1} + q_{i,i+1})
$$

For **boundary states** (Eq. 5.4):

**Lower boundary** ($i = 1$): Only upward transitions are allowed.

$$
q_{1,2} = \frac{\sigma_V^2(v_1)}{\delta_1^2} + \frac{\max(\mu_V(v_1),\, 0)}{\delta_1}
$$

**Upper boundary** ($i = m$): Only downward transitions are allowed.

$$
q_{m,m-1} = \frac{\sigma_V^2(v_m)}{\delta_{m-1}^2} + \frac{\max(-\mu_V(v_m),\, 0)}{\delta_{m-1}}
$$

### 4.3 Properties

The resulting generator $Q^{(m)}$ is a **tridiagonal** matrix satisfying:
- Off-diagonal entries $q_{i,j} \geq 0$ for $i \neq j$ (transition rates),
- Row sums $\sum_j q_{i,j} = 0$ for all $i$ (conservation of probability),
- The transition matrix is $P_V(\Delta t) = e^{Q^{(m)} \Delta t}$.

### 4.4 Grid Design: sinh Transformation

The variance grid uses a **sinh (hyperbolic sine) transformation** to concentrate points around the initial variance $V_0$ while maintaining adequate coverage at the boundaries:

$$
v_i = c + \alpha \cdot \sinh\big(c_2 u_i + c_1(1 - u_i)\big)
$$

where $u_i = i/(m-1)$, $c = V_0$ (center), and:

$$
c_1 = \text{arcsinh}\!\Big(\frac{v_{\min} - c}{\alpha}\Big), \quad c_2 = \text{arcsinh}\!\Big(\frac{v_{\max} - c}{\alpha}\Big)
$$

The parameter $\alpha$ controls the concentration: smaller $\alpha$ produces denser grids near $V_0$. The grid range $[v_{\min}, v_{\max}]$ is set to cover the relevant variance domain (e.g., $[0.008, 0.120]$).

---

## 5. Layer 2: Price Process CTMC

### 5.1 Regime-Dependent Generator Construction

Conditional on the variance being in state $v_l$ (i.e., $V_t^{(m)} = v_l$), the auxiliary process $X_t$ has constant diffusion coefficient $\sigma_X(v_l)$ and drift $\mu_X(x, v_l)$. We construct a separate CTMC for $X_t$ in each variance **regime** $l = 1, \ldots, m$.

For the log-price grid $\mathcal{X} = \{x_1, x_2, \ldots, x_N\}$, the regime-$l$ generator $G_l^{(N)} \in \mathbb{R}^{N \times N}$ is built using the same local consistency method, with the auxiliary process coefficients:

$$
\mu_X(v_l) = r - \frac{v_l}{2} - \rho \cdot \frac{\kappa(\theta - v_l)}{\sigma_v}
$$

$$
\sigma_X^2(v_l) = (1 - \rho^2)\,v_l
$$

Note that for each regime $l$, the drift and diffusion coefficients are **constant** (independent of $x$), so the generator has a uniform structure within each regime.

### 5.2 Regime-Dependent Price Grid

The log-price grid $\mathcal{X}$ is shared across all regimes but constructed to accommodate the full variance range. The grid center is:

$$
x_0 = \ln S_0 - \rho \cdot \gamma(V_0) = \ln S_0 - \frac{\rho\,V_0}{\sigma_v}
$$

The grid range is determined by the maximum volatility:

$$
[x_{\min},\, x_{\max}] = \big[x_0 - 3\sqrt{v_{\max} T},\; x_0 + 3\sqrt{v_{\max} T}\big]
$$

A sinh transformation concentrates points around $x_0$.

### 5.3 Resulting Structure

The Layer 2 construction produces:
- $m$ generator matrices $G_1^{(N)}, G_2^{(N)}, \ldots, G_m^{(N)}$, each of size $N \times N$,
- Each $G_l^{(N)}$ is tridiagonal, matching the dynamics of $X_t$ in regime $l$.

---

## 6. SVJ Extension: Regime-Dependent Jump Generators

### 6.1 Jump Component in the Auxiliary Frame

Under the SVJ model, the auxiliary process acquires an additional jump component. In log-space, jumps in $S_t$ translate directly to jumps in $X_t$ (since $X_t = \ln S_t - \rho\gamma(V_t)$ and $\gamma(V_t)$ is continuous).

The **compensated drift** for the auxiliary process becomes:

$$
\mu_X^{\text{SVJ}}(v) = \mu_X^{\text{SV}}(v) - \lambda\bar{k}
$$

where $\bar{k} = e^{\mu_J + \sigma_J^2/2} - 1$ is the mean jump size compensator.

### 6.2 Jump Generator Matrix $\Lambda_J$

For a log-price grid $\{x_1, \ldots, x_N\}$, the jump from state $x_i$ to state $x_j$ corresponds to a log-jump of size $Z = x_j - x_i$. The jump rate is computed by integrating the jump size density over a bin centered at $x_j$:

$$
\Lambda_J[i, j] = \lambda \int_{z_{j}^{\text{lo}}}^{z_{j}^{\text{hi}}} f_Z(z)\,dz
$$

where $z_j^{\text{lo}}$ and $z_j^{\text{hi}}$ are the bin boundaries around $x_j$ relative to $x_i$, and $f_Z$ is the jump size density.

For **Merton (log-normal) jumps** with $Z \sim \mathcal{N}(\mu_J, \sigma_J^2)$:

$$
\Lambda_J[i, j] = \lambda \Big[\Phi\!\Big(\frac{z_j^{\text{hi}} - \mu_J}{\sigma_J}\Big) - \Phi\!\Big(\frac{z_j^{\text{lo}} - \mu_J}{\sigma_J}\Big)\Big]
$$

where $\Phi$ is the standard normal CDF.

The diagonal elements are set to ensure row sums equal zero:

$$
\Lambda_J[i, i] = -\sum_{j \neq i} \Lambda_J[i, j]
$$

### 6.3 Combined SVJ Generator

For each regime $l$, the complete SVJ generator is:

$$
G_l^{\text{SVJ}} = G_l^{\text{SVJ-drift}} + \Lambda_J
$$

where $G_l^{\text{SVJ-drift}}$ is the tridiagonal diffusion generator with the compensated drift $\mu_X^{\text{SVJ}}(v_l)$, and $\Lambda_J$ is the (shared) jump matrix.

The combined generator $G_l^{\text{SVJ}}$ is **no longer tridiagonal** — the jump matrix introduces transitions between all states — but retains the zero-row-sum property.

### 6.4 Critical Design Note

The jump matrix $\Lambda_J$ is **independent of the variance regime** $l$. This is because:
1. Jumps affect only the asset price (not variance), so the jump size distribution is regime-independent.
2. The drift correction $-\lambda\bar{k}$ is absorbed into the tridiagonal generator construction.
3. Only the bin boundaries depend on the grid, which is shared across regimes.

This results in a significant computational saving: $\Lambda_J$ is computed once and reused for all $m$ regimes.

---

## 7. European Option Pricing

### 7.1 Fast Algorithm (Algorithm 10.3)

The European option pricing exploits the block structure of the 2D CTMC. Define the value matrix:

$$
B \in \mathbb{R}^{m \times N}, \quad B[l, n] = \text{option value in state } (v_l, x_n)
$$

**Terminal condition** ($t = T$):

$$
B[l, n] = \text{payoff}\big(S(v_l, x_n)\big)
$$

where $S(v_l, x_n) = \exp(x_n + \rho\gamma(v_l))$ is the recovered price.

**Backward induction** (for $z = M-1, \ldots, 0$ with $\Delta t = T/M$):

$$
\tilde{E}[l, :] = P_X^{(l)} \cdot B[l, :] \quad \text{(price transition within regime } l\text{)}
$$

$$
B[:, n] = P_V \cdot \tilde{E}[:, n] \quad \text{(variance transition across regimes)}
$$

$$
B \leftarrow e^{-r\Delta t} \cdot B \quad \text{(discounting)}
$$

where $P_X^{(l)} = e^{G_l \Delta t}$ and $P_V = e^{Q^{(m)} \Delta t}$ are pre-computed transition matrices.

**Extraction:** At $t = 0$, locate the initial state $(l_0, n_0)$ corresponding to $(V_0, X_0)$:

$$
V_{\text{price}}^{\text{European}} = B[l_0, n_0]
$$

### 7.2 Computational Complexity

For each time step:
- $m$ matrix-vector products of size $N$ (price transitions): $O(mN^2)$
- $N$ matrix-vector products of size $m$ (variance transitions): $O(m^2 N)$

Total: $O(M(mN^2 + m^2 N))$ per option, after pre-computing $m + 1$ matrix exponentials.

This is dramatically cheaper than the regular algorithm which requires exponentiating the full $mN \times mN$ combined generator.

---

## 8. American Option Pricing

### 8.1 Backward Induction with Early Exercise

American option pricing extends the European algorithm by adding an **early exercise check** at each time step. The backward induction becomes:

**Terminal condition:**

$$
B^{\text{am}}[l, n] = \text{payoff}(S(v_l, x_n))
$$

**Backward step** (for $z = M-1, \ldots, 0$):

1. **Transition:**

$$
\tilde{E}[l, :] = P_X^{(l)} \cdot B^{\text{am}}[l, :], \quad B^{\text{am}}[:, n] = P_V \cdot \tilde{E}[:, n]
$$

2. **Discount:**

$$
B^{\text{am}} \leftarrow e^{-r\Delta t} \cdot B^{\text{am}}
$$

3. **Early exercise** (for each state $(l, n)$):

$$
B^{\text{am}}[l, n] = \max\!\Big(B^{\text{am}}[l, n],\; \text{payoff}(S(v_l, x_n))\Big)
$$

### 8.2 Early Exercise Boundary

The algorithm naturally identifies the **early exercise boundary** $S^*(t, v_l)$ for each variance regime. For a put option:

$$
S^*(t, v_l) = \max\big\{S(v_l, x_n) : n \text{ such that } \text{payoff}(S(v_l, x_n)) > B^{\text{am}}[l, n]\big\}
$$

This produces a **2D exercise boundary surface** $S^*(t, V)$ that depends on both time and variance level — a unique output of the 2D CTMC method that is not readily available from Monte Carlo or 1D methods.

### 8.3 Early Exercise Premium

The **early exercise premium (EEP)** is defined as:

$$
\text{EEP} = V_{\text{price}}^{\text{American}} - V_{\text{price}}^{\text{European}}
$$

By running both American and European backward induction simultaneously (the European path skips the exercise check), the EEP is obtained at no additional computational cost.

### 8.4 Properties

For a put option, the EEP is always non-negative:

$$
\text{EEP} \geq 0
$$

and increases with:
- **Time to maturity** $T$ (more exercise opportunities),
- **Moneyness** (ATM options have larger relative EEP),
- **Volatility** (higher variance increases optionality value).

---

## 9. Grid Construction

### 9.1 Sinh Grid (Tavella-Randall)

Both Layer 1 and Layer 2 grids use the sinh transformation:

$$
g_i = c + \alpha \cdot \sinh\!\big(c_2 u_i + c_1(1 - u_i)\big), \quad u_i = \frac{i}{K-1},\; i = 0, \ldots, K-1
$$

$$
c_1 = \text{arcsinh}\!\Big(\frac{a - c}{\alpha}\Big), \quad c_2 = \text{arcsinh}\!\Big(\frac{b - c}{\alpha}\Big)
$$

**Properties:**
- The grid is symmetric around the center $c$ when the domain $[a, b]$ is symmetric.
- Smaller $\alpha$ produces denser spacing near $c$.
- As $\alpha \to \infty$, the grid degenerates to a uniform grid.

### 9.2 Variance Grid (Layer 1)

- Grid center: $c = V_0$
- Grid range: $[v_{\min}, v_{\max}]$, typically $[0.008, 0.120]$ for equity markets
- Number of points: $m = 40$
- Concentration parameter: $\alpha = 0.5$

### 9.3 Price Grid (Layer 2)

- Grid center: $c = \ln S_0 - \rho\gamma(V_0)$
- Grid range: $[c - 3\sqrt{v_{\max} T},\; c + 3\sqrt{v_{\max} T}]$
- Number of points: $N = 200$
- Concentration parameter: $\alpha = 1.0$

---

## 10. Two-Stage Calibration

### 10.1 Stage 1: Heston Parameters

Calibrate the Heston parameters $\hat{\Theta}_{\text{Heston}} = \{\hat{V}_0, \hat{\kappa}, \hat{\theta}, \hat{\sigma}_v, \hat{\rho}\}$ by minimizing the sum of squared pricing errors against observed European option prices:

$$
\hat{\Theta}_{\text{Heston}} = \arg\min_{\Theta} \sum_{i=1}^{N_{\text{obs}}} \big(V_i^{\text{market}} - V_i^{\text{Heston}}(\Theta)\big)^2
$$

This uses the Heston semi-closed-form solution for fast evaluation. The optimization employs L-BFGS-B with multiple random restarts to avoid local minima.

### 10.2 Stage 2: Jump Parameters

With $\hat{\Theta}_{\text{Heston}}$ fixed, calibrate the jump parameters $\hat{\Theta}_{\text{Jump}} = \{\hat{\lambda}, \hat{\mu}_J, \hat{\sigma}_J\}$:

$$
\hat{\Theta}_{\text{Jump}} = \arg\min_{\Theta_J} \sum_{i=1}^{N_{\text{obs}}} \big(V_i^{\text{market}} - V_i^{\text{SVJ}}(\hat{\Theta}_{\text{Heston}}, \Theta_J)\big)^2
$$

This two-stage approach is computationally efficient and avoids the ill-conditioning that arises when calibrating all 8 parameters simultaneously.

---

## 11. Numerical Results

### 11.1 Model Parameters (Default Configuration)

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Initial price | $S_0$ | 100 |
| Initial variance | $V_0$ | 0.04 |
| Risk-free rate | $r$ | 0.03 |
| Mean-reversion speed | $\kappa$ | 2.0 |
| Long-run variance | $\theta$ | 0.04 |
| Vol-of-vol | $\sigma_v$ | 0.3 |
| Leverage correlation | $\rho$ | -0.7 |
| Jump intensity | $\lambda$ | 0.1 |
| Log-jump mean | $\mu_J$ | -0.05 |
| Log-jump std | $\sigma_J$ | 0.1 |

### 11.2 European Option Accuracy (SVJ, T = 1.0)

| Strike | Analytical | CTMC | RE (%) |
|--------|-----------|------|--------|
| 90 | 2.337 | 2.846 | 21.77 |
| 95 | 3.990 | 4.307 | 7.94 |
| **100** | **6.229** | **6.235** | **0.10** |
| 105 | 9.064 | 8.672 | 4.32 |
| 110 | 12.434 | 11.628 | 6.49 |

ATM accuracy is excellent (0.10% relative error). Deep ITM/OTM wings show systematic bias from the CTMC approximation.

### 11.3 American Option Results (SVJ Put, K = 100)

| Maturity | American | European | EEP | EEP (%) |
|----------|----------|----------|-----|---------|
| T = 0.25 | 3.799 | 3.700 | 0.099 | 2.68 |
| T = 0.50 | 5.150 | 4.837 | 0.314 | 6.48 |
| T = 1.00 | 6.963 | 6.235 | 0.728 | 11.68 |

### 11.4 Early Exercise Boundary (ATM Put, K = 100, T = 1.0)

| $t/T$ | $V = 0.008$ | $V = 0.037$ | $V = 0.065$ | $V = 0.094$ | $V = 0.120$ |
|-------|-------------|-------------|-------------|-------------|-------------|
| 0.00 | N/A | 78.19 | 76.65 | 73.71 | 87.25 |
| 0.20 | N/A | 79.68 | 78.09 | 75.09 | 88.86 |
| 0.40 | N/A | 81.94 | 79.55 | 76.49 | 89.68 |
| 0.60 | N/A | 84.26 | 81.79 | 78.63 | 91.34 |
| 0.80 | N/A | 88.25 | 84.85 | 81.57 | 93.89 |
| 0.99 | N/A | 96.73 | 96.45 | 95.29 | 99.24 |

The boundary increases as $t \to T$ (approaching $K$ at expiry) and varies significantly with the variance level, demonstrating the importance of the 2D treatment.

---

## 12. Implementation Architecture

```
2D-CTMC-SVJ/
├── commonConfig.py          # Model parameters, grid config, optimizer settings
├── main.py                  # 4-phase pipeline (calibration → Heston → SVJ → American)
├── src/
│   ├── grid_construction.py # sinh/uniform grid builders
│   ├── layer1_variance.py   # CIR generator Q^(m) via local consistency
│   ├── layer2_price.py      # Auxiliary process generators G_l^(N)
│   ├── jump_generator.py    # Regime-dependent jump matrices Lambda_J
│   ├── combined_generator.py# Full 2D combined generator (for reference)
│   ├── option_pricing.py    # European pricing (fast + regular algorithms)
│   ├── american_pricing.py  # American pricing with early exercise boundary
│   ├── heston_analytical.py # Heston semi-closed-form (validation)
│   ├── svj_analytical.py    # SVJ semi-closed-form (validation)
│   ├── calibration.py       # Two-stage parameter calibration
│   └── data_loader.py       # Market data loading + mock data
├── utility/
│   ├── file_utils.py        # I/O utilities
│   └── visualization.py     # Plotting utilities
├── test/
│   └── test_2d_ctmc.py      # Unit test suite (31 tests)
├── doc/                     # Documentation
├── input/                   # Market data (Wind terminal exports)
└── result/                  # Pricing results, comparison tables
```

---

## 13. References

1. **Bates, D.S.** (1996). Jumps and Stochastic Volatility: Exchange Rate Processes Implicit in Deutsche Mark Options. *Review of Financial Studies*, 9(1), 69-107.

2. **Cox, J.C., Ingersoll, J.E., and Ross, S.A.** (1985). A Theory of the Term Structure of Interest Rates. *Econometrica*, 53(2), 385-407.

3. **Heston, S.L.** (1993). A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options. *Review of Financial Studies*, 6(2), 327-343.

4. **Longstaff, F.A. and Schwartz, E.S.** (2001). Valuing American Options by Simulation: A Simple Least-Squares Approach. *Review of Financial Studies*, 14(1), 113-147.

5. **Mackay, A., Vachon, M., and Cui, Z.** (2023). Continuous-Time Markov Chain Approximation of Multi-Dimensional Stochastic Processes. Working Paper.

6. **Merton, R.C.** (1976). Option Pricing When Underlying Stock Returns Are Discontinuous. *Journal of Financial Economics*, 3(1-2), 125-144.

7. **Tavella, D. and Randall, C.** (2000). *Pricing Financial Instruments: The Finite Difference Method*. John Wiley & Sons.
