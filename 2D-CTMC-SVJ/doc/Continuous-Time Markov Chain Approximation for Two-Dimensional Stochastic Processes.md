<div align=center>
    <font size=6>Continuous-Time Markov Chain Approximation for Two-Dimensional Stochastic Processes</font>
</div>


[TOC]

# Member List

Yi Hong (FAM/SMP)

[Author]



# Part I — Preliminary Knowledge

## 1. Continuous-Time Markov Chain

### 1.1 Definition

A stochastic process $\tilde{X} = \{\tilde{X}_t\}_{t \geq 0}$ taking values on a countable state space $\mathcal{S}$ is a **continuous-time Markov chain** (CTMC) if, for all states $\tilde{x}_j, \tilde{x}_{i_1}, \ldots, \tilde{x}_{i_{n-1}} \in \mathcal{S}$ and any time sequence $t_1 < t_2 < \cdots < t_n$, the following holds:

$$
\mathbb{P}\!\left(\tilde{X}_{t_n} = \tilde{x}_j \;\middle|\; \tilde{X}_{t_1} = \tilde{x}_{i_1}, \ldots, \tilde{X}_{t_{n-1}} = \tilde{x}_{i_{n-1}}\right) = \mathbb{P}\!\left(\tilde{X}_{t_n} = \tilde{x}_j \;\middle|\; \tilde{X}_{t_{n-1}} = \tilde{x}_{i_{n-1}}\right)
$$

Intuitively, the future evolution of the chain depends on the past only through the current state: this is the **Markov property** in continuous time.

The chain is said to be **time-homogeneous** if the transition probability $p_{ij}(s, t) := \mathbb{P}(\tilde{X}_t = \tilde{x}_j \mid \tilde{X}_s = \tilde{x}_i)$ depends only on the elapsed time $t - s$, so that we may write $p_{ij}(t - s)$ without ambiguity.

### 1.2 Transition Rates and the Generator Matrix

For a time-homogeneous CTMC on a **finite** state space $\mathcal{S} = \{\tilde{x}_1, \ldots, \tilde{x}_m\}$, there exist constants $\{q_{ij}\}_{1 \leq i, j \leq m}$, called the **transition rates** (or intensities), that characterize the infinitesimal behavior of the chain. Specifically, for a small time increment $h > 0$:

$$
p_{ij}(h) =
\begin{cases}
q_{ij} \, h + o(h) & i \neq j \\
1 + q_{ii} \, h + o(h) & i = j
\end{cases}
\tag{1.1}
$$

The rate $q_{ij}$ ($i \neq j$) can be interpreted as the instantaneous intensity of jumping from state $\tilde{x}_i$ to state $\tilde{x}_j$, while $-q_{ii}$ is the total rate of leaving state $\tilde{x}_i$.

The transition rates are subject to three constraints:

1. **Non-negativity:** $q_{ij} \geq 0$ for all $i \neq j$.
2. **Non-positivity of diagonal:** $q_{ii} \leq 0$ for all $i$.
3. **Row-sum condition:** $\displaystyle\sum_{j=1}^{m} q_{ij} = 0$ for every $i$.

The matrix $\mathbf{Q} := [q_{ij}]_{m \times m}$ satisfying these three properties is called the **generator** (or **Q-matrix**) of the CTMC.

**Example.** A two-state CTMC on $\{0, 1\}$ with transition rates $q_{01} = \alpha$ and $q_{10} = \beta$ has generator
$$
\mathbf{Q} = \begin{pmatrix} -\alpha & \alpha \\ \beta & -\beta \end{pmatrix}
$$

### 1.3 Holding Times and Jump Chain

The generator encodes two aspects of the chain's dynamics:

- **Holding time.** In state $\tilde{x}_i$, the chain remains for an exponentially distributed time with parameter $\lambda_i := -q_{ii} = \sum_{j \neq i} q_{ij}$. This gives $\mathbb{E}[\text{holding time in } \tilde{x}_i] = 1/\lambda_i$.

- **Jump chain.** Upon leaving $\tilde{x}_i$, the chain transitions to state $\tilde{x}_j$ with probability $q_{ij} / \lambda_i$ (for $j \neq i$). The sequence of visited states forms a discrete-time Markov chain called the **embedded** (or **jump**) **chain**.

These two pieces together completely determine the law of the CTMC.

### 1.4 The Transition Semigroup and Matrix Exponential

Under standard regularity conditions (the semigroup is standard and uniform), the transition probability matrix $\mathbf{P}_t := [p_{ij}(t)]$ is given by the **matrix exponential**:

$$
\mathbf{P}_t = e^{\mathbf{Q} t} := \sum_{k=0}^{\infty} \frac{(\mathbf{Q} \, t)^k}{k!}
\tag{1.2}
$$

This is the continuous-time analogue of matrix powers for discrete-time chains. The family $\{\mathbf{P}_t\}_{t \geq 0}$ satisfies the **semigroup property**:

$$
\mathbf{P}_{s+t} = \mathbf{P}_s \, \mathbf{P}_t \quad \forall \, s, t \geq 0
$$

and the **Kolmogorov forward and backward equations**:

$$
\mathbf{P}_t' = \mathbf{P}_t \, \mathbf{Q} \quad (\text{forward}) \qquad \mathbf{P}_t' = \mathbf{Q} \, \mathbf{P}_t \quad (\text{backward})
$$

### 1.5 Expectations via Matrix–Vector Products

The matrix exponential representation is the **computational cornerstone** of the CTMC approximation methodology. If $f : \mathcal{S} \to \mathbb{R}$ is any function on the state space and we define the column vector $\mathbf{f} = [f(\tilde{x}_1), \ldots, f(\tilde{x}_m)]^\top$, then:

$$
\mathbb{E}\!\left[f(\tilde{X}_t) \;\middle|\; \tilde{X}_0 = \tilde{x}_i\right] = \mathbf{e}_i \, e^{\mathbf{Q} t} \, \mathbf{f}
\tag{1.3}
$$

where $\mathbf{e}_i$ is the $i$-th standard basis row vector ($1 \times m$). More generally, the discounted conditional expectation $\mathbb{E}[e^{-rt} f(\tilde{X}_t) \mid \tilde{X}_0 = \tilde{x}_i]$ equals $\mathbf{e}_i \, e^{(\mathbf{Q} - r\mathbf{I}) t} \, \mathbf{f}$.

This reduces many pricing and expectation problems to **matrix exponentials** and **matrix–vector products**, both of which are well-studied numerical operations.

### 1.6 Approximation of Diffusions by CTMCs

The key idea underlying this entire document is that a **continuous diffusion process** can be approximated by a **finite-state CTMC** whose generator is constructed to match the local drift and volatility of the diffusion. The theoretical justification rests on two classical results:

- **Kushner (1990):** If the first two conditional moments of the CTMC transition match those of the diffusion up to $o(h)$, the CTMC converges weakly to the diffusion as the grid spacing tends to zero. These moment-matching conditions are called the **local consistency conditions**.

- **Ethier & Kurtz (2005):** Semigroup convergence — if the generators of Markov processes converge appropriately to the generator of a limiting process, then the processes themselves converge weakly.



## 2. Stochastic Volatility Processes

### 2.1 General Two-Dimensional Framework

Consider a filtered probability space $(\Omega, \mathcal{F}, \mathbb{F}, \mathbb{Q})$ supporting a two-dimensional correlated Brownian motion $W = \{(W_t^{(1)}, W_t^{(2)})\}_{t \geq 0}$ with $\text{d}[W^{(1)}_t, W^{(2)}_t] = \rho \, \text{d}t$, $\rho \in [-1, 1]$. A **stochastic volatility (SV)** model specifies the risk-neutral dynamics of an asset price $S$ and a volatility-driving factor $V$ as:

$$
\begin{aligned}
\text{d}S_t &= r \, S_t \, \text{d}t + \sigma_S(V_t) \, S_t \, \text{d}W_t^{(1)} \\[4pt]
\text{d}V_t &= \mu_V(V_t) \, \text{d}t + \sigma_V(V_t) \, \text{d}W_t^{(2)}
\end{aligned}
\tag{2.1}
$$

with initial conditions $S_0 = s_0 > 0$ and $V_0 = v_0 \in \mathcal{S}_V$, where $\mathcal{S}_V$ denotes the state space of $V$ (typically $(0, \infty)$ or $\mathbb{R}$). The process $V$ is referred to as the **variance process** throughout, although in some models it may represent the inverse variance or the log-volatility.

The key objects are the three coefficient functions:

| Function          | Role                                   | Domain                           |
| :---------------- | :------------------------------------- | :------------------------------- |
| $\mu_V(\cdot)$    | Drift of the variance process          | $\mathcal{S}_V \to \mathbb{R}$   |
| $\sigma_V(\cdot)$ | Volatility of the variance process     | $\mathcal{S}_V \to \mathbb{R}_+$ |
| $\sigma_S(\cdot)$ | Volatility function linking $V$ to $S$ | $\mathcal{S}_V \to \mathbb{R}_+$ |

### 2.2 Regularity Assumptions

The following **standing assumptions** are imposed throughout:

**(A1)** $\mu_V : \mathcal{S}_V \to \mathbb{R}$ is continuous.

**(A2)** $\sigma_S, \sigma_V : \mathcal{S}_V \to \mathbb{R}_+$ are continuously differentiable, with $\sigma_S(\cdot) > 0$ and $\sigma_V(\cdot) > 0$ on $\mathcal{S}_V$.

**(A3)** The SDE system (2.1) admits a **unique-in-law weak solution**.

**(A4)** (Martingale condition) The discounted price $\{e^{-rt} S_t\}$ is a true $\mathbb{Q}$-martingale, which may impose additional parameter restrictions depending on the specific model.

These conditions above guarantee: existence and uniqueness of the process $(S, V)$; absence of arbitrage under the risk neutral measure $\mathbb{Q}$; and applicability of the weak convergence theory.

### 2.3 The Log-Price Process

Applying Ito formula to $\ln S_t$ yields the **log-price dynamics**:

$$
\text{d}\ln S_t = \left(r - \tfrac{1}{2}\sigma_S^2(V_t)\right) \text{d}t + \sigma_S(V_t) \, \text{d}W_t^{(1)}
\tag{2.2}
$$

Since $\sigma_S(V_t)$ is stochastic, the log-price is **not** Gaussian in general, and analytical solutions are usually unavailable.

### 2.4 Specific Stochastic Volatility Models

The following table describes SV models covered by the two-layer CTMC framework. In all cases, the asset dynamics follow (2.1).

| Model | Dynamics | Parameters | Conditions for Martingale Measure |
|:------|:---------|:-----------|:----------------------------|
| **Heston (1993)** | $dS_t = rS_t dt + \sqrt{V_t}S_t dW^{(1)}$  $dV_t = \kappa(\theta - V_t)dt + \sigma\sqrt{V_t} dW^{(2)}$ | $S_0 > 0$,  $\kappa, \theta, \sigma, V_0 > 0$ | No additional conditions |
| **3/2 (1997)** | $dS_t = rS_t dt + S_t/\sqrt{V_t} dW^{(1)}$  $dV_t = \kappa(\theta - V_t)dt - \sigma\sqrt{V_t} dW^{(2)}$ | $S_0 > 0$,  $\kappa, \theta, \sigma, V_0 > 0$  with $\kappa\theta \geq \sigma^2/2$ | $\rho \leq 0$ |
| **4/2 (2017)** | $dS_t = rS_t dt + S_t[a\sqrt{V_t} + b/\sqrt{V_t}] dW^{(1)}$  $dV_t = \kappa(\theta - V_t)dt + \sigma\sqrt{V_t} dW^{(2)}$ | $a, b \in \mathbb{R}$, $S_0 > 0$,  $\kappa, \theta, \sigma, V_0 > 0$,  with $\kappa\theta \geq \sigma^2/2$ | $\sigma^2 \leq 2\kappa\theta + \min(0, 2\rho\sigma b)$ |
| **Hull–White (1987)** | $dS_t = rS_t dt + \sqrt{V_t}S_t dW^{(1)}$  $dV_t = \alpha V_t dt + \beta V_t dW^{(2)}$ | $S_0, \alpha, \beta, V_0 > 0$ | $\rho \leq 0$ |
| **Scott (1987)** | $dS_t = rS_t dt + e^{V_t}S_t dW^{(1)}$  $dV_t = \kappa(\theta - V_t)dt + \sigma dW^{(2)}$ | $S_0 > 0$,  $\kappa, \theta, V_0 \in \mathbb{R}$, $\sigma > 0$ | $\rho \leq 0$ |
| **α-Hypergeometric (2016)** | $dS_t = rS_t + e^{V_t}S_t dW^{(1)}$  $dV_t = (a - be^{\alpha V_t})dt + \sigma dW^{(2)}$ | $S_0, \alpha, b, \sigma > 0$, $a, V_0 \in \mathbb{R}$ | If $\alpha \geq 2$ or $\alpha < 2$ and either  $\rho \leq 0, \alpha > 1$ or $\alpha = 1$ and $b \geq \rho\sigma$ |

### 2.5 Correlation Parameter $\rho$

The parameter $\rho = \text{Corr}(\text{d}W_t^{(1)}, \text{d}W_t^{(2)})$ governs the **leverage effect**: the empirical observation that asset returns and volatility changes are negatively correlated (i.e., $\rho < 0$). In equity markets, typical values range from $-0.9$  to $-0.5$.

From a technical standpoint, $\rho \neq 0$ introduces a **coupling** between $S$ and $V$ that prevents us from treating the two processes independently. Decoupling this correlation is a central step in the CTMC methodology.



## 3. Stochastic Volatility with Jumps (SVJ)

### 3.1 Motivation

While SV models capture the stochastic nature of volatility, they are not able to capture the **short-maturity implied volatility smile** observed in option markets. Adding **jumps** in the asset price addresses this limitation: sudden large moves in $S$ generate the excess kurtosis needed to fit the smile at short maturities.

### 3.2 General SVJ Dynamics

An SVJ model augments the SV dynamics (2.1) with a jump component in the asset price:

$$
\begin{aligned}
\frac{\text{d}S_t}{S_{t^-}} &= (r - \lambda\bar{k}) \, \text{d}t + \sigma_S(V_t) \, \text{d}W_t^{(1)} + \text{d}J_t \\
\text{d}V_t &= \mu_V(V_t) \, \text{d}t + \sigma_V(V_t) \, \text{d}W_t^{(2)}
\end{aligned}
\tag{3.1}
$$

where:

- $J_t = \sum_{i=1}^{N_t}(e^{Z_i} - 1)$ is a **compound Poisson process**
- $N = \{N_t\}_{t \geq 0}$ is a Poisson process with intensity $\lambda > 0$
- $Z_1, Z_2, \ldots$ are i.i.d. random variables with distribution $f_Z$ (the **log-jump size distribution**)
- $\bar{k} := \mathbb{E}[e^Z - 1]$ is the **jump compensator**, ensuring $\{e^{-rt} S_t\}$ remains a martingale

The Poisson process $N$ and the jump sizes $\{Z_i\}$ are assumed independent of each other and of the Brownian motions $(W^{(1)}, W^{(2)})$.

### 3.3 Log-Price Dynamics under SVJ

Applying Itô's formula for jump-diffusions to $\ln S_t$:

$$
\text{d}\ln S_t = \left(r - \lambda\bar{k} - \tfrac{1}{2}\sigma_S^2(V_t)\right)\text{d}t + \sigma_S(V_t)\,\text{d}W_t^{(1)} + \text{d}\!\left(\sum_{i=1}^{N_t} Z_i\right)
\tag{3.2}
$$

The continuous part is identical to the SV log-price dynamics (2.2) up to the drift correction $\lambda\bar{k}$, while the jump part adds random displacements of size $Z_i$ at Poisson arrival times.

### 3.4 Common Jump Specifications

| Model             | Jump Distribution $f_Z$                 | Parameters                   | $\bar{k}$                                                    |
| :---------------- | :-------------------------------------- | :--------------------------- | :----------------------------------------------------------- |
| **Merton** (1976) | $Z \sim \mathcal{N}(\mu_J, \sigma_J^2)$ | $\lambda, \mu_J, \sigma_J$   | $e^{\mu_J + \sigma_J^2/2} - 1$                               |
| **Kou** (2002)    | Double exponential                      | $\lambda, p, \eta_1, \eta_2$ | $\frac{p\eta_1}{\eta_1 - 1} + \frac{(1-p)\eta_2}{\eta_2 + 1} - 1$ |

The **Bates (1996) model** combines Heston SV dynamics with Merton-type log-normal jumps.

### 3.5 Role in the CTMC Framework

From the perspective of the CTMC approximation, jump components affect **only the second layer** (the auxiliary/log-price process). The first-layer construction for the variance process $V$ is entirely unchanged, because the jumps enter only through $S$ (or equivalently, $\ln S$).



# Part II — Methodology

## 4. Overview of the Two-Layer CTMC Approximation

### 4.1 Basic Setup

The goal is to approximate the continuous two-dimensional process $(S, V)$ governed by (2.1) — or more generally (3.1) in the SVJ case — by a **finite-state CTMC** that is amenable to efficient matrix computation. The method proceeds in two stages:

$$
\underbrace{V_t}_{\text{continuous}} \;\xrightarrow{\;\text{Layer 1}\;}\; \underbrace{V_t^{(m)}}_{\text{CTMC on } m \text{ states}}
\qquad\qquad
\underbrace{X_t \mid V^{(m)}}_{\text{regime-switching diffusion}} \;\xrightarrow{\;\text{Layer 2}\;}\; \underbrace{X_t^{(m,N)}}_{\text{CTMC on } N \text{ states per regime}}
$$

where $X_t := \ln S_t - \rho\,\gamma(V_t)$ is an **auxiliary process** obtained by a decorrelation transformation.

### 4.2 Why Two Layers?

A direct discretization of the two-dimensional process $(S, V)$ on a product grid would require matching the **cross-variation** $\text{d}[S_t, V_t] \neq 0$ (since $\rho \neq 0$), which complicates the generator construction. The two-layer approach circumvents this by:

1. First approximating $V$ alone — a **one-dimensional** CTMC approximation problem.
2. Then, **conditional on each discrete $V^{(m)}$**, approximating the auxiliary process $X$ — again a **one-dimensional** problem in each regime.
3. Finally, **combining** the two layers into a single enlarged CTMC.

The decorrelation transformation ensures that, conditioned on $V^{(m)}$, the driving noise of $X$ is **independent** of the driving noise of $V$, making the framework suitable for both single contracts and portfolios of derivatives.

### 4.3 Notation Summary

| Symbol                                       | Meaning                                                      |
| :------------------------------------------- | :----------------------------------------------------------- |
| $m$                                          | Number of states in the variance CTMC                        |
| $N$                                          | Number of states in the auxiliary CTMC (per regime)          |
| $\mathcal{S}_V^{(m)} = \{v_1, \ldots, v_m\}$ | State space of the first-layer CTMC                          |
| $\mathcal{S}_X^{(N)} = \{x_1, \ldots, x_N\}$ | State space of the second-layer CTMC                         |
| $\mathbf{Q}^{(m)}$                           | Generator of $V^{(m)}$ ($m \times m$, tri-diagonal)          |
| $\mathbf{G}_l^{(N)}$                         | Generator of $X^{(m,N)}$ in regime $v_l$ ($N \times N$, tri-diagonal) |
| $\mathbf{G}^{(m,N)}$                         | Combined generator on enlarged space ($mN \times mN$)        |
| $\gamma(\cdot)$                              | Decorrelation function: $\gamma(x) = \int^x \sigma_S(u)/\sigma_V(u)\,\text{d}u$ |



## 5. First Layer: Approximation of the Variance Process

### 5.1 Objective

Construct a CTMC $\{V_t^{(m)}\}_{t \geq 0}$ on a **finite** state space

$$
\mathcal{S}_V^{(m)} := \{v_1, v_2, \ldots, v_m\} \quad v_1 < v_2 < \cdots < v_m \quad m \in \mathbb{N}
$$

that converges weakly to the original diffusion $\{V_t\}$ as $m \to \infty$.

### 5.2 Grid Design

#### Boundary Selection

The boundaries $v_1$ and $v_m$ must be sufficiently extreme to cover the relevant portion of $\mathcal{S}_V$. Common choices include:

- **Simple scaling:** $v_1 = \alpha V_0$, $v_m = \beta V_0$ for a small (resp. large) constant $\alpha$ (resp. $\beta$)
- **Moment-based:** use the mean $\pm$ several standard deviations of the stationary distribution of $V$.

#### Tavella–Randall Non-Uniform Grid

The grid concentrates more points near the initial value $V_0$, improving accuracy for a given $m$:

$$
v_i = V_0 + \tilde{\alpha} \, \sinh\!\left(c_2 \, \frac{i}{m} + c_1 \left(1 - \frac{i}{m}\right)\right), \quad i = 2 \ldots, m-1,
\tag{5.1}
$$

where

$$
c_1 = \sinh^{-1}\!\left(\frac{v_1 - V_0}{\tilde{\alpha}}\right), \qquad c_2 = \sinh^{-1}\!\left(\frac{v_m - V_0}{\tilde{\alpha}}\right)
$$

and $\tilde{\alpha} > 0$ controls the degree of non-uniformity. Smaller $\tilde{\alpha}$ produces a more concentrated grid around $V_0$. When $\tilde{\alpha} \to \infty$, the non-uniform grid will be converted into a uniform grid.

### 5.3 Generator Construction via Local Consistency

The generator $\mathbf{Q}^{(m)} = [q_{ij}]_{m \times m}$ is determined by the **local consistency conditions**. The first two conditional moments of the CTMC transitions must match those of the diffusion, which can be expressed as follows:

$$
\begin{aligned}
\mathbb{E}\!\left[V_{t+h}^{(m)} - V_t^{(m)} \;\middle|\; V_t^{(m)} = v_i\right] &= \mu_V(v_i) \, h + o(h) \\[4pt]
\mathbb{E}\!\left[\left(V_{t+h}^{(m)} - V_t^{(m)}\right)^2 \;\middle|\; V_t^{(m)} = v_i\right] &= \sigma_V^2(v_i) \, h + o(h)
\end{aligned}
\tag{5.2}
$$

Define the grid spacings $\delta_i := v_{i+1} - v_i$ for $i = 1, \ldots, m-1$.

**Interior rates** ($2 \leq i \leq m - 1$):
$$
q_{i,\,i-1} = \frac{\sigma_V^2(v_i) - \delta_i \, \mu_V(v_i)}{\delta_{i-1}\,(\delta_{i-1} + \delta_i)}\qquad
q_{i,\,i+1} = \frac{\sigma_V^2(v_i) + \delta_{i-1} \, \mu_V(v_i)}{\delta_i\,(\delta_{i-1} + \delta_i)}

\tag{5.3}
$$

$$
q_{ii} = -(q_{i,\,i-1} + q_{i,\,i+1}) \qquad q_{ij} = 0 \;\text{ for }\; |i - j| > 1
$$

**Boundary rates:**

$$
q_{12} = \frac{|\mu_V(v_1)|}{\delta_1}\quad q_{11} = -q_{12} \qquad q_{m,\,m-1} = \frac{|\mu_V(v_m)|}{\delta_{m-1}}\quad q_{mm} = -q_{m,\,m-1}
\tag{5.4}
$$

All other boundary rates are zero.

**Remark (Tri-diagonal structure).** By construction, $q_{ij} = 0$ for $|i - j| > 1$. The CTMC can only jump to **nearest-neighbor** states, reflecting the continuous sample paths of the original diffusion $V$. This tri-diagonal sparsity is key to computational efficiency.

### 5.4 Derivation of the Interior Rates

To see why (5.3) holds, consider state $v_i$ with $2 \leq i \leq m-1$. The CTMC can jump to $v_{i-1}$ or $v_{i+1}$ with rates $q_{i,i-1}$ and $q_{i,i+1}$. Over an infinitesimal interval $h$:

$$
\mathbb{E}[V_{t+h}^{(m)} - v_i \mid V_t^{(m)} = v_i] = q_{i,i+1}\,\delta_i \, h - q_{i,i-1}\,\delta_{i-1}\,h + o(h)
$$

$$
\mathbb{E}[(V_{t+h}^{(m)} - v_i)^2 \mid V_t^{(m)} = v_i] = q_{i,i+1}\,\delta_i^2\,h + q_{i,i-1}\,\delta_{i-1}^2\,h + o(h)
$$

Setting these equal to $\mu_V(v_i)\,h$ and $\sigma_V^2(v_i)\,h$ respectively gives a $2 \times 2$ linear system in $(q_{i,i-1}, q_{i,i+1})$, whose solution is exactly (5.3).

### 5.5 Well-Definedness Conditions

For the generator to be valid ($q_{ij} \geq 0$ for $i \neq j$), the grid spacing must satisfy:

- If $\mu_V(v_i) < 0$: $\;\delta_{i-1} \leq \sigma_V^2(v_i) / |\mu_V(v_i)|$.
- If $\mu_V(v_i) > 0$: $\;\delta_i \leq \sigma_V^2(v_i) / \mu_V(v_i)$.

A sufficient (but sometimes overly restrictive) condition is:

$$
\max_{1 \leq i \leq m-1} \delta_i \leq \min_{2 \leq i \leq m-1} \frac{\sigma_V^2(v_i)}{|\mu_V(v_i)|}
\tag{5.5}
$$

When this condition is not satisfied, more points should be added to the existing grid. However, it can sometimes be too restrictive, particularly in the case of approximating a two-dimensional process, where adding more points to the state-space becomes too expensive computationally. In that case, condition (5.5) may be replaced by the two simple requirements above condition (5.5), which is less restrictive. However, from a numerical perspective, we observe that such conditions are not necessary to obtain good approximation results.



## 6. Second Layer: Approximation of the Auxiliary Process

### 6.1 Decorrelation via Change of Variable

The two Brownian motions $W^{(1)}_t$ and $W^{(2)}_t$ are correlated when $\rho \neq 0$. To construct the second layer which is independent of the first one, we must first **remove this correlation**. This is achieved through the following change of variable.

**Lemma 6.1** Define the function
$$
\gamma(x) := \int^x \frac{\sigma_S(u)}{\sigma_V(u)} \, \text{d}u
\tag{6.1}
$$

and the auxiliary process

$$
X_t := \ln S_t - \rho \, \gamma(V_t)
\tag{6.2}
$$

Then $(X, V)$ satisfies:

$$
\begin{aligned}
\text{d}X_t &= \mu_X(X_t, V_t) \, \text{d}t + \sigma_X(V_t) \, \text{d}W_t^* \\[4pt]
\text{d}V_t &= \mu_V(V_t) \, \text{d}t + \sigma_V(V_t) \, \text{d}W_t^{(2)}
\end{aligned}
\tag{6.3}
$$

where

$$
W_t^* := \frac{W_t^{(1)} - \rho \, W_t^{(2)}}{\sqrt{1 - \rho^2}}
\tag{6.4}
$$

is a standard Brownian motion **independent** of $W^{(2)}$, and:

$$
\sigma_X(y) := \sqrt{1 - \rho^2} \; \sigma_S(y)
\tag{6.5}
$$

$$
\mu_X(x, y) := r - \frac{\sigma_S^2(y)}{2} - \rho \, \psi(y)
\tag{6.6}
$$

with

$$
\psi(y) := \mu_V(y) \, \frac{\sigma_S(y)}{\sigma_V(y)} + \frac{1}{2}\!\left[\sigma_V(y) \, \sigma_S'(y) - \sigma_V'(y) \, \sigma_S(y)\right]
\tag{6.7}
$$

**Proof.** We apply Ito formula to $X_t = \ln S_t - \rho \gamma(V_t)$.

**Step 1: Compute $d\ln S_t$**

From equation (2.1), we have:
$$
dS_t = r S_t dt + \sigma_S(V_t) S_t dW_t^{(1)}
$$

Applying Ito formula to $\ln S_t$:
$$
d\ln S_t = \frac{1}{S_t} dS_t - \frac{1}{2S_t^2} d[S_t,S_t]
$$
where the quadratic variation is:
$$
d[S_t,S_t] = \sigma_S^2(V_t) S_t^2 dt
$$

Therefore:
$$
d\ln S_t = \left(r - \frac{\sigma_S^2(V_t)}{2}\right) dt + \sigma_S(V_t) dW_t^{(1)} \tag{6.8}
$$

**Step 2: Compute $d\gamma(V_t)$**

By definition, $\gamma(x) = \int^x \frac{\sigma_S(u)}{\sigma_V(u)} du$, so:
$$
\gamma'(x) = \frac{\sigma_S(x)}{\sigma_V(x)} \qquad \gamma''(x) = \frac{\sigma_V(x)\sigma_S'(x) - \sigma_V'(x)\sigma_S(x)}{\sigma_V^2(x)}
$$

Applying Ito formula to $\gamma(V_t)$:
$$
d\gamma(V_t) = \gamma'(V_t) dV_t + \frac{1}{2} \gamma''(V_t) d[V_t, V_t]
$$

From equation (2.1), we have $dV_t = \mu_V(V_t) dt + \sigma_V(V_t) dW_t^{(2)}$ and $d[V_t,V_t] = \sigma_V^2(V_t) dt$. Substituting:
$$
\begin{aligned}
d\gamma(V_t) &= \frac{\sigma_S(V_t)}{\sigma_V(V_t)} \left[\mu_V(V_t) dt + \sigma_V(V_t) dW_t^{(2)}\right]  + \frac{1}{2} \cdot \frac{\sigma_V(V_t)\sigma_S'(V_t) - \sigma_V'(V_t)\sigma_S(V_t)}{\sigma_V^2(V_t)} \cdot \sigma_V^2(V_t) dt \\
&= \left[\mu_V(V_t) \frac{\sigma_S(V_t)}{\sigma_V(V_t)} + \frac{1}{2}\left(\sigma_V(V_t)\sigma_S'(V_t) - \sigma_V'(V_t)\sigma_S(V_t)\right)\right] dt + \sigma_S(V_t) dW_t^{(2)}
\end{aligned}
$$

Define:
$$
\psi(V_t) := \mu_V(V_t) \frac{\sigma_S(V_t)}{\sigma_V(V_t)} + \frac{1}{2}\left[\sigma_V(V_t)\sigma_S'(V_t) - \sigma_V'(V_t)\sigma_S(V_t)\right]
$$

Then:
$$
d\gamma(V_t) = \psi(V_t) dt + \sigma_S(V_t) dW_t^{(2)} \tag{6.9}
$$

**Step 3: Compute $dX_t$**

From $X_t = \ln S_t - \rho \gamma(V_t)$, combining (6.8) and (6.9):
$$
\begin{aligned}
dX_t &= d\ln S_t - \rho d\gamma(V_t) \\
&= \left(r - \frac{\sigma_S^2(V_t)}{2}\right) dt + \sigma_S(V_t) dW_t^{(1)} - \rho \left[\psi(V_t) dt + \sigma_S(V_t) dW_t^{(2)}\right] \\
&= \left[r - \frac{\sigma_S^2(V_t)}{2} - \rho \psi(V_t)\right] dt + \sigma_S(V_t) \left(dW_t^{(1)} - \rho dW_t^{(2)}\right)
\end{aligned}
$$

**Step 4: Define the new Brownian motion $W_t^*$**

Define:
$$
W_t^* := \frac{W_t^{(1)} - \rho W_t^{(2)}}{\sqrt{1-\rho^2}}
$$

We need to verify that $W_t^*$ is a standard Brownian motion:
- $W_0^* = 0$ (since $W_0^{(1)} = W_0^{(2)} = 0$).
- Continuous paths: Follows from the continuity of $W^{(1)}$ and $W^{(2)}$.
- Independent increments: Follows from the independent increments of $W^{(1)}$ and $W^{(2)}$.
- Quadratic variation:
  $$
  \begin{aligned}
  d[W^*_t,W^*_t] &= \frac{1}{1-\rho^2} d[(W^{(1)}_t - \rho W^{(2)}_t),(W^{(1)}_t - \rho W^{(2)}_t)] \\
  &= \frac{1}{1-\rho^2} \left(dt - 2\rho \cdot \rho dt + \rho^2 dt\right) \\
  &= \frac{1-\rho^2}{1-\rho^2} dt = dt
  \end{aligned}
  $$

We verify that $W_t^*$ is independent of $W_t^{(2)}$:
$$
\begin{aligned}
d[W^*_t, W^{(2)}_t] &= \frac{1}{\sqrt{1-\rho^2}} d[W^{(1)} - \rho W^{(2)}, W^{(2)}]_t \\
&= \frac{1}{\sqrt{1-\rho^2}} \left(\rho dt - \rho dt\right) = 0
\end{aligned}
$$

Since $W^*_t$ and $W^{(2)}_t$ are jointly Gaussian with zero quadratic covariation, they are independent.

**Step 5: Final form**

From $dW_t^{(1)} - \rho dW_t^{(2)} = \sqrt{1-\rho^2} dW_t^*$, substituting into $dX_t$:
$$
\begin{aligned}
dX_t &= \left[r - \frac{\sigma_S^2(V_t)}{2} - \rho \psi(V_t)\right] dt + \sigma_S(V_t) \sqrt{1-\rho^2} dW_t^* \\
&= \mu_X(X_t, V_t) dt + \sigma_X(V_t) dW_t^*
\end{aligned}
$$
where:
$$
\begin{aligned}
\sigma_X(y) &:= \sqrt{1-\rho^2} \, \sigma_S(y) \\
\mu_X(x, y) &:= r - \frac{\sigma_S^2(y)}{2} - \rho \psi(y)
\end{aligned}
$$

The dynamics of $V_t$ remain unchanged:
$$
dV_t = \mu_V(V_t) dt + \sigma_V(V_t) dW_t^{(2)}
$$

This completes the proof.

**Key insight.** The driving noises $W^*$ and $W^{(2)}$ are **independent** by construction. This independence is what enables the two-layer approach: the first layer (approximating $V$) and the second layer (approximating $X$ conditional on $V$) can be constructed as independent one-dimensional CTMC problems.

**Recovery.** The original price can be recovered via the inverse transformation $S_t = \exp(X_t + \rho \, \gamma(V_t))$.

### 6.2 Regime-Switching Diffusion

After replacing $V$ by its first-layer CTMC $V^{(m)}$, the auxiliary process becomes a **regime-switching diffusion**:

$$
\text{d}X_t^{(m)} = \mu_X\!\left(X_t^{(m)}, V_t^{(m)}\right) \text{d}t + \sigma_X\!\left(V_t^{(m)}\right) \text{d}W_t^*
\tag{6.8}
$$

When $V_t^{(m)} = v_l$, the process $X^{(m)}$ evolves as a **one-dimensional diffusion** with drift $\mu_X(\cdot, v_l)$ and volatility $\sigma_X(v_l)$. The regimes switch according to the generator $\mathbf{Q}^{(m)}$.

### 6.3 CTMC Approximation in Each Regime

For each variance state $v_l$ ($l = 1, \ldots, m$), we construct a state space on the price dynamics: 

$$
\mathcal{S}_X^{(N)} = \{x_1, x_2, \ldots, x_N\}, \quad x_1 < x_2 < \cdots < x_N, \quad N \in \mathbb{N}
$$

with generator $\mathbf{G}_l^{(N)} = [\lambda_{ij}^{(l)}]_{N \times N}$ defined by the same local consistency approach, using coefficients $\sigma_X(v_l)$ and $\mu_X(x_i, v_l)$:

$$
\lambda_{i,\,i-1}^{(l)} = \frac{\sigma_X^2(v_l) - \delta_i^x \, \mu_X(x_i, v_l)}{\delta_{i-1}^x\,(\delta_{i-1}^x + \delta_i^x)}, \qquad
\lambda_{i,\,i+1}^{(l)} = \frac{\sigma_X^2(v_l) + \delta_{i-1}^x \, \mu_X(x_i, v_l)}{\delta_i^x\,(\delta_{i-1}^x + \delta_i^x)}
\tag{6.9}
$$

for $2 \leq i \leq N-1$, where $\delta_i^x = x_{i+1} - x_i$. 

**Boundary rates**
$$
\lambda_{12}^l = \frac{|\mu_X(x_1, v_l)|}{\delta_1^x}, \lambda_{11}^l = -\lambda_{12}^l\qquad \lambda_{N,N-1}^l = \frac{|\mu_X(x_N, v_l)|}{\delta_{N-1}^x}, \lambda_{N,N}^l = -\lambda_{N,N-1}^l
$$
**Remark.** The state space $\mathcal{S}_X^{(N)}$ is shared across all regimes — only the generators $\mathbf{G}_l^{(N)}$ differ from regime to regime. This is a design choice that simplifies the combined generator in the next section.



## 7. Combining the Two Layers: The Enlarged Generator

### 7.1 Product State Space

The bivariate CTMC $(X^{(m,N)}, V^{(m)})$ takes values in the product space $\mathcal{S}_X^{(N)} \times \mathcal{S}_V^{(m)}$, which has $m \cdot N$ elements. A bijective mapping flattens this to a one-dimensional index:

$$
\psi : (x_n, v_l) \mapsto (l - 1)N + n, \quad 1 \leq n \leq N, \; 1 \leq l \leq m.
\tag{7.1}
$$

Intuitively, this is like flattening a large matrix row - by - row (or block - by - block) into a long vector. When the variance is at state $v_1$, the corresponding states of $X$ occupy indices 1 to $N$. When the variance is at state $v_2$, the corresponding states of $X$ occupy indices $N+1$ to $2N$.

### 7.2 The Combined Generator

**Proposition 7.1** The generator of the one-dimensional CTMC $Y^{(m,N)} := \psi(X^{(m,N)}, V^{(m)})$ on $\{1, \ldots, m \cdot N\}$ is the $mN \times mN$ block matrix:
$$
\mathbf{G}^{(m,N)} =
\begin{pmatrix}
q_{11}\mathbf{I}_N + \mathbf{G}_1^{(N)} & q_{12}\mathbf{I}_N & \cdots & q_{1m}\mathbf{I}_N \\[4pt]
q_{21}\mathbf{I}_N & q_{22}\mathbf{I}_N + \mathbf{G}_2^{(N)} & \cdots & q_{2m}\mathbf{I}_N \\[2pt]
\vdots & \vdots & \ddots & \vdots \\[2pt]
q_{m1}\mathbf{I}_N & q_{m2}\mathbf{I}_N & \cdots & q_{mm}\mathbf{I}_N + \mathbf{G}_m^{(N)}
\end{pmatrix},
\tag{7.2}
$$

where $\mathbf{I}_N$ is the $N \times N$ identity, $\mathbf{G}_l^{(N)}$ are the regime-specific generators from (6.9), and $\mathbf{Q}^{(m)} = [q_{ij}]$ is the first-layer generator from (5.3)–(5.4).

**Interpretation of the block structure:**

- **Diagonal block** $(l, l)$: $q_{ll}\mathbf{I}_N + \mathbf{G}_l^{(N)}$. The term $\mathbf{G}_l^{(N)}$ governs the evolution of $X$ within regime $v_l$; the term $q_{ll}\mathbf{I}_N$ accounts for the "drain" of probability due to transitions out of regime $l$.

- **Off-diagonal block** $(l, j)$: $q_{lj}\mathbf{I}_N$. When $V^{(m)}$ jumps from $v_l$ to $v_j$, the auxiliary state $X$ remains unchanged (identity matrix). The rate of this transition is $q_{lj}$.

**Sparsity.** Because $\mathbf{Q}^{(m)}$ is tri-diagonal (only nearest-neighbor variance transitions) and each $\mathbf{G}_l^{(N)}$ is tri-diagonal, the combined generator $\mathbf{G}^{(m,N)}$ is a **sparse block tri-diagonal matrix**. This is critical for computational efficiency.

### 7.3 Equivalence

For any measurable functional $\Psi$ with finite expectation:

$$
\mathbb{E}\!\left[\Psi(X^{(m,N)}, V^{(m)}) \;\middle|\; X_0 = x_i, V_0 = v_k\right]
= \mathbb{E}\!\left[\Psi(\psi^{-1}(Y^{(m,N)})) \;\middle|\; Y_0 = (k-1)N + i\right]
\tag{7.3}
$$

All computations involving the two-dimensional CTMC reduce to **matrix operations** on the $mN$-dimensional state space. This mapping allows us to utilize standard matrix exponential algorithms designed for one-dimensional state spaces, without loss of generality.

To understand why this mapping is necessary, consider the following intuition:

1.  **The "Flattening" Analogy** 
    Imagine the product state space $\mathcal{S}_X^{(N)} \times \mathcal{S}_V^{(m)}$ as a **chessboard** with $m$ rows (variance states) and $N$ columns (price states).  
    - The **2D view** $(x_n, v_l)$ is natural for humans to understand (Price vs. Volatility).  
    - The **1D view** $Y^{(m,N)}$ is like numbering every square on the chessboard from $1$ to $mN$. 
    The mapping $\psi$ is simply the rule that tells us "Square at Row $l$, Column $n$ corresponds to Number $(l-1)N + n$".

2.  **Why Necessary?**
    Most numerical linear algebra libraries (like MATLAB's `expm` or Python's `scipy.linalg.expm`) are designed to work with **vectors and matrices**, not multi-dimensional arrays. 
    By flattening the state space into a 1D index, we can construct the single combined generator $\mathbf{G}^{(m,N)}$ and use standard matrix exponentials to compute transition probabilities. Without this step, we would need to write custom algorithms for 2D tensor operations, which is inefficient and complex.

3.  **No Loss of Information** 
    Equation (7.3) guarantees that this "flattening" is **lossless**. 
    Calculating an expectation on the 1D vector $Y^{(m,N)}$ gives the **exact same result** as calculating it on the 2D pair $(X^{(m,N)}, V^{(m)})$, provided we use the correct generator $\mathbf{G}^{(m,N)}$ from Proposition 7.1. 
    Essentially, we are just changing the **addressing system** of the states, not the underlying probability dynamics.



## 8. Weak Convergence

### 8.1 Main Theorem

**Theorem 8.1** Under assumptions (A1)–(A4), as $m, N \to \infty$:
$$
(X^{(m,N)}, V^{(m)}) \Rightarrow (X, V)
\tag{8.1}
$$

in the Skorokhod $J_1$-topology on $D([0,T]; \mathbb{R}^2)$. Consequently, by the continuous mapping theorem:
$$
S^{(m,N)} := \exp\!\left(X^{(m,N)} + \rho \, \gamma(V^{(m)})\right) \Rightarrow S.
$$

### 8.2 Proof Outline

The proof proceeds in three steps:

**Step 1 (First layer).** The CTMC $V^{(m)}$ converges weakly to the diffusion $V$ as $m \to \infty$. This follows from:

- The local consistency conditions (5.2) ensure that the generator of $V^{(m)}$ converges pointwise to the infinitesimal generator of $V$ on a core of its domain.
- By the **Ethier–Kurtz theorem**, semigroup convergence implies weak convergence.

**Step 2 (Second layer).** For each fixed regime trajectory $V^{(m)}$, the CTMC $X^{(m,N)}$ converges weakly to the regime-switching diffusion $X^{(m)}$ as $N \to \infty$, by the same argument applied to each generator $\mathbf{G}_l^{(N)}$.

**Step 3 (Combined).** Passing $m \to \infty$ in the second step, the regime-switching diffusion $X^{(m)}$ converges weakly to $X$ (the true auxiliary process). The joint convergence of $(X^{(m,N)}, V^{(m)})$ to $(X, V)$ follows from the convergence of the combined generator $\mathbf{G}^{(m,N)}$ to the infinitesimal generator of $(X, V)$.

The **continuous mapping theorem** then extends weak convergence to the original price process $S = \exp(X + \rho\gamma(V))$.



# Part III — Applications: VIX Approximation and Variable Annuity Pricing

Part III aims to demonstrate the practical computational power of this framework. We selected two representative financial problems:

- **VIX Index Approximation (Section 10)**: This is a **one-dimensional problem** that depends only on the first-layer CTMC. It serves to validate the effectiveness of the variance process approximation and provides foundational data for handling complex fee structures.

- **Variable Annuity Pricing (Section 11)**: This is a **two-dimensional problem** requiring the full two-layer framework. It demonstrates the method's efficiency in handling long-maturity, option-embedded (early surrender) products, with particular emphasis on the **Fast Algorithm**.



## 9. CTMC Approximation of the VIX Index

The VIX index provides a natural and illustrative example of how the first-layer CTMC can be used to obtain efficient matrix-based approximations for quantities that may otherwise lack analytical expressions. This section follows **Proposition 4.13** and **Algorithm 6** in Mackay, Vachon, and Cui (2023).

### 9.1 Definition of the VIX

The squared VIX index is defined as the risk-neutral expected average variance over the next $\tau$ units of time ($\tau = 30/365 \approx 0.0822$ years):

$$
\text{VIX}_t^2 := \mathbb{E}\!\left[\frac{1}{\tau}\int_t^{t+\tau} \sigma_S^2(V_s)\,\text{d}s \;\middle|\; \mathcal{F}_t\right].
\tag{10.1}
$$

This is a model-free definition in the sense that it applies to any SV model with volatility function $\sigma_S(\cdot)$. However, computing the right-hand side of (10.1) analytically requires knowledge of the conditional distribution of the path $\{V_s\}_{s \in [t, t+\tau]}$, which is available in closed form only for specific models (e.g., Heston).

### 9.2 The Challenge: Non-Linear $\sigma_S$

Under the Heston model, $\sigma_S^2(V) = V$ is linear, and the integral in (10.1) reduces to $\frac{1}{\tau}\int_t^{t+\tau}\mathbb{E}[V_s \mid V_t]\,\text{d}s$, which is elementary because $\mathbb{E}[V_s \mid V_t]$ has a known closed form (the CIR mean). But under models where $\sigma_S$ is non-linear — such as:

*   **3/2 model:** $\sigma_S^2(V) = 1/V$ (where $V$ follows a 3/2-type process),
*   **4/2 model:** $\sigma_S^2(V) = (a\sqrt{V} + b/\sqrt{V})^2$,
*   **Scott model:** $\sigma_S^2(V) = e^{2V}$,

In the cases above, an analytical evaluation is generally impossible. This is where the CTMC approximation proves useful.

### 9.3 CTMC Approximation Formula

Given the first-layer CTMC $V^{(m)}$ with generator $\mathbf{Q}^{(m)}$ and current state $V_t^{(m)} = v_k$, we approximate (10.1) as:

$$
\left(\text{VIX}_t^{(m),k}\right)^2 = \frac{1}{\tau}\int_0^{\tau} \mathbf{e}_k \, \exp(\mathbf{Q}^{(m)} s) \, \mathbf{H} \,\text{d}s,
\tag{10.2}
$$

where $\mathbf{e}_k$ is the $k$-th standard basis row vector ($1 \times m$) and $\mathbf{H} \in \mathbb{R}^{m \times 1}$ is the column vector of instantaneous variances:

$$
\mathbf{H} = \begin{pmatrix} \sigma_S^2(v_1) \\ \sigma_S^2(v_2) \\ \vdots \\ \sigma_S^2(v_m) \end{pmatrix}.
\tag{10.3}
$$

**Derivation.** Under the CTMC approximation:

$$
\mathbb{E}\!\left[\sigma_S^2(V_s^{(m)}) \;\middle|\; V_t^{(m)} = v_k\right]
= \sum_{j=1}^{m} \sigma_S^2(v_j) \, \mathbb{P}(V_s^{(m)} = v_j \mid V_t^{(m)} = v_k)
= \mathbf{e}_k \, \exp(\mathbf{Q}^{(m)}(s-t)) \, \mathbf{H}.
$$

Substituting into (10.1) and integrating over $s \in [t, t+\tau]$ yields (10.2).

### 9.4 Time-Homogeneity

Because $V^{(m)}$ is a time-homogeneous CTMC, the expression (10.2) depends on $V_t^{(m)} = v_k$ but not on $t$ itself. This means:

1.  The VIX approximation needs to be computed only **once** at the beginning of the pricing procedure.
2.  It provides the VIX value simultaneously for **all $m$ variance states**.

### 9.5 Numerical Evaluation: Quadrature Algorithm

The integral in (10.2) is evaluated by a simple quadrature with $n$ subintervals of size $\Delta_n = \tau / n$.

---

**Algorithm 10.1: CTMC Approximation of the VIX**

**Input:** Generator $\mathbf{Q}^{(m)}$; payoff vector $\mathbf{H}$; quadrature steps $n$.

1.  Set $\Delta_n \leftarrow \tau / n$.
2.  Compute $\mathbf{A} \leftarrow \exp(\Delta_n \, \mathbf{Q}^{(m)})$. (Single $m \times m$ matrix exponential.)
3.  Initialize $\mathbf{E} \leftarrow \mathbf{H}$, $\;\mathbf{S} \leftarrow \mathbf{0}_{m \times 1}$.
4.  **For** $z = n, n-1, \ldots, 1$:
    *   $\mathbf{E} \leftarrow \mathbf{A} \, \mathbf{E}$
    *   $\mathbf{S} \leftarrow \mathbf{S} + \mathbf{E}$
5.  **Return** $\text{VIX}^{(m),k} = \sqrt{\mathbf{e}_k \, \mathbf{S} \, \Delta_n / \tau}$ for each $k = 1, \ldots, m$.

---

**Computational cost:** One $m \times m$ matrix exponential plus $n$ matrix–vector multiplications. With $m = 50$ and $n = 100$, this takes a fraction of a second.

### 9.6 Benchmark: Heston Model Closed-Form VIX

Under the Heston model, the VIX admits an exact expression (since $\sigma_S^2(V) = V$ and $V$ is a CIR process):

$$
\text{VIX}_t^2 = B + A \, V_t,
\tag{10.4}
$$

where

$$
A = \frac{1 - e^{-\kappa\tau}}{\kappa\tau}, \qquad B = \frac{\theta(\kappa\tau - 1 + e^{-\kappa\tau})}{\kappa\tau}.
$$

This closed-form solution serves as a useful benchmark for validating the CTMC approximation. Numerical experiments in **Appendix B.2** of Mackay et al. (2023) confirm that $\text{VIX}^{(m),k}$ converges rapidly to the exact value as $m$ increases: with $m = 50$, the relative error is typically below $10^{-4}$.

### 9.7 Models Without Closed-Form VIX

For models such as the 3/2, 4/2, Scott, or $\alpha$-Hypergeometric models, no closed-form VIX expression is available. In these cases, the CTMC approximation provides a systematic, efficient, and arbitrarily accurate alternative. This is one of the key practical advantages of the CTMC framework.



## 10. Variable Annuity Pricing Algorithms

This section presents the complete pricing algorithms for variable annuities (VAs) under the two-layer CTMC framework. We cover both European-style contracts (no early surrender) and Bermudan-style contracts (with early surrender rights), along with efficient fast algorithms for long-maturity products and procedures to approximate the optimal surrender surface.

### 10.1 European VA: No Early Surrender

#### Problem Setup

Consider a variable annuity with guaranteed minimum maturity benefit (GMMB). The policyholder deposits an initial premium $F_0$ into a sub-account invested in the risky asset. At maturity $T$, the payoff is:

$$
\phi(T, F_T, V_T) = \max(G, F_T),
\tag{11.1}
$$

where $G \in \mathbb{R}_+$ is the predetermined guaranteed amount.

Under the risk-neutral measure $\mathbb{Q}$, the time-0 value of the VA assuming no early surrender is:

$$
v_e(0, F_0, V_0) = \mathbb{E}\!\left[e^{-rT} \max(G, F_T) \;\middle|\; F_0, V_0\right].
\tag{11.2}
$$

#### CTMC Approximation

Using the two-layer CTMC approximation $(X^{(m,N)}, V^{(m)})$ from Section 7, the fund value is approximated by:

$$
F_t^{(m,N)} = \exp\!\left(X_t^{(m,N)} + \rho \, \gamma(V_t^{(m)})\right).
\tag{11.3}
$$

Let $\mathbf{G}^{(m,N)}$ be the combined generator from Proposition 7.1. Define the payoff vector $\mathbf{H} \in \mathbb{R}^{mN \times 1}$ with entries:

$$
h_{(l-1)N + n} = \max\!\left(G, \, e^{x_n + \rho \gamma(v_l)}\right), \quad 1 \leq l \leq m, \; 1 \leq n \leq N.
\tag{11.4}
$$

**Proposition 11.1 (European VA Price).** Let $F_0 > 0$ be the initial premium, with $X_0^{(m,N)} = \ln(F_0) - \rho \gamma(V_0) = x_i \in \mathcal{S}_X^{(N)}$ and $V_0 = v_k \in \mathcal{S}_V^{(m)}$. The risk-neutral value at time 0 of a VA contract held until maturity $T$ can be approximated by:

$$
v_e^{(m,N)}(0, F_0, V_0) = e^{-rT} \, \mathbf{e}_{ik} \, \exp\!\left(T \, \mathbf{G}^{(m,N)}\right) \mathbf{H},
\tag{11.5}
$$

where $\mathbf{e}_{ik}$ is the standard basis row vector of size $1 \times mN$ with a 1 at position $(k-1)N + i$.

**Proof.** This follows directly from the matrix exponential representation of conditional expectations for CTMCs (Section 1.5). 

---

**Algorithm 10.1: European VA via CTMC (Regular)**

**Input:** Combined generator $\mathbf{G}^{(m,N)}$; payoff vector $\mathbf{H}$; maturity $T$; risk-free rate $r$; initial state indices $(i, k)$.

1.  Compute $\mathbf{P}_T \leftarrow \exp(T \, \mathbf{G}^{(m,N)})$. ($mN \times mN$ matrix exponential.)
2.  Compute $\mathbf{V} \leftarrow \mathbf{P}_T \, \mathbf{H}$.
3.  **Return** $v_e^{(m,N)} = e^{-rT} \, \mathbf{e}_{ik} \, \mathbf{V}$.

---

**Computational cost:** One $mN \times mN$ matrix exponential. For $m = 50, N = 2000$, this becomes computationally prohibitive for long maturities.



### 10.2 Bermudan VA: With Early Surrender

#### Problem Setup

When the policyholder has the right to surrender the contract before maturity, the pricing problem becomes an optimal stopping problem. We approximate the American-style surrender right by a Bermudan contract with $M$ exercise dates $\mathcal{H}_M = \{t_0, t_1, \ldots, t_M\}$, where $t_z = z \Delta_M$ and $\Delta_M = T/M$.

At each exercise date $t_z < T$, the policyholder receives the surrender value:

$$
\phi(t_z, F_{t_z}, V_{t_z}) = g(t_z, V_{t_z}) \, F_{t_z},
\tag{11.6}
$$

where $g: [0, T] \times \mathcal{S}_V \to [0, 1]$ is the surrender charge function. A common form is $g(t, y) = e^{-\kappa(T-t)}$ for some constant $\kappa \geq 0$.

At maturity $t_M = T$, the payoff is $\phi(T, F_T, V_T) = \max(G, F_T)$.

#### Dynamic Programming Formulation

Let $B_z^{(m,N)}$ denote the VA value at time $t_z$ under the CTMC approximation. By the principle of dynamic programming:

$$
\begin{cases}
B_M^{(m,N)} = \mathbf{H}^{(1)}, \\[6pt]
B_z^{(m,N)} = \max\!\left\{\mathbf{H}_z^{(2)}, \; e^{-r \Delta_M} \exp\!\left(\Delta_M \, \mathbf{G}^{(m,N)}\right) B_{z+1}^{(m,N)}\right\}, \quad 0 \leq z \leq M-1,
\end{cases}
\tag{11.7}
$$

where the maximum is taken element-wise, and:

*   $\mathbf{H}^{(1)} \in \mathbb{R}^{mN \times 1}$ with entries $h_{(l-1)N+n}^{(1)} = \max\!\left(G, \, e^{x_n + \rho \gamma(v_l)}\right)$,
*   $\mathbf{H}_z^{(2)} \in \mathbb{R}^{mN \times 1}$ with entries $h_{z, (l-1)N+n}^{(2)} = g(t_z, v_l) \, e^{x_n + \rho \gamma(v_l)}$.

**Proposition 11.2 (Bermudan VA Price).** Under the CTMC approximation, the time-0 value of a Bermudan VA contract is:

$$
b_M^{(m,N)}(0, F_0, V_0) = \mathbf{e}_{ik} \, B_0^{(m,N)}.
\tag{11.8}
$$

As $M \to \infty$, $b_M^{(m,N)} \to v(0, F_0, V_0)$, the American-style VA value (Proposition 4.5, Mackay et al., 2023).

---

**Algorithm 10.2: Bermudan VA via CTMC (Regular)**

**Input:** Combined generator $\mathbf{G}^{(m,N)}$; payoff vectors $\mathbf{H}^{(1)}, \{\mathbf{H}_z^{(2)}\}_{z=0}^{M-1}$; number of time steps $M$; $\Delta_M = T/M$; $r$; initial state indices $(i, k)$.

1.  Set $B_M \leftarrow \mathbf{H}^{(1)}$.
2.  Compute $\mathbf{A}_{\Delta_M} \leftarrow \exp(\Delta_M \, \mathbf{G}^{(m,N)}) \, e^{-r \Delta_M}$. (Single $mN \times mN$ matrix exponential.)
3.  **For** $z = M-1, M-2, \ldots, 0$:
    *   $B_z \leftarrow \max\!\left\{\mathbf{H}_z^{(2)}, \; \mathbf{A}_{\Delta_M} \, B_{z+1}\right\}$. (Element-wise maximum.)
4.  **Return** $b_M^{(m,N)} = \mathbf{e}_{ik} \, B_0$.

---

**Computational cost:** One $mN \times mN$ matrix exponential plus $M$ matrix–vector multiplications. For long-maturity VAs ($T = 10$–$20$ years), this is still computationally expensive.



### 10.3 Fast Algorithms for Long-Maturity VAs

For VAs with long maturities, computing the exponential of the large $mN \times mN$ matrix $\mathbf{G}^{(m,N)}$ is computationally prohibitive. The **Fast Algorithm** exploits the block structure of $\mathbf{G}^{(m,N)}$ and the tower property of conditional expectations to reduce computational cost significantly.

#### Key Idea

Instead of computing $\exp(T \, \mathbf{G}^{(m,N)})$ directly, we work with the smaller matrices $\{\mathbf{G}_l^{(N)}\}_{l=1}^m$ ($N \times N$) and $\mathbf{Q}^{(m)}$ ($m \times m$) separately over short time intervals $\Delta_M = T/M$. This is based on the approximation (Proposition 4.3, Mackay et al., 2023):

$$
\begin{aligned}
&\mathbb{E}\!\left[\phi(t+h, X_{t+h}^{(m,N)}, V_{t+h}^{(m)}) \;\middle|\; X_t^{(m,N)} = x_i, V_t^{(m)} = v_k\right] \\
&\quad \approx \sum_{j=1}^m \mathbb{E}\!\left[\phi(t+h, X_{t+h}^{(m,N)}, v_j) \;\middle|\; V_t^{(m)} = V_{t+h}^{(m)} = v_j, X_t^{(m,N)} = x_i\right] \cdot \mathbb{P}(V_{t+h}^{(m)} = v_j \mid V_t^{(m)} = v_k).
\end{aligned}
\tag{11.9}
$$

#### Fast Algorithm for European VA

---

**Algorithm 10.3: European VA via CTMC (Fast)**

**Input:** First-layer generator $\mathbf{Q}^{(m)}$; regime-specific generators $\{\mathbf{G}_l^{(N)}\}_{l=1}^m$; payoff matrix $\mathbf{\Phi} \in \mathbb{R}^{m \times N}$ with $\Phi_{l,n} = \phi(T, e^{x_n + \rho \gamma(v_l)}, v_l)$; $M$ time steps; $\Delta_M = T/M$; $r$; initial state indices $(i, k)$.

1.  **Pre-compute transition matrices:**
    *   **For** $l = 1, 2, \ldots, m$:
        *   $\mathbf{P}_l^X \leftarrow \exp(\Delta_M \, \mathbf{G}_l^{(N)})$. ($m$ matrix exponentials of size $N \times N$.)
    *   $\mathbf{P}^V \leftarrow \exp(\Delta_M \, \mathbf{Q}^{(m)})$. (One matrix exponential of size $m \times m$.)
2.  **Initialize:** $B_{l,n} \leftarrow \Phi_{l,n}$ for $l=1,\ldots,m$, $n=1,\ldots,N$.
3.  **Backward induction:** **For** $z = M-1, M-2, \ldots, 0$:
    *   **For** $l = 1, 2, \ldots, m$:
        *   $\tilde{\mathbf{E}}_{l, \ast} \leftarrow \mathbf{P}_l^X \, B_{l, \ast}^\top$. (Continuation value within regime $l$.)
    *   **For** $n = 1, 2, \ldots, N$:
        *   $B_{\ast, n} \leftarrow \mathbf{P}^V \, \tilde{\mathbf{E}}_{\ast, n}$. (Aggregate across variance regimes.)
    *   Apply discounting: $B \leftarrow e^{-r \Delta_M} \, B$.
4.  **Return** $v_e^{(m,N)} = B_{k, i}$.

---

**Computational cost:** $m$ matrix exponentials of size $N \times N$ plus one of size $m \times m$, plus $M \times (m + N)$ matrix–vector multiplications. For $m = 50, N = 100, M = 5000$, this takes approximately 6 seconds compared to hours for the regular algorithm.

#### Fast Algorithm for Bermudan VA

The Bermudan case adds one additional step: comparing the continuation value with the immediate surrender payoff at each exercise date.

---

**Algorithm 10.4: Bermudan VA via CTMC (Fast)**

**Input:** Same as Algorithm 11.3, plus surrender payoff matrices $\{\mathbf{\Psi}_z\}_{z=0}^{M-1}$ with $(\Psi_z)_{l,n} = g(t_z, v_l) \, e^{x_n + \rho \gamma(v_l)}$. **Note:** To compute the surrender surface later, store all matrices $\{B_z\}_{z=0}^M$ during the loop.

1.  **Pre-compute transition matrices:** (Same as Algorithm 11.3, Step 1.)
2.  **Initialize:** $B_{l,n} \leftarrow \Phi_{l,n}$ for $l=1,\ldots,m$, $n=1,\ldots,N$.
3.  **Backward induction:** **For** $z = M-1, M-2, \ldots, 0$:
    *   **For** $l = 1, 2, \ldots, m$:
        *   $\tilde{\mathbf{E}}_{l, \ast} \leftarrow \mathbf{P}_l^X \, B_{l, \ast}^\top$.
    *   **For** $n = 1, 2, \ldots, N$:
        *   $B_{\ast, n} \leftarrow \mathbf{P}^V \, \tilde{\mathbf{E}}_{\ast, n}$.
    *   Apply discounting: $B \leftarrow e^{-r \Delta_M} \, B$.
    *   **Early exercise check:** $B \leftarrow \max(B, \mathbf{\Psi}_z)$. (Element-wise maximum.)
    *   (Optional for Surface): Store $B_z \leftarrow B$.
4.  **Return** $b_M^{(m,N)} = B_{k, i}$.

---

**Accuracy:** The absolute difference between regular and fast algorithms is typically around $10^{-3}$, with relative difference around $10^{-5}$ (Section 5.2, Mackay et al., 2023).

### 10.4 Optimal Surrender Surface

The optimal surrender strategy is characterized by the surrender region $\mathcal{D}$ and continuation region $\mathcal{C}$:

$$
\begin{aligned}
\mathcal{C} &= \{(t, x, y) : v(t, x, y) > \phi(t, x, y)\}, \\
\mathcal{D} &= \{(t, x, y) : v(t, x, y) = \phi(t, x, y)\}.
\end{aligned}
\tag{11.10}
$$

Under the **threshold-type assumption** (common in practice), for each $(t, y)$, there exists a critical fund value $f^*(t, y)$ such that surrender is optimal if and only if $F_t \geq f^*(t, y)$.

#### Approximation Algorithm

Using the Bermudan VA values $B_z^{(m,N)}$ from Algorithm 11.4, we can approximate the optimal surrender surface.

---

**Algorithm 10.5: Optimal Surrender Surface (Threshold Type)**

**Input:** Bermudan VA value matrices $\{B_z\}_{z=0}^M$ stored from Algorithm 11.4; surrender payoff matrices $\{\mathbf{\Psi}_z\}_{z=0}^{M-1}$; grid states $\{x_n\}_{n=1}^N$, $\{v_l\}_{l=1}^m$; transformation $\gamma(\cdot)$.

1.  **Initialize:** $f^* \leftarrow \text{empty matrix of size } M \times m$.
2.  **For** $z = 0, 1, \ldots, M-1$:
    *   **For** $l = 1, 2, \ldots, m$:
        *   $n \leftarrow 1$.
        *   **While** $B_{z, l, n} > (\Psi_z)_{l, n}$ and $n < N$:
            *   $n \leftarrow n + 1$.
        *   $f^*(t_z, v_l) \leftarrow e^{x_n + \rho \gamma(v_l)}$.
3.  **Return** $f^*$.

---

**Output:** A matrix $f^*$ where $f^*_{z, l}$ approximates the critical fund value at time $t_z$ and variance state $v_l$.

**Remark:** Algorithms 11.4 and 11.5 do not require the surrender region to be of threshold type. The threshold assumption is only used for interpreting and visualizing the optimal surrender strategy.

### 10.5 Integration with VIX-Linked Fees

The VIX approximation from Section 10 integrates seamlessly with the VA pricing algorithms, enabling efficient pricing of VAs with volatility-dependent fee structures. This follows Section 5.3 of Mackay, Vachon, and Cui (2023).

#### VIX-Linked Fee Structures

Three common VIX-linked fee structures are considered:

1.  **Uncapped VIX²:** $c_t = \tilde{c} + \tilde{m} \, \text{VIX}_t^2$
2.  **Capped VIX²:** $c_t = \min(\tilde{c} + \tilde{m} \, \text{VIX}_t^2, K)$
3.  **Uncapped VIX:** $c_t = \tilde{c} + \tilde{m} \, \text{VIX}_t$

#### Implementation Procedure

The complete pricing procedure for VAs with VIX-linked fees is as follows:

---

**Algorithm 10.6: VA Pricing with VIX-Linked Fees**

1.  **Pre-compute VIX:** Use Algorithm 10.1 to compute $\text{VIX}^{(m),k}$ for all $k = 1, \ldots, m$.
2.  **Determine fee rates:** For each variance state $v_k$, compute $c^{(m),k} = c(\text{VIX}^{(m),k})$ based on the fee structure.
3.  **Adjust drift:** Incorporate $c^{(m),k}$ into the second-layer drift function $\mu_X(x, v_k)$ (Eq. 6.6).
4.  **Construct generators:** Build regime-specific generators $\mathbf{G}_l^{(N)}$ using the adjusted drift.
5.  **Price VA:** Use Algorithm 11.3 or 11.4 to price the VA contract.

