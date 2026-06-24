"""Diagnose 2D CTMC Heston pricing accuracy."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loguru import logger
logger.remove()

from commonConfig import LAYER1_GRID_CONFIG, LAYER2_GRID_CONFIG
from src.grid_construction import build_variance_grid, build_price_grid
from src.layer1_variance import construct_variance_generator
from src.layer2_price import construct_all_regime_generators, compute_initial_auxiliary_value, get_price_state_index
from src.layer1_variance import get_variance_state_index
from src.option_pricing import price_european_strang
from src.heston_analytical import compute_heston_price

mp = {
    'S_0': 100.0, 'V_0': 0.04, 'r': 0.03,
    'kappa': 2.0, 'theta': 0.04, 'sigma_v': 0.3, 'rho': -0.7,
    'underlying_type': 'spot'
}
op = {'K': 100.0, 'T': 1.0, 'option_type': 'put'}

ref = compute_heston_price(100, 100, 0.04, 0.03, 2.0, 0.04, 0.3, -0.7, 1.0, 'put')
print(f'Heston analytical put (ref): {ref:.6f}')

# ============================================================
# Experiment 1: Convergence with grid size
# ============================================================
print("\n=== Grid Convergence (Strang splitting, 200 steps) ===")
for m, N in [(20,100),(30,150),(40,200),(50,300),(60,400)]:
    l1 = dict(LAYER1_GRID_CONFIG, m=m)
    l2 = dict(LAYER2_GRID_CONFIG, N=N)
    vg = build_variance_grid(l1, mp)
    pg = build_price_grid(l2, mp, vg)
    Qv = construct_variance_generator(vg, mp)
    Gt = construct_all_regime_generators(pg, vg, mp, Q_variance=Qv)
    price = price_european_strang(Qv, Gt, pg, vg, mp, op, n_time_steps=200)
    err = abs(price - ref)
    print(f'  m={m:2d} N={N:3d}: price={price:.6f}, err={err:.4f}, rel={err/ref*100:.2f}%')

# ============================================================
# Experiment 2: Time step convergence (fixed m=40, N=200)
# ============================================================
print("\n=== Time Step Convergence (m=40, N=200) ===")
l1 = dict(LAYER1_GRID_CONFIG, m=40)
l2 = dict(LAYER2_GRID_CONFIG, N=200)
vg = build_variance_grid(l1, mp)
pg = build_price_grid(l2, mp, vg)
Qv = construct_variance_generator(vg, mp)
Gt = construct_all_regime_generators(pg, vg, mp, Q_variance=Qv)

for n_steps in [50, 100, 200, 400, 800]:
    price = price_european_strang(Qv, Gt, pg, vg, mp, op, n_time_steps=n_steps)
    err = abs(price - ref)
    print(f'  steps={n_steps:4d}: price={price:.6f}, err={err:.4f}, rel={err/ref*100:.2f}%')

# ============================================================
# Experiment 3: Effect of rho on accuracy
# ============================================================
print("\n=== Rho Sensitivity (m=40, N=200, 400 steps) ===")
for rho_val in [-0.9, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 0.9]:
    mp_rho = dict(mp, rho=rho_val)
    ref_rho = compute_heston_price(100, 100, 0.04, 0.03, 2.0, 0.04, 0.3, rho_val, 1.0, 'put')
    
    vg = build_variance_grid(l1, mp_rho)
    pg = build_price_grid(l2, mp_rho, vg)
    Qv = construct_variance_generator(vg, mp_rho)
    Gt = construct_all_regime_generators(pg, vg, mp_rho, Q_variance=Qv)
    price = price_european_strang(Qv, Gt, pg, vg, mp_rho, op, n_time_steps=400)
    err = abs(price - ref_rho)
    print(f'  rho={rho_val:+.1f}: ref={ref_rho:.6f}, CTMC={price:.6f}, err={err:.4f}, rel={err/ref_rho*100:.2f}%')

# ============================================================
# Experiment 4: rho=0 case (should be most accurate)
# ============================================================
print("\n=== rho=0 Special Case ===")
mp0 = dict(mp, rho=0.0)
ref0 = compute_heston_price(100, 100, 0.04, 0.03, 2.0, 0.04, 0.3, 0.0, 1.0, 'put')
print(f'  ref: {ref0:.6f}')
for m, N in [(20,100),(40,200),(60,400)]:
    l1 = dict(LAYER1_GRID_CONFIG, m=m)
    l2 = dict(LAYER2_GRID_CONFIG, N=N)
    vg = build_variance_grid(l1, mp0)
    pg = build_price_grid(l2, mp0, vg)
    Qv = construct_variance_generator(vg, mp0)
    Gt = construct_all_regime_generators(pg, vg, mp0, Q_variance=Qv)
    price = price_european_strang(Qv, Gt, pg, vg, mp0, op, n_time_steps=400)
    err = abs(price - ref0)
    print(f'  m={m:2d} N={N:3d}: CTMC={price:.6f}, err={err:.6f}, rel={err/ref0*100:.4f}%')
