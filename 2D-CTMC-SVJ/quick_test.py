"""Quick test: 2D CTMC Heston pricing accuracy."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loguru import logger
logger.remove()

from commonConfig import LAYER1_GRID_CONFIG, LAYER2_GRID_CONFIG
from src.grid_construction import build_variance_grid, build_price_grid
from src.layer1_variance import construct_variance_generator
from src.layer2_price import construct_all_regime_generators
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

for m, N in [(20, 100), (30, 150), (40, 200)]:
    l1_cfg = dict(LAYER1_GRID_CONFIG, m=m)
    l2_cfg = dict(LAYER2_GRID_CONFIG, N=N)

    v_grid = build_variance_grid(l1_cfg, mp)
    p_grid = build_price_grid(l2_cfg, mp, v_grid)

    Q_v = construct_variance_generator(v_grid, mp)
    G_tensor = construct_all_regime_generators(p_grid, v_grid, mp, Q_variance=Q_v)

    t0 = time.perf_counter()
    result = price_european_strang(Q_v, G_tensor, p_grid, v_grid, mp, op, n_time_steps=200)
    elapsed = time.perf_counter() - t0

    ctmc_price = result['price']
    err = abs(ctmc_price - ref)
    rel = err / ref * 100
    print(f'm={m:2d}, N={N:3d}: CTMC={ctmc_price:.6f}, err={err:.6f}, rel={rel:.4f}%, time={elapsed:.2f}s')
