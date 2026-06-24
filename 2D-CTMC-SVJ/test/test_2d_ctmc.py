# -*- coding: utf-8 -*-
"""
文件名: test/test_2d_ctmc.py
功能描述: 2D-CTMC 核心模块单元测试
         覆盖网格构建、生成元构造、Heston 闭式解、定价流程
作者: [Author]
创建日期: 2026-05-06
"""

import unittest
import copy
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commonConfig import HESTON_DEFAULT_PARAMS, LAYER1_GRID_CONFIG, LAYER2_GRID_CONFIG
from src.grid_construction import (
    generate_uniform_grid,
    generate_sinh_grid,
    build_variance_grid,
    build_price_grid,
    validate_grid,
)
from src.layer1_variance import (
    compute_cir_coefficients,
    construct_variance_generator,
    get_variance_state_index,
)
from src.layer2_price import (
    compute_auxiliary_coefficients,
    compute_decorrelation_function,
    construct_regime_generator,
    construct_all_regime_generators,
    compute_initial_auxiliary_value,
    recover_price_from_auxiliary,
)
from src.combined_generator import (
    construct_combined_generator,
    flatten_state_index,
    unflatten_state_index,
)
from src.heston_analytical import compute_heston_price


class TestGridConstruction(unittest.TestCase):
    """网格构建测试"""


    def test_uniform_grid_length(self):
        grid = generate_uniform_grid(0.0, 1.0, 50)
        self.assertEqual(len(grid), 50)

    def test_uniform_grid_monotone(self):
        grid = generate_uniform_grid(0.0, 10.0, 100)
        self.assertTrue(np.all(np.diff(grid) > 0))

    def test_sinh_grid_length(self):
        grid = generate_sinh_grid(0.01, 0.5, 30, 0.04, 0.3)
        self.assertEqual(len(grid), 30)

    def test_sinh_grid_monotone(self):
        grid = generate_sinh_grid(0.01, 0.5, 30, 0.04, 0.3)
        self.assertTrue(validate_grid(grid))

    def test_sinh_grid_concentration(self):
        grid = generate_sinh_grid(0.01, 0.5, 50, 0.04, 0.1)
        center_idx = np.argmin(np.abs(grid - 0.04))
        center_spacing = grid[center_idx + 1] - grid[center_idx]
        edge_spacing = (grid[-1] - grid[-2])
        self.assertLess(center_spacing, edge_spacing)

    def test_sinh_grid_invalid_alpha(self):
        with self.assertRaises(ValueError):
            generate_sinh_grid(0.01, 0.5, 30, 0.04, -1.0)

    def test_build_variance_grid(self):
        model_params = copy.deepcopy(HESTON_DEFAULT_PARAMS)
        config = copy.deepcopy(LAYER1_GRID_CONFIG)
        config['m'] = 20
        grid = build_variance_grid(config, model_params)
        self.assertEqual(len(grid), 20)
        self.assertTrue(np.all(grid > 0))

    def test_build_price_grid(self):
        model_params = copy.deepcopy(HESTON_DEFAULT_PARAMS)
        l1_config = copy.deepcopy(LAYER1_GRID_CONFIG)
        l1_config['m'] = 20
        v_grid = build_variance_grid(l1_config, model_params)
        l2_config = copy.deepcopy(LAYER2_GRID_CONFIG)
        l2_config['N'] = 50
        p_grid = build_price_grid(l2_config, model_params, v_grid)
        self.assertEqual(len(p_grid), 50)


class TestLayer1Variance(unittest.TestCase):
    """Layer 1 方差过程测试"""


    def setUp(self):
        self.model_params = copy.deepcopy(HESTON_DEFAULT_PARAMS)
        self.variance_grid = np.linspace(0.005, 0.2, 30)

    def test_cir_coefficients(self):
        mu, sigma_sq = compute_cir_coefficients(0.04, self.model_params)
        self.assertAlmostEqual(mu, 0.0, places=10)
        self.assertAlmostEqual(sigma_sq, 0.04 * 0.3 ** 2, places=10)

    def test_cir_coefficients_vectorized(self):
        v = np.array([0.01, 0.04, 0.1])
        mu, sigma_sq = compute_cir_coefficients(v, self.model_params)
        self.assertEqual(len(mu), 3)
        self.assertTrue(np.all(sigma_sq >= 0))

    def test_generator_shape(self):
        Q = construct_variance_generator(self.variance_grid, self.model_params)
        self.assertEqual(Q.shape, (30, 30))

    def test_generator_row_sums_zero(self):
        Q = construct_variance_generator(self.variance_grid, self.model_params)
        row_sums = np.abs(Q.sum(axis=1))
        self.assertTrue(np.all(row_sums < 1e-8))

    def test_generator_off_diag_nonneg(self):
        Q = construct_variance_generator(self.variance_grid, self.model_params)
        off_diag = Q.copy()
        np.fill_diagonal(off_diag, 0)
        self.assertTrue(np.all(off_diag >= -1e-10))

    def test_generator_diag_nonpos(self):
        Q = construct_variance_generator(self.variance_grid, self.model_params)
        self.assertTrue(np.all(np.diag(Q) <= 1e-10))

    def test_get_variance_state_index(self):
        idx = get_variance_state_index(0.04, self.variance_grid)
        self.assertLessEqual(
            abs(self.variance_grid[idx] - 0.04),
            (self.variance_grid[1] - self.variance_grid[0]) * 1.1
        )


class TestLayer2Price(unittest.TestCase):
    """Layer 2 价格过程测试"""


    def setUp(self):
        self.model_params = copy.deepcopy(HESTON_DEFAULT_PARAMS)
        self.price_grid = np.linspace(3.5, 5.5, 50)
        self.variance_grid = np.linspace(0.005, 0.2, 20)

    def test_decorrelation_heston(self):
        gamma_val, gamma_prime = compute_decorrelation_function(
            0.04, self.model_params
        )
        self.assertAlmostEqual(gamma_val, 0.04 / 0.3, places=10)
        self.assertAlmostEqual(gamma_prime, 1.0 / 0.3, places=10)

    def test_auxiliary_coefficients(self):
        mu, sigma_sq = compute_auxiliary_coefficients(
            4.5, 0.04, self.model_params
        )
        self.assertTrue(np.isfinite(mu))
        self.assertTrue(sigma_sq >= 0)

    def test_regime_generator_shape(self):
        G = construct_regime_generator(
            self.price_grid, 0.04, self.model_params
        )
        self.assertEqual(G.shape, (50, 50))

    def test_regime_generator_row_sums(self):
        G = construct_regime_generator(
            self.price_grid, 0.04, self.model_params
        )
        row_sums = np.abs(G.sum(axis=1))
        self.assertTrue(np.all(row_sums < 1e-8))

    def test_all_regime_generators(self):
        G_tensor = construct_all_regime_generators(
            self.price_grid, self.variance_grid, self.model_params
        )
        self.assertEqual(G_tensor.shape, (20, 50, 50))

    def test_initial_auxiliary_value(self):
        X_0 = compute_initial_auxiliary_value(
            self.model_params['S_0'], self.model_params['V_0'], self.model_params
        )
        rho = self.model_params['rho']
        sigma_v = self.model_params['sigma_v']
        expected = np.log(self.model_params['S_0']) - rho * self.model_params['V_0'] / sigma_v
        self.assertAlmostEqual(X_0, expected, places=10)

    def test_recover_price(self):
        S_0 = self.model_params['S_0']
        V_0 = self.model_params['V_0']
        X_0 = compute_initial_auxiliary_value(S_0, V_0, self.model_params)
        S_recovered = recover_price_from_auxiliary(X_0, V_0, self.model_params)
        self.assertAlmostEqual(S_recovered, S_0, places=5)


class TestCombinedGenerator(unittest.TestCase):
    """组合生成元测试类"""

    def setUp(self):
        self.model_params = copy.deepcopy(HESTON_DEFAULT_PARAMS)
        self.model_params['V_0'] = 0.04

        self.variance_grid = np.linspace(0.005, 0.2, 10)
        self.price_grid = np.linspace(4.0, 5.0, 20)

        self.Q = construct_variance_generator(self.variance_grid, self.model_params)
        self.G_list = construct_all_regime_generators(
            self.price_grid, self.variance_grid, self.model_params
        )

    def test_combined_shape(self):
        G = construct_combined_generator(
            self.Q, self.G_list, use_sparse=False
        )
        self.assertEqual(G.shape, (200, 200))

    def test_combined_row_sums(self):
        G = construct_combined_generator(
            self.Q, self.G_list, use_sparse=False
        )
        row_sums = np.abs(G.sum(axis=1))
        self.assertTrue(np.all(row_sums < 1e-8))

    def test_combined_off_diag(self):
        G = construct_combined_generator(
            self.Q, self.G_list, use_sparse=False
        )
        off_diag = G.copy()
        np.fill_diagonal(off_diag, 0)
        self.assertTrue(np.all(off_diag >= -1e-10))

    def test_sparse_combined(self):
        G = construct_combined_generator(
            self.Q, self.G_list, use_sparse=True
        )
        self.assertEqual(G.shape, (200, 200))

    def test_flatten_unflatten(self):
        N = 20
        for pi in range(5):
            for vi in range(5):
                flat = flatten_state_index(pi, vi, N)
                pi2, vi2 = unflatten_state_index(flat, N)
                self.assertEqual(pi, pi2)
                self.assertEqual(vi, vi2)


class TestHestonAnalytical(unittest.TestCase):
    """Heston 闭式解测试类"""

    def setUp(self):
        self.params = copy.deepcopy(HESTON_DEFAULT_PARAMS)

    def test_call_put_parity(self):
        call = compute_heston_price(
            self.params['S_0'], 100.0, self.params['V_0'], self.params['r'],
            self.params['kappa'], self.params['theta'], self.params['sigma_v'],
            self.params['rho'], 1.0, 'call'
        )
        put = compute_heston_price(
            self.params['S_0'], 100.0, self.params['V_0'], self.params['r'],
            self.params['kappa'], self.params['theta'], self.params['sigma_v'],
            self.params['rho'], 1.0, 'put'
        )
        diff = call - put
        expected = self.params['S_0'] - 100 * np.exp(-self.params['r'])
        self.assertAlmostEqual(diff, expected, places=1)

    def test_call_price_positive(self):
        price = compute_heston_price(
            self.params['S_0'], 100.0, self.params['V_0'], self.params['r'],
            self.params['kappa'], self.params['theta'], self.params['sigma_v'],
            self.params['rho'], 1.0, 'call'
        )
        self.assertGreater(price, 0)

    def test_put_price_positive(self):
        price = compute_heston_price(
            self.params['S_0'], 100.0, self.params['V_0'], self.params['r'],
            self.params['kappa'], self.params['theta'], self.params['sigma_v'],
            self.params['rho'], 1.0, 'put'
        )
        self.assertGreater(price, 0)

    def test_itm_call_gt_otm_call(self):
        itm = compute_heston_price(
            self.params['S_0'], 90.0, self.params['V_0'], self.params['r'],
            self.params['kappa'], self.params['theta'], self.params['sigma_v'],
            self.params['rho'], 1.0, 'call'
        )
        otm = compute_heston_price(
            self.params['S_0'], 110.0, self.params['V_0'], self.params['r'],
            self.params['kappa'], self.params['theta'], self.params['sigma_v'],
            self.params['rho'], 1.0, 'call'
        )
        self.assertGreater(itm, otm)


if __name__ == '__main__':
    unittest.main()
