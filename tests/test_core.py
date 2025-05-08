"""
Test core
"""

import unittest
import pandas as pd
from src.bagel_mean_variance.core import _mu, _cov, _cov_inv, _A, _B, _C, _D, _g, _h
from src.bagel_mean_variance.core import mvp_weights_using_returns, optimal_weights_using_returns, mvp_weights


class TestCore(unittest.TestCase):

    def setUp(self):
        test_returns = pd.read_csv("tests/test_stock_returns.csv", index_col=0)
        self.test_returns = test_returns.to_numpy()
        self.mu = _mu(self.test_returns)
        self.cov = _cov(self.test_returns)
        self.cov_inv = _cov_inv(self.cov)

    def test_A_B_C_D(self):
        print("\n===== Test A B C D g h=====")

        A = _A(self.mu, self.cov_inv)
        B = _B(self.mu, self.cov_inv)
        C = _C(self.cov_inv)
        D = _D(A, B, C)
        g = _g(A, B, D, self.mu, self.cov_inv)
        h = _h(A, C, D, self.mu, self.cov_inv)

        print(f"A: {A}")
        print(f"B: {B}")
        print(f"C: {C}")
        print(f"D: {D}")
        print(f"g: {g}")
        print(f"h: {h}")

    def test_mvp(self):
        print("\n===== Test MVP =====")
        mvp = mvp_weights_using_returns(self.test_returns)
        mvp = mvp_weights(self.cov)

        variance = mvp.T @ self.cov @ mvp
        expected_return = mvp.T @ self.mu

        print(f"MVP variance: {variance}")
        C = _C(self.cov_inv)
        print(f"1/C: {1/C}")
        print(f"MVP expected return: {expected_return}")

        target_return = expected_return
        print("\n===== Test Optimal Weights =====")
        optimal_weights = optimal_weights_using_returns(float(target_return), self.test_returns)
        new_variance = optimal_weights.T @ self.cov @ optimal_weights
        new_expected_return = optimal_weights.T @ self.mu
        print(f"Optimal Weights variance: {new_variance}")
        print(f"Optimal Weights expected return: {new_expected_return}")



if __name__ == "__main__":
    unittest.main()

        

