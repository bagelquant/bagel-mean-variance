"""
Test for class implementation

"""

import unittest
import pandas as pd
import numpy as np

from src.bagel_mean_variance import EqualWeightPortfolio, optimal_portfolio, mvp_portfolio


class TestPortfolio(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        data = {
            'Asset1': [0.01, 0.02, 0.015, 0.03, -0.025, 0.035, 0.04, 0.045, 0.25, 0.055, 0.06, 0.065],
            'Asset2': [0.02, 0.025, 0.03, 0.035, 0.04, -0.025, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075],
            'Asset3': [0.015, 0.02, 0.025, 0.03, 0.035, -0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07]
        }
        self.returns = pd.DataFrame(data)
        self.target_return = 0.025
    
    def test_eqw_portfolio(self):
        print("\n===== EqualWeightPortfolio =====")
        # Test EqualWeightPortfolio
        eqw_portfolio = EqualWeightPortfolio(self.returns)
        
        # Check if weights are equal
        expected_weights = pd.Series([1/3] * 3, index=self.returns.columns)
        pd.testing.assert_series_equal(eqw_portfolio.weights, expected_weights, check_exact=True)
        
        print("EqualWeightPortfolio Weights:\n", eqw_portfolio.weights)
        print("EqualWeightPortfolio Expected Return:", eqw_portfolio.expected_return)
        print("EqualWeightPortfolio Variance:", eqw_portfolio.variance)

    def test_optimal_portfolio(self):
        print("\n===== OptimalPortfolio =====")
        # Test optimal_portfolio
        optimal_port = optimal_portfolio(self.returns, self.target_return)
        
        # Check if the expected return is close to the target return
        self.assertAlmostEqual(optimal_port.expected_return, self.target_return, places=4)  # type: ignore
        
        print("Optimal Portfolio Weights:\n", optimal_port.weights)
        print("Optimal Portfolio Expected Return:", optimal_port.expected_return)
        print("Optimal Portfolio Variance:", optimal_port.variance)

    def test_mvp_portfolio(self):
        print("\n===== MVP Portfolio =====")
        # Test mvp_portfolio
        mvp_port = mvp_portfolio(self.returns)
        
        print("MVP Portfolio Weights:\n", mvp_port.weights)
        print("MVP Portfolio Expected Return:", mvp_port.expected_return)
        print("MVP Portfolio Variance:", mvp_port.variance)


if __name__ == "__main__":
    unittest.main()














