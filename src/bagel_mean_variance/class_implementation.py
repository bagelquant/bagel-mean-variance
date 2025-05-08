"""
A Class implementation of the functions in core.py
"""

import pandas as pd
from dataclasses import dataclass, field
from .core import mvp_weights_using_returns, optimal_weights_using_returns


@dataclass(slots=True)
class Portfolio:
    
    returns: pd.DataFrame
    weights: pd.Series

    @property
    def assets(self):
        return self.returns.columns

    @property
    def mean_returns(self):
        return self.returns.mean()

    @property
    def covariance_matrix(self):
        return self.returns.cov()

    @property
    def expected_return(self):
        return self.mean_returns @ self.weights

    @property
    def variance(self):
        return self.weights.T @ self.covariance_matrix @ self.weights

    @property
    def volatility(self):
        return self.variance ** 0.5

    @property
    def sharpe_ratio(self, rf: float = 0.0):
        return (self.expected_return - rf) / self.volatility

    @property
    def accumulated_returns_asset(self):
        return (self.returns + 1).cumprod() - 1

    @property
    def accumulated_returns_portfolio(self):
        return (self.accumulated_returns_asset + 1) @ self.weights - 1



@dataclass(slots=True)
class EqualWeightPortfolio(Portfolio):

    returns: pd.DataFrame
    weights: pd.Series = field(init=False)

    def __post_init__(self):
        self.weights = pd.Series(1 / len(self.returns.columns), index=self.returns.columns)



def optimal_portfolio(returns: pd.DataFrame, target_return: float) -> Portfolio:
    """
    Calculate the optimal portfolio for a given target return.
    
    :param returns (np.ndarray): The returns of the assets.
    :param target_return (float): The target return.
    :return: The optimal portfolio.
    """
    weights = optimal_weights_using_returns(target_return, returns.values)
    weights = pd.Series(weights, index=returns.columns)
    return Portfolio(returns, weights)


def mvp_portfolio(returns: pd.DataFrame) -> Portfolio:
    """
    Calculate the minimum variance portfolio.
    
    :param returns (np.ndarray): The returns of the assets.
    :return: The minimum variance portfolio.
    """
    weights = mvp_weights_using_returns(returns.values)
    weights = pd.Series(weights, index=returns.columns)
    return Portfolio(returns, weights)

