# bagel-mean-variance

`bagel-mean-variance` is a Python package designed to calculate optimal portfolio weights using the mean-variance optimization method. The package is implemented using pure matrix operations, avoiding the use of optimization libraries. It is simple, efficient, and flexible, making it ideal for financial analysis and portfolio management tasks.

## Features

- **Mean-Variance Optimization**: Calculate optimal portfolio weights for a given target return.
- **Minimum Variance Portfolio**: Compute the weights for the minimum variance portfolio.
- **Class-Based Implementation**: Includes a `Portfolio` class for easy portfolio analysis, with properties like expected return, variance, volatility, and Sharpe ratio.
- **Support for Pandas DataFrames**: Works seamlessly with `pandas` for input and output.

## Installation

Install the package using pip:

```bash
pip install bagel-mean-variance
```

## Usage

### Functional API

The core functionality is exposed through the following functions:

- `optimal_weights(target_return: float, mu: np.ndarray, cov: np.ndarray) -> np.ndarray`: Calculate optimal portfolio weights for a given target return.
- `optimal_weights_using_returns(target_return: float, returns: np.ndarray) -> np.ndarray`: Calculate optimal portfolio weights directly from asset returns.
- `mvp_weights(cov_matrix: np.ndarray) -> np.ndarray`: Compute weights for the minimum variance portfolio using a covariance matrix.
- `mvp_weights_using_returns(returns: np.ndarray) -> np.ndarray`: Compute weights for the minimum variance portfolio directly from asset returns.

### Class-Based API

The package also provides a class-based interface for portfolio analysis:

#### Portfolio Class

```python
from bagel_mean_variance.class_implementation import Portfolio

portfolio = Portfolio(returns, weights)
print(portfolio.expected_return)
print(portfolio.volatility)
print(portfolio.sharpe_ratio)
```

#### EqualWeightPortfolio Class

```python
from bagel_mean_variance.class_implementation import EqualWeightPortfolio

eq_portfolio = EqualWeightPortfolio(returns)
print(eq_portfolio.weights)
```

#### Optimal Portfolio

```python
from bagel_mean_variance.class_implementation import optimal_portfolio

portfolio = optimal_portfolio(returns, target_return=0.02)
print(portfolio.weights)
```

#### Minimum Variance Portfolio

```python
from bagel_mean_variance.class_implementation import mvp_portfolio

portfolio = mvp_portfolio(returns)
print(portfolio.weights)
```

## Documentation

For detailed documentation, refer to the `docs/` folder in the repository.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
