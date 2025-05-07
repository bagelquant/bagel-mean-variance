# Mean-Variance Optimization Core Module

This module provides the essential functions for conducting **mean-variance portfolio optimization**, including computing the **minimum variance portfolio (MVP)** and **optimal weights** for a target return based on asset return data.

All calculations are referenced from [Solution to the Mean-Variance Optimization Problem](https://bagelquant.com/mean-variance/solution-to-the-mean-variance-optimization-problem/) and are implemented in Python using NumPy.

## Public Interface

You can use the following functions externally:

### `optimal_weights(target_return: float, mu: np.ndarray, cov: np.ndarray) -> np.ndarray`

Computes optimal portfolio weights given a target return, expected returns (`mu`), and covariance matrix (`cov`).

### `optimal_weights_using_returns(target_return: float, returns: np.ndarray) -> np.ndarray`

Computes optimal weights using raw historical returns. Internally calculates `mu` and `cov`.

### `mvp_weights(cov_matrix: np.ndarray) -> np.ndarray`

Returns the weights of the **minimum variance portfolio** (MVP) using the covariance matrix directly.

### `mvp_weights_using_returns(returns: np.ndarray) -> np.ndarray`

Same as `mvp_weights`, but derives the covariance matrix from return data.

## Internal Utilities

These functions are used internally to compute intermediate quantities:

| Function | Description |
|----------|-------------|
| `_mu(returns)` | Mean return vector |
| `_cov(returns)` | Covariance matrix of returns |
| `_cov_inv(cov)` | Inverse of covariance matrix |
| `_A(mu, cov_inv)` | $A = \mu^\top \Sigma^{-1} \mathbf{1}$ |
| `_B(mu, cov_inv)` | $B = \mu^\top \Sigma^{-1} \mu$ |
| `_C(cov_inv)` | $C = \mathbf{1}^\top \Sigma^{-1} \mathbf{1}$ |
| `_D(A, B, C)` | $D = BC - A^2$ |
| `_g(A, B, D, mu, cov_inv)` | Vector used in target return computation |
| `_h(A, C, D, mu, cov_inv)` | Vector used in target return computation |
| `_optimal_weights(target_return, g, h)` | Final weights computed from `g` and `h` |

---

## Formulas Used

### Minimum Variance Portfolio (MVP) Weights

$$
\mathbf{w}_{\text{mvp}} = \frac{\Sigma^{-1} \mathbf{1}}{\mathbf{1}^\top \Sigma^{-1} \mathbf{1}}
$$

### Efficient Frontier Weights (Target Return $r$)

$$
\mathbf{w} = \mathbf{g} + r \cdot \mathbf{h}
$$

Where:

- $ \mathbf{g} = \frac{B \Sigma^{-1} \mathbf{1} - A \Sigma^{-1} \mu}{D} $
- $ \mathbf{h} = \frac{C \Sigma^{-1} \mu - A \Sigma^{-1} \mathbf{1}}{D} $
- $ A = \mu^\top \Sigma^{-1} \mathbf{1} $
- $ B = \mu^\top \Sigma^{-1} \mu $
- $ C = \mathbf{1}^\top \Sigma^{-1} \mathbf{1} $
- $ D = BC - A^2 $

## Testing

The `_test()` function simulates returns and validates:

- MVP weights
- Optimal weights from raw returns vs precalculated `mu`/`cov`
- Internal consistency across implementations

