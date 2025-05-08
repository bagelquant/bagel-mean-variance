"""
The core module to handle the calculation of the mean-variance optimization.

Outside use:

    - optimal_weights(target_return: float, mu: np.ndarray, cov: np.ndarray) -> np.ndarray
    - optimal_weights_using_returns(target_return: float, returns: np.ndarray) -> np.ndarray
    - mvp_weights(cov_matrix: np.ndarray) -> np.ndarray
    - mvp_weights_using_returns(returns: np.ndarray) -> np.ndarray
"""


import numpy as np


def _mu(returns: np.ndarray) -> np.ndarray:
    return np.mean(returns, axis=0)


def _cov(returns: np.ndarray) -> np.ndarray:
    return np.cov(returns, rowvar=False)


def _cov_inv(cov_matrix: np.ndarray) -> np.ndarray:
    return np.linalg.inv(cov_matrix)


def _A(mu: np.ndarray, cov_inv: np.ndarray) -> float:
    """
    Calculate the A = mu^T * cov_inv * ones.

    :param mu (np.ndarray): The mean of the returns.
    :param cov_inv (np.ndarray): The inverse of the covariance matrix.
    :return: The A value.
    """
    return np.dot(mu, cov_inv @ np.ones(mu.shape[0]))


def _B(mu: np.ndarray, cov_inv: np.ndarray) -> float:
    """
    Calculate the B = mu^T * cov_inv * mu.

    :param cov_inv (np.ndarray): The inverse of the covariance matrix.
    :return: The B value.
    """
    return np.dot(mu, cov_inv @ mu)


def _C(cov_inv: np.ndarray) -> float:
    """
    Calculate the C = ones^T * cov_inv * ones.

    :param cov_inv (np.ndarray): The inverse of the covariance matrix.
    :return: The C value.
    """
    return np.dot(np.ones(cov_inv.shape[0]), cov_inv @ np.ones(cov_inv.shape[0]))


def _D(A: float, B: float, C: float) -> float:
    """
    Calculate the D = A * C - B^2.

    :param A (float): The A value = mu^T * cov_inv * ones
    :param B (float): The B value = mu^T * cov_inv * mu
    :param C (float): The C value = ones^T * cov_inv * ones
    :return: The D value.
    """
    return B * C - A**2


def _g(A: float, 
       B: float, 
       D: float,
       mu: np.ndarray,
       cov_inv: np.ndarray) -> np.ndarray:
    """
    Calculate the g (N*1) = 1/d * (B * cov_inv * ones - A * cov_inv * mu).

    :param A (float): The A value = mu^T * cov_inv * ones
    :param B (float): The B value = mu^T * cov_inv * mu
    :param C (float): The C value = ones^T * cov_inv * ones
    :param D (float): The D value = A * C - B^2
    :param mu (np.ndarray): The mean of the returns.
    :param cov_inv (np.ndarray): The inverse of the covariance matrix.
    :return: The g value.
    """
    return (B * cov_inv @ np.ones(mu.shape[0]) - A * cov_inv @ mu) / D


def _h(A: float, 
       C: float, 
       D: float,
       mu: np.ndarray,
       cov_inv: np.ndarray) -> np.ndarray:
    """
    Calculate the h (N*1) = 1/d * (C * cov_inv * mu - A * cov_inv * ones).

    :param A (float): The A value = mu^T * cov_inv * ones
    :param B (float): The B value = mu^T * cov_inv * mu
    :param C (float): The C value = ones^T * cov_inv * ones
    :param D (float): The D value = A * C - B^2
    :param mu (np.ndarray): The mean of the returns.
    :param cov_inv (np.ndarray): The inverse of the covariance matrix.
    :return: The h value.
    """
    return (C * cov_inv @ mu - A * cov_inv @ np.ones(mu.shape[0])) / D


def _optimal_weights(target_return: float,
                     g: np.ndarray,
                     h: np.ndarray) -> np.ndarray:
    """
    Calculate the weight of the assets.
    The weight is calculated by the formula: w = g + (target_return - h) * g / (g^T * g)
    The weight is a N*1 vector.
    :param target_return (float): The target return.
    :param g (np.ndarray): The g value.
    :param h (np.ndarray): The h value.
    :return: The weight of the assets.
    """
    return g + h * target_return


def optimal_weights(target_return: float, 
                    mu: np.ndarray,
                    cov: np.ndarray) -> np.ndarray:
    """
    Calculate the weight of the assets for the minimum variance portfolio.

    The weight is a N*1 vector.
    :param mu (np.ndarray): The mean of the returns.
    :param cov (np.ndarray): The covariance matrix of the returns.
    :return: The weight of the assets.
    """
    cov_inv = _cov_inv(cov)
    A = _A(mu, cov_inv)
    B = _B(mu, cov_inv)
    C = _C(cov_inv)
    D = _D(A, B, C)
    g = _g(A, B, D, mu, cov_inv)
    h = _h(A, C, D, mu, cov_inv)
    return _optimal_weights(target_return, g, h)


def optimal_weights_using_returns(target_return: float,
                                  returns: np.ndarray) -> np.ndarray:
    """
    Calculate the weight of the assets for the minimum variance portfolio.

    The weight is calculated by the formula: w = g + (target_return - h) * g / (g^T * g)
    The weight is a N*1 vector.
    :param target_return (float): The target return.
    :param returns (np.ndarray): The returns of the assets.
    :return: The weight of the assets.
    """
    mu = _mu(returns)
    cov = _cov(returns)
    return optimal_weights(target_return, mu, cov)


def mvp_weights(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the weight of the assets for the minimum variance portfolio.
    The weight is calculated by the formula: w = cov_inv * ones / (ones^T * cov_inv * ones)
    The weight is a N*1 vector.
    :param cov_matrix (np.ndarray): The covariance matrix of the returns.
    :return: The weight of the assets.
    """
    cov_inv = _cov_inv(cov_matrix)
    C = _C(cov_inv)
    return cov_inv @ np.ones(cov_inv.shape[0]) / C


def mvp_weights_using_returns(returns: np.ndarray) -> np.ndarray:
    """
    Calculate the weight of the assets for the minimum variance portfolio.
    The weight is calculated by the formula: w = cov_inv * ones / (ones^T * cov_inv * ones)
    The weight is a N*1 vector.
    :param returns (np.ndarray): The returns of the assets.
    :return: The weight of the assets.
    """
    cov = _cov(returns)
    return mvp_weights(cov)

