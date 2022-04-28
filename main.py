import numpy as np
from itertools import combinations
from unittest import *

def portfolio_asset_pairs(portfolio_assets) -> np.array:
    """
    Inputs: 
        An iterable of distinct assets/stocks

    Output:
        A numpy matrix of all unordered pairs in the portfolio

        e.g.:
        given portfolio_assets = [A, B, C, X]
        we could return np.array([[A, B], [A, C], [A, X], [B, C], [B, X], [C, X]])
    """

    try:
        iterator = iter(portfolio_assets)
    except:
        raise TypeError(f"Input should be an iterable, not a {type(portfolio_assets)}")

    if len(portfolio_assets) < 2:
        raise ValueError(f"Length of input must be at least 2, not {len(portfolio_assets)}")

    if len(portfolio_assets) != len(set(portfolio_assets)):
        raise ValueError(f"Input contains duplicate asset names")

    return np.array(list(combinations(portfolio_assets, 2)))


def covariance(asset_a_returns, asset_b_returns) -> float:
    """
    Inputs: 
        Two iterables of daily returns for two assets/stocks
        Lists should be the same length (same timeframe)

    Output: 
        The covariance of those assets in that timeframe
    """

    try:
        iterator = iter(asset_a_returns)
        iterator = iter(asset_b_returns)
    except:
        raise TypeError(f"Inputs should be iterables, not {type(asset_a_returns)} and {type(asset_b_returns)}")

    n, m = len(asset_a_returns), len(asset_b_returns)

    if n != m or n <= 1 or m <= 1:
        raise ValueError(f"Length of asset returns must be equal and >1, not {n} and {m}")


    asset_matrix = np.array([asset_a_returns, asset_b_returns])

    return np.cov(asset_matrix)[0, 1] #anything off the main diagonal (which would be own-variance)

def variance(asset_returns) -> float:
    """
    A special case of covariance

    Inputs:
        An iterable of daily returns of an asset in some timeframe
    Output:
        The variance of that asset in that timeframe
    """

    return covariance(asset_returns, asset_returns)

def portfolio_variance(weights: np.array, covariance_matrix: np.array) -> float:
    """
    Given a portfolio of weights (a vector) for each asset and a covariance matrix,
    we can calculate the overall variance of the portfolio which is a metric for the risk of the portfolio

    Inputs:
        weights: An iterable representing a vector
            weights must add to 1 and be shape (,n)
        covariances: a matrix representing covariance between every asset pair
            must be shape (n, n)
        e.g.:
            weights = [
                        0.1
                        0.6
                        0.3
                        ]
            covariances = 
                        [
                            0.11, 0.3, -0.2
                            -0.5, 1,  0.1
                            0.1,  0.22, 0.37
                        ]

    Output:
        The overall portfolio variance
        calculated by Wtranspose * C_mat * W
    """

    if len(weights) == 0 or covariance_matrix.shape != (len(weights), len(weights)):
        raise ValueError(f"Matrix shape doesn't match given number of weights")

    delta = 0.01
    weight_sum = np.sum(weights.values())
    if abs(weight_sum - 1) > delta:
        raise ValueError(f"Weights must add to 1, not {weight_sum}")

    return weights.T @ covariance_matrix @ weights



def main():
    test = TestCase()
    a_returns = [0, 1, 2]
    b_returns = [2, 1, 0]
    test.assertAlmostEqual(covariance(a_returns, b_returns), -1)

    a_returns = [1.1, 1.7, 2.1, 1.4, 0.2]
    b_returns = [3.0, 4.2, 4.9, 4.1, 2.5]
    test.assertAlmostEqual(covariance(a_returns, b_returns), 0.665)

    a_returns = [1, 1, 1, 1, 1]
    b_returns = [1, 1, 1, 1, 1]
    test.assertAlmostEqual(covariance(a_returns, b_returns), 0)

    print(portfolio_asset_pairs(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']))

if __name__ == '__main__':
    main()