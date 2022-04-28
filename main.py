import numpy as np
from unittest import *

def covariance(asset_a_returns, asset_b_returns) -> float:
    """
    Inputs: Two iterables of daily returns for two assets/stocks
    Lists should be the same length (same window size)

    Output: the covariance of the two assets as a float
    """

    try:
        iterator = iter(asset_a_returns)
        iterator = iter(asset_b_returns)
    except:
        raise TypeError(f"Inputs should be iterables, not {type(asset_a_returns)} and {type(asset_b_returns)}")

    n, m = len(asset_a_returns), len(asset_b_returns)

    if n != m or n <= 1 or m <= 1:
        raise ValueError(f"Length of input lists must be equal and >1, not {n} and {m}")


    asset_matrix = np.array([asset_a_returns, asset_b_returns])

    return np.cov(asset_matrix)[0, 1] #anything off the main diagonal (which would be own-variance)


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

if __name__ == '__main__':
    main()