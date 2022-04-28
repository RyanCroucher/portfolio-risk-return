import numpy as np

def covariance(asset_a_returns, asset_b_returns):
    """
    Inputs: Two iterables of daily returns for two assets/stocks
    Lists should be the same length (same window size)

    Calculated as:
    The sum of ((each a_i return minus average a return) * (each b_i return minus average b return))
    over the sample size - 1

    Output: the covariance of the assets as a float
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

    return np.cov(asset_matrix)[0, 1] #anything off the main diagonal (which would be self-variance)


def main():
    a_returns = [0, 1, 2]
    b_returns = [2, 1, 0]
    print(covariance(a_returns, b_returns))

if __name__ == '__main__':
    main()