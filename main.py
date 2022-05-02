import numpy as np
from itertools import combinations
from unittest import *
import random
import time

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

    if len(portfolio_assets) == 0:
        raise ValueError(f"Length of input must be at least 1, not {len(portfolio_assets)}")

    if len(portfolio_assets) != len(set(portfolio_assets)):
        raise ValueError(f"Input contains duplicate asset names")

    pairs = set()

    #includes self, i.e. (asset_a, asset_a) as we need the variance too
    for asset_a_i in range(len(portfolio_assets)):
        for asset_b_i in range(asset_a_i, len(portfolio_assets)):
            asset_a = portfolio_assets[asset_a_i]
            asset_b = portfolio_assets[asset_b_i]
            pairs.add(tuple(sorted([asset_a, asset_b]))) #sorted tuple

    return pairs
    #return np.array(list(combinations(portfolio_assets, 2)))


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
    Given a vector of the weights of each asset (in the portfolio) and a covariance matrix,
    we can calculate the overall variance of the portfolio which is a metric for the risk of the portfolio
    Note that volatility/stdev is the square root of this variance

    Inputs:
        weights: An iterable representing a vector
            weights must add to 1 and be shape (,n)
        covariances: a matrix representing covariance between every asset pair
            must be shape (n, n)
        e.g.:
            weights = [
                        0.1     #a1
                        0.6     #a2
                        0.3     #a3
                        ]
            covariances = 
                            #a1   #a2   #a3
                        [
                    #a1      0.11, 0.3, -0.2
                    #a2      -0.5, 1,  0.1
                    #a3      0.1,  0.22, 0.37
                        ]

    Output:
        The overall portfolio variance
        calculated by W^T * C_mat * W
    """

    if len(weights) == 0 or covariance_matrix.shape != (len(weights), len(weights)):
        raise ValueError(f"Matrix shape doesn't match given number of weights")

    delta = 0.01
    weight_sum = np.sum(weights)
    if abs(weight_sum - 1) > delta:
        raise ValueError(f"Weights must add to 1, not {weight_sum}")

    return (weights.T @ covariance_matrix @ weights)[0, 0] #The single element of this 'matrix'

def portfolios_risk(portfolios):
    """
    input:
    An iterable containing Portfolios
    A portfolio is a dictionary of assets where
    An asset maps asset_name: (weight, returns)

    We calculate all occuring covariances, then use asset weights + covariances
    to calculate the risk of each portfolio
    """

    all_assets_name_to_returns = {}
    for portfolio in portfolios:
        for asset_name, asset_data in portfolio.items():
            _, asset_returns = asset_data
            all_assets_name_to_returns[asset_name] = asset_returns

    occuring_pairs = set()

    for portfolio in portfolios:
        occuring_pairs |= portfolio_asset_pairs(list(portfolio.keys()))

    all_covariances = {(a, b): covariance(all_assets_name_to_returns[a], all_assets_name_to_returns[b]) for a, b in occuring_pairs}

    portfolio_risks = []

    #create weight vector and covariance matrix for each portfolio
    for portfolio in portfolios:
        sorted_asset_names = sorted(portfolio.keys())
        weight_vector = np.empty((len(sorted_asset_names), 1))
        cov_matrix = np.empty((len(sorted_asset_names), len(sorted_asset_names)))
        for i, name in enumerate(sorted_asset_names):
            weight_vector[i, 0] = portfolio[name][0]
            for j in range(len(sorted_asset_names)):
                asset_a = sorted_asset_names[i]
                asset_b = sorted_asset_names[j]
                if asset_a > asset_b:
                    asset_a, asset_b = asset_b, asset_a #swap them to keep alphabetical order
                cov_matrix[i, j] = all_covariances[(asset_a, asset_b)]

        #print(weight_vector)
        #print(cov_matrix)
        #print("Volatility: ", portfolio_variance(weight_vector, cov_matrix) ** 0.5)
        portfolio_risks.append(portfolio_variance(weight_vector, cov_matrix) ** 0.5)

    return portfolio_risks

    
def test_covariance():
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

def test_asset_pairs():
    print(portfolio_asset_pairs(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']))

def main():
    portfolio_a = {}
    portfolio_a['asset_a'] = (0.1, [1, 3, 4, 1, -6, -1, 1, -1, 2])
    portfolio_a['asset_b'] = (0.6, [2, 2, 3, 0, 0, -1, -1, 2, 2])
    portfolio_a['asset_c'] = (0.3, [-1, -1, 1, 1, 0, 1, -1, 1, 2])

    portfolio_b = {}
    portfolio_b['asset_d'] = (0.5, [1, 2, 3, 4, 5, -1, -2, -3])
    portfolio_b['asset_e'] = (0.5, [-1, -2, -3, -4, -5, 1, 2, 3])

    portfolio_c = {}
    portfolio_c['asset_d'] = (0.5, [1, 2, 3, 4, 5, -1, -2, -3])
    portfolio_c['asset_f'] = (0.5, [2, 3, 6, 7, 4, 0, 2, -1])

    print(portfolios_risk([portfolio_a, portfolio_b, portfolio_c]))


    num_days = 365
    num_assets = 100
    num_portfolios = 10**6
    max_assets_per_portfolio = 10

    print(f"Generating {num_portfolios=}, {num_days=}, {num_assets=}, {max_assets_per_portfolio=}")

    assets = [[] for _ in range(num_assets)]
    for i in range(num_assets):
        for _ in range(num_days):
            assets[i].append(random.randint(-10, 10))

    portfolios = [{} for _ in range(num_portfolios)]
    for portfolio_i in range(num_portfolios):
        seed = random.randint(1, 1000)
        num_portfolio_assets = random.randint(1, max_assets_per_portfolio)
        weights = np.random.dirichlet(np.ones(num_portfolio_assets)*seed, size=1)[0]
        portfolio_assets = random.choices(list(range(num_assets)), k=num_portfolio_assets)
        for asset_i in range(len(portfolio_assets)):
            portfolios[portfolio_i][asset_i] = (weights[asset_i], assets[portfolio_assets[asset_i]])

    print(f"Calculating risk...")
    start_time = time.time()
    portfolios_risk(portfolios)
    print(f"Done! ({time.time() - start_time:.1f} seconds)")




if __name__ == '__main__':
    main()