def covariance(asset_a_returns, asset_b_returns):
    """
    Inputs: Two lists of daily returns for two assets/stocks
    Lists should be the same length (same window size)

    Calculated as:
    The sum of ((each a_i return minus average a return) * (each b_i return minus average b return))
    over the sample size - 1

    Output: the covariance of the assets as a float
    """

    if len(asset_a_returns) != len(asset_b_returns) or len(asset_a_returns) == 0 or len(asset_b_returns) == 0:
        raise ValueError(f"Length of input lists must be equal and non-zero, not {len(asset_a_returns)} and {len(asset_b_returns)}")

def main():
    print("Hello")

if __name__ == '__main__':
    main()