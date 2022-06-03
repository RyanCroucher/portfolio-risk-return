from google.cloud import bigquery
import numpy as np
from dotenv import load_dotenv
import pandas as pd
import os

class Portfolio:
    def __init__(self, portfolio_id):
        self.portfolio_id = portfolio_id
        self.assets = {}
        self.total_value = 0

    def add_asset(self, asset):
        asset_id = asset.asset_id
        self.assets[asset_id] = asset
        asset_value = asset.asset_value
        self.total_value += asset_value

    def get_asset_weight(self, asset_id):
        asset = self.assets[asset_id]
        return asset.asset_value / self.total_value

    def get_sorted_asset_names(self):
        sorted_asset_names = sorted(self.assets.keys())
        return sorted_asset_names

    def get_weight_vector(self):
        sorted_asset_names = self.get_sorted_asset_names()
        weight_vector = np.empty((len(sorted_asset_names), 1))
        for i, asset_name in enumerate(sorted_asset_names):
            weight_vector[i, 0] = self.get_asset_weight(asset_name)

        return weight_vector

class Asset:
    def __init__(self, id, value):
        self.asset_id = id
        self.asset_value = value


def build_portfolio_dict_from_df(account_instrument_df):

    #build portfolios from aua data
    all_portfolios = {}
    for _, row in account_instrument_df.iterrows():
        portfolio_id = row['ClAccountID']
        asset_id = row['InstrumentCode']
        asset_value = row['Value']

        if portfolio_id not in all_portfolios:
            all_portfolios[portfolio_id] = Portfolio(portfolio_id)
        
        all_portfolios[portfolio_id].add_asset(Asset(asset_id, asset_value))

    return all_portfolios

def build_covariance_dict_from_df(intrument_pair_cov_df):
    covariance_dict = {}
    for _, row in intrument_pair_cov_df.iterrows():
        instrument_a = row['instrument_a']
        instrument_b = row['instrument_b']
        covariance = row['covariance_1y']

        covariance_dict[(instrument_a, instrument_b)] = covariance

    return covariance_dict

def calculate_portfolio_volatility(portfolio, covariance_dict):

    portfolio_asset_names = portfolio.get_sorted_asset_names()

    covariance_matrix = np.empty((len(portfolio_asset_names), len(portfolio_asset_names)))
    weight_vector = portfolio.get_weight_vector()

    for i, asset_a in enumerate(portfolio_asset_names):
        for j, asset_b in enumerate(portfolio_asset_names):
            try:
                covariance_matrix[i, j] = covariance_dict[(asset_a, asset_b)]
            except KeyError:
                #print(f"({asset_a}, {asset_b}) not in covariance dictionary, setting covariance to 1 if equal else 0")
                covariance_matrix[i, j] = 1 if asset_a == asset_b else 0

    portfolio_variance = (weight_vector.T @ covariance_matrix @ weight_vector)[0, 0] #the only element of the result 'matrix'
    portfolio_stdev = np.sqrt(portfolio_variance)
    return portfolio_stdev

def get_platform_data(bq, platform):
    platforms_file = open("platform_names.txt")
    valid_platform_names = set(platforms_file.read().splitlines())
    platforms_file.close()

    if platform not in valid_platform_names:
        raise ValueError("Invalid platform name")

    aua_query_base = os.getenv('AUA_QUERY') # ~200mb full data
    aua_query = aua_query_base.replace("{platform_name}", platform)
    acc_inst_val_df = bq.query(aua_query, location='EU').to_dataframe()
    portfolios_dict = build_portfolio_dict_from_df(acc_inst_val_df)

    covariance_query_base = os.getenv('COVARIANCE_QUERY') # ~25mb full data
    covariance_query = covariance_query_base.replace("{platform_name}", platform)
    inst_pair_cov_df = bq.query(covariance_query, location='EU').to_dataframe()
    covariance_dict = build_covariance_dict_from_df(inst_pair_cov_df)

    return portfolios_dict, covariance_dict

def get_all_portfolio_volatilities(bq, platform):

    portfolios_dict, covariance_dict = get_platform_data(bq, platform)

    portfolio_to_volatility = {}
    for portfolio_name, portfolio in portfolios_dict.items():
        portfolio_to_volatility[portfolio_name] = calculate_portfolio_volatility(portfolio, covariance_dict)

    print("Calculated implied volatility for", len(portfolio_to_volatility), "portfolios.")

    return portfolio_to_volatility


def write_portfolio_volatilities_to_bq(bq, portfolio_to_volatility, destination_table):

    volatilities_df = pd.DataFrame(portfolio_to_volatility.items(), columns=['ClAccountID', 'volatility_1yr'])

    # Since string columns use the "object" dtype, pass in a (partial) schema
    job_config = bigquery.LoadJobConfig(
        schema=[bigquery.SchemaField("ClAccountID", "STRING")],
        write_disposition="WRITE_TRUNCATE")

    job = bq.load_table_from_dataframe(
        volatilities_df, destination_table, job_config=job_config
    )

    # Wait for the load job to complete.
    job.result()

def get_bq_connection():

    load_dotenv()
    PROJECT = os.getenv('PROJECT')
    bq = bigquery.Client(project=PROJECT)

    return bq