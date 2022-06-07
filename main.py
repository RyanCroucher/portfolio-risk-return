import os
import implied_volatility
from flask import Flask

app = Flask(__name__)


@app.route("/")
def hello_world():
    name = os.environ.get("NAME", "World")
    return "Hello {}!".format(name)

@app.route("/implied_volatility/<platform_name>")
def retrieve_volatilities(platform_name):
    
    print("Connecting to BQ")
    bq = implied_volatility.get_bq_connection()
    
    print("Reading data and calculating")
    portfolio_to_volatility = implied_volatility.get_all_portfolio_volatilities(bq, platform_name)
    
    dataset = os.getenv("DATASET")
    destination_table = f"{dataset}.{platform_name}_implied_volatility"
    print("Writing data to BQ")
    #implied_volatility.write_portfolio_volatilities_to_bq(bq, portfolio_to_volatility, destination_table)

    volatilities = sorted(portfolio_to_volatility.values())
    return "Wrote {} portfolio volatilities to BQ. Min: {}, Max: {}, Median: {}".format(len(volatilities), volatilities[0], volatilities[-1], volatilities[len(volatilities)//2])


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))