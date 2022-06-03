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
    portfolio_to_volatility = implied_volatility.get_all_portfolio_volatilities(platform_name)
    volatilities = sorted(portfolio_to_volatility.values())
    return "Min: {}, Max: {}, Median: {}".format(volatilities[0], volatilities[-1], volatilities[len(volatilities)//2])


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))