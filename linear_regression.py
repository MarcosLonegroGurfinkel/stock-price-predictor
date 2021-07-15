import sys
from decouple import config
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from get_data import DataGetter
from write_data import DataWritter

class LinearRegressionModel:
    def __init__(self, training_data, forecast_out):
        self.training_data = training_data
        self.forecast_out = forecast_out

        self.__assert_valid_forecast_out()

        self.regression = LinearRegression()

    def __assert_valid_forecast_out(self):
        if self.forecast_out <= 0 or self.forecast_out >= self.training_data.shape[0]:
            sys.exit(f"For this ticker, the number of days passed has to be between 1 and {self.training_data.shape[0] - 1}")

    def separate_variables(self):
        """Create x and y, train and test values from training data"""
        # create target variable
        self.training_data["prediction"] = self.training_data[["adjClose"]].shift(-self.forecast_out)

        # set x
        x = np.array(self.training_data.drop(['prediction'], 1))
        x = x[:-self.forecast_out]

        # set y
        y = np.array(self.training_data['prediction'])
        y = y[:-self.forecast_out]

        # train test split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

        return x_train, x_test, y_train, y_test

    def build_model(self, x_train, y_train):
        self.regression.fit(x_train, y_train)
        return self.regression

    def validate_model(self, x_test, y_test):
        score = self.regression.score(x_test, y_test)
        print(f"Model score in cross-validation was {score}")
        return score

    def predict_with_model(self):
        # predict next n days of prices and print them
        x_for_prediction = np.array(self.training_data.drop(["prediction"], 1))[-self.forecast_out:]
        prediction = self.regression.predict(x_for_prediction)
        return prediction
    
    def save_prediction_to_file(self, prediction):
        dates = [self.training_data.index[-1] + timedelta(days=i + 1) for i in range(self.forecast_out)]
        df = pd.DataFrame({"dates": dates, "adj Close prediction": prediction})
        data_writer = DataWritter(df, "predictions")
        data_writer.write_data_to_csv()


if __name__ == "__main__":
    data_query = DataGetter(config("TIINGO_API_KEY"))
    ticker = input("Enter the ticker of the company you want to use (example AAPL): ")
    training_data = data_query.download_training_data(ticker)

    n_days = int(input("Enter the amount of days forward that you want to forecast for: "))
    linear_regression = LinearRegressionModel(training_data, n_days)
    x_train, x_test, y_train, y_test = linear_regression.separate_variables()
    linear_regression.build_model(x_train, y_train)
    linear_regression.validate_model(x_test, y_test)
    prediction = linear_regression.predict_with_model()
    linear_regression.save_prediction_to_file(prediction)
