import sys
from decouple import config
import pandas_datareader as pd_data 

from write_data import DataWritter

class DataGetter:
    def __init__(self, tiingo_api_key):
        self.tiingo_api_key = tiingo_api_key

    def download_training_data(self, ticker):
        """ get data from tiingo API, save it into a pandas dataframe """
        try:
            self.training_data = pd_data.get_data_tiingo(ticker, 
                                                         api_key=self.tiingo_api_key)
        except Exception:
            sys.exit("Data for ticker selected was not found, please try again with a different one.")


        # Make date the only index
        self.training_data.reset_index(level="symbol", inplace=True)
        self.training_data.drop(['symbol'], 1, inplace=True)
        return self.training_data


if __name__ == "__main__":
    data_query = DataGetter(config("TIINGO_API_KEY"))
    ticker = input("Enter the ticker of the company you want to use (example AAPL): ")
    training_data = data_query.download_training_data(ticker)
    training_data.reset_index(level="date", inplace=True)
    data_writter = DataWritter(training_data, "training_data")
    data_writter.write_data_to_csv()