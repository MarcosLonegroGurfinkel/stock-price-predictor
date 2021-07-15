import sys
import numpy as np 
from decouple import config
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

from write_data import DataWritter
from get_data import DataGetter

class NeuralNetworkModel:

    def __init__(self, training_data, past_n_days):
        self.training_data = training_data
        self.past_n_days = past_n_days

        self.__assert_valid_past_n_days()

        self.ninty_percent_len = int(0.9 * self.training_data.shape[0])  # 10% of data will be used for validation

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.nn_model = Sequential()

    def __assert_valid_past_n_days(self):
        if self.past_n_days <= 0 or self.past_n_days >= self.training_data.shape[0]:
            sys.exit(f"For this ticker, the number of days passed has to be between 1 and {self.training_data.shape[0] - 1}")

    def format_adjclose(self):
        # extract values from training data dataframe
        adjclose = self.training_data["adjClose"].values.reshape(-1, 1)

        # scale from -1 to 1 using the MinMaxScaler object
        scaled_adjclose = self.scaler.fit_transform(adjclose)
        return scaled_adjclose

    def separate_variables(self, scaled_adjclose):
        """Create x and y, train and test values from training data"""
        scaled_adjclose_for_train = scaled_adjclose[:self.ninty_percent_len]
        scaled_adjclose_for_test = scaled_adjclose[self.ninty_percent_len:]
        
        x_train = []
        y_train = []
        for i in range(self.past_n_days, len(scaled_adjclose_for_train)):
            x_train.append(scaled_adjclose_for_train[i - self.past_n_days:i, 0])
            y_train.append(scaled_adjclose_for_train[i, 0])

        x_test = []
        for i in range(self.past_n_days, len(scaled_adjclose_for_test)):
            x_test.append(scaled_adjclose_for_test[i - self.past_n_days: i, 0])
        
        y_test = [scaled_adjclose_for_test[self.past_n_days:, :]]

        x_test = np.array(x_test)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        # Reshape x data: number of samples, number of timesteps, number of features (as expected by LSTM model)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        return x_train, x_test, y_train, y_test
    
    def build_model(self, x_train, y_train):
        # build model layers
        self.nn_model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        self.nn_model.add(LSTM(50, return_sequences=False))
        self.nn_model.add(Dense(25))
        self.nn_model.add(Dense(1))

        # compile model
        self.nn_model.compile(optimizer="adam", loss="mean_squared_error")

        # train model
        self.nn_model.fit(x_train, y_train, batch_size=1, epochs=1)

    def validate_model(self, x_test, y_test):
        predictions = self.nn_model.predict(x_test)
        predictions = self.scaler.inverse_transform(predictions)
        rmse = np.sqrt(np.mean(((predictions - y_test[self.past_n_days:])**2)))
        print(f"Root mean squared error is {rmse}")
        return predictions

    def visualize_results(self, predictions):
        train_values = self.training_data["adjClose"].values.reshape(-1, 1)[:self.ninty_percent_len]
        
        data = self.training_data.filter(["adjClose"])
        train_values = data[:self.ninty_percent_len] 
        validation_values = data[self.ninty_percent_len + self.past_n_days:]  # FIXME: hay una parte que no estas testeando
        validation_values["predictions"] = predictions

        plt.figure()
        plt.title("Neural Network Model Validation")
        plt.xlabel("Date")
        plt.ylabel("ajdClose")
        plt.plot(train_values)
        plt.plot(validation_values[["adjClose", "predictions"]])
        plt.legend(["Train", "Validate", "Predictions"])
        plt.show()

    def save_validation_data_to_file(self, predictions):
        df = self.training_data[["adjClose"]][self.ninty_percent_len + self.past_n_days:]
        df["predictions"] = predictions
        df.reset_index(level="date", inplace=True)
        data_writter = DataWritter(df, "predictions")
        data_writter.write_data_to_csv()


if __name__ == "__main__":    
    
    ticker = input("Enter the ticker of the company you want to use (example AAPL): ")

    data_query = DataGetter(config("TIINGO_API_KEY"))
    training_data = data_query.download_training_data(ticker)

    days = int(input("Enter the number of days of data to be used to make a prediction (recommended 60): "))
    
    nn_model = NeuralNetworkModel(training_data, days)
    scaled_adjclose = nn_model.format_adjclose()
    x_train, x_test, y_train, y_test = nn_model.separate_variables(scaled_adjclose)
    nn_model.build_model(x_train, y_train)
    predictions = nn_model.validate_model(x_test, y_test)
    nn_model.visualize_results(predictions)
    nn_model.save_validation_data_to_file(predictions)