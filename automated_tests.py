import threading
import concurrent.futures
import os
from decouple import config

from linear_regression import LinearRegressionModel
from neural_network import NeuralNetworkModel
from get_data import DataGetter
from write_data import DataWritter


# Test cases:
def check_linear_regression():
    """Test linear regression happy path"""
    ticker = "FB"
    forecast_out = 5

    data_query = DataGetter(config("TIINGO_API_KEY"))
    training_data = data_query.download_training_data(ticker)

    try:
        linear_regression = LinearRegressionModel(training_data, forecast_out)
        x_train, x_test, y_train, y_test = linear_regression.separate_variables()
        linear_regression.build_model(x_train, y_train)
        linear_regression.validate_model(x_test, y_test)
        linear_regression.predict_with_model()
    except Exception as e:
        raise TestException(f"Error in linear regression model: {e}")


def check_neural_network():
    """Test neural network happy path"""
    ticker = "AAPL"
    past_n_days = 60

    data_query = DataGetter(config("TIINGO_API_KEY"))
    training_data = data_query.download_training_data(ticker)

    try:
        nn_model = NeuralNetworkModel(training_data, past_n_days)
        scaled_adjclose = nn_model.format_adjclose()
        x_train, x_test, y_train, y_test = nn_model.separate_variables(scaled_adjclose)
        nn_model.build_model(x_train, y_train)
        nn_model.validate_model(x_test, y_test)
    except Exception as e:
        raise TestException(f"Error in neural network model: {e}")


def check_get_data():
    """Test get data happy path"""

    ticker = "GOOGL"
    try:
        data_query = DataGetter(config("TIINGO_API_KEY"))
        data_query.download_training_data(ticker)
    except Exception as e:
        raise TestException(f"Error when getting data from tiingo api: {e}")


def check_write_data():
    """Test write data happy path"""
    ticker = "GOOGL"
    data_query = DataGetter(config("TIINGO_API_KEY"))
    training_data = data_query.download_training_data(ticker)
    training_data.reset_index(level="date", inplace=True)

    try:
        data_writter = DataWritter(training_data, "training_data")
        data_writter.write_data_to_csv(save_to_csv="yes", csv_file_name="automated_test.csv")
        assert os.path.isfile("automated_test.csv")
    except Exception as e:
        raise TestException(f"Error when writing data: {e}")

    os.remove("automated_test.csv")


# Testing framework:
class TestException(Exception):
    def __init__(self, message):
        super().__init__(message)

class TestCase:
    def __init__(self, test_function):
        self.test_function = test_function

class TestSuite:
    def __init__(self):
        self.all_tests = []
        self.correct_tests = []
        self.failed_tests = []

    def add(self, test: TestCase):
        self.all_tests.append(test)
    
    def run(self):
        """Runs all test cases in parallel"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.all_tests)) as executor:
            futures = []
            for test_case in self.all_tests:
                future = executor.submit(self.__run_test_in_thread,
                                         test_case)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                print(future.result())

    def __run_test_in_thread(self, test_case):
        """Runs a specific test case in a single thread"""
        func = test_case.test_function
        threading.currentThread().setName(f"TestThread_{func.__name__}")
        try:
            print(f"Starting test {func.__name__}")
            func()
            self.correct_tests.append(test_case)
        except TestException as e:
            self.failed_tests.append(test_case)
            print(f"Test {func.__name__} Failed")
            print(e)
        return f"Test Ended: {func.__name__}"

    def report(self):
        """Reports testsuite results after it has finished running"""
        if self.failed_tests:
            print("SOME TESTS FAILED: ")
            for test in self.failed_tests:
                print("--> ", test.test_function.__name__)
        else:
            print("ALL TESTS PASSED")


if __name__ == "__main__":
    print("Running automated tests")

    test_suite = TestSuite()
    test_suite.add(TestCase(check_linear_regression))
    test_suite.add(TestCase(check_neural_network))
    test_suite.add(TestCase(check_get_data))
    test_suite.add(TestCase(check_write_data))

    test_suite.run()
    test_suite.report()