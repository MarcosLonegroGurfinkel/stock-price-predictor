I am not a financial advisor. Please conduct your own research before making investment decisions.

# Stock Price Predictor

This program attempts to train a machine learning model using past stock price data to predict future prices.

It provides the option of training a simple linear regression model, or a more complicated neural network model.

At runtime, you can also choose which specific company's data you want to train the model with, and the program validates the trained model against a validation dataset that we separate from the training data.

When the model has finished training and the predictions are made, you can choose to serialize those predictions to a .csv file in your project directory.

If you choose to make changes to the program, you can find a series of automated tests in automated_tests.py to verify that the program runs successfully. These tests are run in parallel for time efficiency.

## Installation

Clone the repository to your local machine using git

```bash
git clone <use ssh or http cloning url>
```

## Configuration

Navigate to the root of the project and install the required python libraries with

```bash
pip install -r requirements.txt
```

You will also need a tiingo api token, to get one you can create a free account at [TIINGO](https://www.tiingo.com/)

Finally, create a .env file in the project's root directory and write the following line in it:

```.evn
TIINGO_API_KEY=<your tiingo api key>
```

Congratulations, you're ready to create your own model and predict future stock prices with it!

## Running the project

You can create and run the linear regression with

```bash
python linear_regression.py
```

You can create and run the neural network with 

```bash
python neural_network.py
```

If you only want to download some training data, you can run the command
```bash
python get_data.py
```

## Running the automated tests
You can run the automated tests with
```bash
python automated_tests.py
```

## Contributing
Please create an issue if you want to contribute or want me to make some changes. Thank you!