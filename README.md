# Stock Market Prediction Using ARIMA and LSTM

This project compares two machine learning approaches, ARIMA and LSTM, for stock market price prediction. The goal is to analyze their performance and determine which is more effective under different scenarios.

## Project Overview
- **Objective**: Predict stock prices and compare the performance of ARIMA and LSTM models.
- **Dataset**: Tata Motors stock prices for 10 years monthly close price.
- **Source**: Yahoo Finance.

## Models Implemented
### ARIMA Model
- Checked stationarity using the Augmented Dickey-Fuller (ADF) test.
- Differenced non-stationary data to make it stationary.
- Used grid search to find the best parameters (p, d, q).
- Evaluated the model using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

### LSTM Model
- Scaled data using MinMaxScaler for better neural network performance.
- Used a sliding window (look-back) to prepare sequential data for time series prediction.
- Built a two-layer LSTM model with 50 units in each LSTM layer and a Dense output layer.
- Evaluated the model using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

## Results and Insights
- The **ARIMA model** works well for short-term predictions in stable markets.
- The **LSTM model** captures non-linear patterns better, making it suitable for long-term or volatile market predictions.

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
  
2. Install the required Python libraries:
   ```bash
    pip install -r requirements.txt

3. Place your stock price dataset (CSV file) in the project directory.
  
4. Run the Python script:
     ```bash
    python main.py

## Future Enhancements
Add more features like trading volume and macroeconomic indicators.
Explore hybrid models combining ARIMA and LSTM.
Perform hyperparameter tuning for better LSTM performance.
This project demonstrates the strengths and weaknesses of ARIMA and LSTM models in financial forecasting. It provides a solid foundation for anyone interested in time series analysis and machine learning in finance.
