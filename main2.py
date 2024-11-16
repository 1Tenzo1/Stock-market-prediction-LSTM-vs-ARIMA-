import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ------------------- ARIMA Implementation -------------------
print("#########################   ARIMA Model   ###################################")
# Load the CSV file for ARIMA
file_path = 'TATAMOTORS 2023-2024 (1 year).csv'
data_arima = pd.read_csv(file_path)

# Convert the 'Date' column to datetime
data_arima['Date'] = pd.to_datetime(data_arima['Date'])

# Set the 'Date' column as the index
data_arima.set_index('Date', inplace=True)

# Use only the 'Close' column for prediction
close_prices_arima = data_arima['Close']
close_prices_arima.plot()
plt.ylabel("Close price")
plt.show()

plot_acf(close_prices_arima)
plot_pacf(close_prices_arima)
plt.show()


# Check for stationarity
result = adfuller(close_prices_arima)
print(f'ADF Statistic (ARIMA): {result[0]}')
print(f'p-value (ARIMA): {result[1]}')

# If the p-value is > 0.05, the series is not stationary
# Differencing the series if not stationary
if result[1] > 0.05:
    close_prices_arima = close_prices_arima.diff().dropna()

# Split the data into training and testing sets
train_size_arima = int(len(close_prices_arima) * 0.8)
train_arima, test_arima = close_prices_arima[:train_size_arima], close_prices_arima[train_size_arima:]

# Custom Evaluation using MAE and RMSE for ARIMA
best_score_arima, best_cfg_arima = float("inf"), None
for p in range(0, 6):
    for d in range(0, 3):
        for q in range(0, 6):
            order = (p, d, q)
            try:
                model_arima = ARIMA(train_arima, order=order)
                model_fit_arima = model_arima.fit()
                predictions_arima = model_fit_arima.forecast(steps=len(test_arima))
                mae_arima = mean_absolute_error(test_arima, predictions_arima)
                if mae_arima < best_score_arima:
                    best_score_arima, best_cfg_arima = mae_arima, order
            except:
                continue

print(f'Best ARIMA {best_cfg_arima} MAE={best_score_arima}')

# Fit the best ARIMA model
model_arima = ARIMA(train_arima, order=best_cfg_arima)
model_fit_arima = model_arima.fit()

# Make predictions with ARIMA
predictions_arima = model_fit_arima.forecast(steps=len(test_arima))
predictions_arima = pd.Series(predictions_arima, index=test_arima.index)
predictions_arima = predictions_arima.dropna()

# Ensure the test and prediction series have the same length
min_len_arima = min(len(test_arima), len(predictions_arima))
test_arima_aligned = test_arima.iloc[:min_len_arima]
predictions_arima_aligned = predictions_arima.iloc[:min_len_arima]

# Calculate RMSE for ARIMA with aligned data
rmse_arima = np.sqrt(mean_squared_error(test_arima_aligned, predictions_arima_aligned))

print(f'RMSE (ARIMA): {rmse_arima}')

# Plot ARIMA predictions against actual values
plt.figure(figsize=(12, 6))
plt.plot(train_arima, label='Train (ARIMA)', color='blue')
plt.plot(test_arima, label='Test (ARIMA)', color='green')
plt.plot(predictions_arima, label='Predictions (ARIMA)', color='red')
plt.title('Stock Price Prediction using ARIMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
print("#############################################################################")

# ------------------- LSTM Implementation -------------------
# Load the CSV file for LSTM
file_path1 = 'TATAMOTORS 2023-2024 (1 year).csv'
data_lstm = pd.read_csv(file_path1)

# Convert the 'Date' column to datetime
data_lstm['Date'] = pd.to_datetime(data_lstm['Date'])
data_lstm.set_index('Date', inplace=True)

# Use only the 'Close' column for prediction in LSTM
close_prices_lstm = data_lstm['Close'].values
close_prices_lstm = close_prices_lstm.reshape(-1, 1)

# Scale the data for LSTM
scaler_lstm = MinMaxScaler(feature_range=(0, 1))
scaled_close_prices_lstm = scaler_lstm.fit_transform(close_prices_lstm)

# Define look_back
look_back_lstm = 3

# Prepare the data for LSTM
def create_dataset_lstm(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Create train and test datasets for LSTM
train_size_lstm = int(len(scaled_close_prices_lstm) * 0.7)
test_size_lstm = len(scaled_close_prices_lstm) - train_size_lstm
train_lstm, test_lstm = scaled_close_prices_lstm[0:train_size_lstm, :], scaled_close_prices_lstm[train_size_lstm:len(scaled_close_prices_lstm), :]

X_train_lstm, Y_train_lstm = create_dataset_lstm(train_lstm, look_back_lstm)
X_test_lstm, Y_test_lstm = create_dataset_lstm(test_lstm, look_back_lstm)

# Reshape input to be [samples, time steps, features] for LSTM
X_train_lstm = np.reshape(X_train_lstm, (X_train_lstm.shape[0], X_train_lstm.shape[1], 1))
X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], X_test_lstm.shape[1], 1))

# Build the LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(50, return_sequences=True, input_shape=(look_back_lstm, 1)))
model_lstm.add(LSTM(50))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mean_squared_error')

# Train the LSTM model
model_lstm.fit(X_train_lstm, Y_train_lstm, epochs=20, batch_size=1, verbose=2)

# Make predictions with LSTM
train_predictions_lstm = model_lstm.predict(X_train_lstm)
test_predictions_lstm = model_lstm.predict(X_test_lstm)

# Inverse transform predictions and actual values
train_predictions_lstm = scaler_lstm.inverse_transform(train_predictions_lstm)
test_predictions_lstm = scaler_lstm.inverse_transform(test_predictions_lstm)
Y_train_lstm = scaler_lstm.inverse_transform([Y_train_lstm])
Y_test_lstm = scaler_lstm.inverse_transform([Y_test_lstm])

# Calculate MAE and RMSE for LSTM
train_mae_lstm = mean_absolute_error(Y_train_lstm[0], train_predictions_lstm[:, 0])
test_mae_lstm = mean_absolute_error(Y_test_lstm[0], test_predictions_lstm[:, 0])
train_rmse_lstm = np.sqrt(mean_squared_error(Y_train_lstm[0], train_predictions_lstm[:, 0]))
test_rmse_lstm = np.sqrt(mean_squared_error(Y_test_lstm[0], test_predictions_lstm[:, 0]))

print(f'Train MAE (LSTM): {train_mae_lstm}')
print(f'Test MAE (LSTM): {test_mae_lstm}')
print(f'Train RMSE (LSTM): {train_rmse_lstm}')
print(f'Test RMSE (LSTM): {test_rmse_lstm}')

# Prepare plot data for LSTM
train_plot_lstm = np.empty_like(scaled_close_prices_lstm)
train_plot_lstm[:, :] = np.nan
train_plot_lstm[look_back_lstm:len(train_predictions_lstm) + look_back_lstm, :] = train_predictions_lstm

test_plot_lstm = np.empty_like(scaled_close_prices_lstm)
test_plot_lstm[:, :] = np.nan
test_plot_lstm[len(train_predictions_lstm) + (look_back_lstm * 2) + 1:len(scaled_close_prices_lstm) - 1, :] = test_predictions_lstm

# Plotting the results for LSTM
plt.figure(figsize=(12, 6))
plt.plot(scaler_lstm.inverse_transform(scaled_close_prices_lstm), label='Actual Stock Price (LSTM)', color='blue')
plt.plot(train_plot_lstm, label='Train Predictions (LSTM)', color='red', linewidth=2)
plt.plot(test_plot_lstm, label='Test Predictions (LSTM)', color='green', linewidth=2)
plt.title('Stock Price Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
