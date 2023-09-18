import pickle
import pandas as pd
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA

# Load the ARIMA model from the saved file
with open('arima_model_final.pkl', 'rb') as model_file:
    loaded_arima_model = pickle.load(model_file)

# Function to make future predictions using the loaded model
def predict_future_temperature(model, start_date, end_date):
    # Generate a date range for future predictions
    future_date_range = pd.date_range(start_date, end_date)

    # Make predictions for each future date
    predictions = model.forecast(steps=len(future_date_range))

    # Create a DataFrame with predicted temperatures and dates
    future_predictions = pd.DataFrame({'Date': future_date_range, 'Predicted_Temperature': predictions})

    return future_predictions

# Example usage: Predict temperatures for future dates using the loaded model
future_start_date = datetime(2023, 9, 1)
future_end_date = datetime(2023, 9, 10)
future_predictions = predict_future_temperature(loaded_arima_model, future_start_date, future_end_date)

# Apply rounding condition to predicted temperatures
rounded_temperatures = []

for temperature in future_predictions['Predicted_Temperature']:
    if temperature >= 0.5:
        rounded_temperatures.append(round(temperature))
    else:
        rounded_temperatures.append(round(temperature - 0.5))

# Add the rounded temperatures as a new column in the DataFrame
future_predictions['Rounded_Temperature'] = rounded_temperatures
future_predictions = future_predictions.drop(['Predicted_Temperature'], axis=1)
future_predictions.rename(columns={'Rounded_Temperature': 'Temperature'}, inplace=True)

conditions = {
    "snowy": range(1, 21),
    "stormy": range(21, 33),
    "cloudy": range(33, 69),
    "warm": range(69, 87),
    "sunny": range(87, 200)
}

future_predictions['Weather'] = [condition for temperature in future_predictions['Temperature']
                                  for condition, temperature_range in conditions.items()
                                  if temperature in temperature_range]

# Print the DataFrame with rounded temperatures and weather conditions
print(future_predictions)
