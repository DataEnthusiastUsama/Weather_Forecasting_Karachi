from flask import Flask, request, jsonify
import pickle
import pandas as pd
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
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

@app.route('/get-current-weather', methods=['GET'])
def get_current_weather():
    # Get the current date
    current_date = datetime.now().date()

    # Calculate the end date for the next two days
    end_date = current_date + timedelta(days=2)

    future_predictions = predict_future_temperature(loaded_arima_model, current_date, end_date)

    # Apply rounding condition to predicted temperatures
    rounded_temperatures = []

    for temperature in future_predictions['Predicted_Temperature']:
        if temperature >= 0.5:
            rounded_temperatures.append(round(temperature))
        else:
            rounded_temperatures.append(round(temperature - 0.5))

    # Add the rounded temperatures as a new column in the DataFrame
    future_predictions['Rounded_Temperature'] = rounded_temperatures

    conditions = {
        "snowy": range(1, 21),
        "stormy": range(21, 33),
        "cloudy": range(33, 69),
        "warm": range(69, 87),
        "sunny": range(87, 200)
    }

    future_predictions['Weather'] = [condition for temperature in future_predictions['Rounded_Temperature']
                                      for condition, temperature_range in conditions.items()
                                      if temperature in temperature_range]

    # Convert the DataFrame to a list of dictionaries with the desired date format
    result = future_predictions.to_dict(orient='records')

    # Convert date values to the "Month Day, Year" format
    for item in result:
        item['Date'] = item['Date'].strftime('%B %d, %Y')

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
