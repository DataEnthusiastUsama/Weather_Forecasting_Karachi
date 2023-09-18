import pickle
import pandas as pd

# Load the pre-trained ARIMA model
with open("model.pkl", "rb") as model_file:
    weather_arima_model = pickle.load(model_file)

# Example: Prepare input data for a specific date
input_date = pd.to_datetime("2023-09-10")
end_input_date = pd.to_datetime("2023-09-13")
# Make predictions for the specified date
temperature_prediction = weather_arima_model.predict(start=input_date, end=end_input_date)

print(f"Predicted temperature for {input_date}: {temperature_prediction.values[0]:.2f} °C")
