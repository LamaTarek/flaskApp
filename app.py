from flask import Flask, render_template, jsonify
from keras.models import load_model
import numpy as np
import pandas as pd
import requests
import joblib


API_KEY = '69b0e0e0f0694b23890134212232206'
MODEL_PATH = 'model.pkl'

app = Flask(__name__)

# Load the trained machine learning model from file
pickled_model = joblib.load(MODEL_PATH)
preprocessing_pipeline = joblib.load('preprocessing.pkl')

def get_weather_data(latitude, longitude, days):
    url = f'https://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={latitude},{longitude}&days={days}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()['forecast']['forecastday']
        weather_data = []
        for day_data in data:
            date = day_data['date']
            temp = day_data['day']['avgtemp_c']
            wind = day_data['day']['maxwind_kph']
            humidity = day_data['day']['avghumidity']
            weather = day_data['day']['condition']['text'] if 'condition' in day_data['day'] else None
            # Apply weather condition mapping
            weather_mapping = {
                'Mist': 'Fog',
                'Haze': 'Fog',
                'Fog': 'Fog',
                'Clear': 'Sunny',
                'Sunny': 'Sunny',
                'Broken clouds': 'Scattered clouds',
                'Scattered clouds': 'Scattered clouds',
                'Overcast': 'Cloudy',
                'More clouds than sun': 'Cloudy',
                'More sun than clouds': 'Sunny',
                'Low clouds': 'Cloudy',
                'Mostly cloudy': 'Cloudy',
                'Cloudy': 'Cloudy',
                'Passing clouds': 'Passing clouds',
                'Partly sunny': 'Partly sunny',
                'Mostly sunny': 'Sunny',
                'Patchy rain possible': 'Scattered clouds',
                'Patchy snow possible': 'Scattered clouds',
                'Patchy sleet possible': 'Scattered clouds',
                'Patchy freezing drizzle possible': 'Scattered clouds',
                'Thundery outbreaks possible': 'Cloudy',
                'Blowing snow': 'Cloudy',
                'Blizzard': 'Cloudy',
                'Freezing fog': 'Fog',
                'Patchy light drizzle': 'Scattered clouds',
                'Light drizzle': 'Scattered clouds',
                'Freezing drizzle': 'Scattered clouds',
                'Heavy freezing drizzle': 'Cloudy',
                'Patchy moderate snow': 'Scattered clouds',
                'Moderate snow': 'Scattered clouds',
                'Patchy heavy snow': 'Cloudy',
                'Heavy snow': 'Cloudy',
                'Ice pellets': 'Scattered clouds',
                'Light rain shower': 'Passing clouds',
                'Moderate or heavy rain shower': 'Passing clouds',
                'Torrential rain shower': 'Passing clouds',
                'Light sleet showers': 'Scattered clouds',
                'Moderate or heavy sleet showers': 'Cloudy',
                'Light snow showers': 'Scattered clouds',
                'Moderate or heavy snow showers': 'Cloudy',
                'Light showers of ice pellets': 'Scattered clouds',
                'Moderate or heavy showers of ice pellets': 'Cloudy',
                'Patchy light rain with thunder': 'Cloudy',
                'Moderate or heavy rain with thunder': 'Cloudy',
                'Patchy light snow with thunder': 'Cloudy',
                'Moderate or heavy snow with thunder': 'Cloudy'
            }
            mapped_weather_condition = weather_mapping.get(weather, None)
            if mapped_weather_condition is None:
                # Replace missing weather value 
                mapped_weather_condition = 'Sunny'
            weather_data.append({'date': date, 'temp': temp, 'wind': wind, 'humidity': humidity, 'weather': mapped_weather_condition})
        return pd.DataFrame(weather_data)
    else:
        return None

@app.route('/<float:latitude>/<float:longitude>')
@app.route('/<float:latitude>/<float:longitude>/<int:days>')
def weather(latitude, longitude, days=14):
    test_features = get_weather_data(latitude, longitude, days)
    # Use the weather data to make a prediction with the trained machine learning model
    prepared_api = preprocessing_pipeline.transform(test_features)
    # Return the predicted solar power as a JSON response for each day in the forecast
    if test_features is not None:
        # Use the weather data to make a prediction with the trained machine learning model
        predicted_pickle = pickled_model.predict(prepared_api)
        # Return the predicted solarpower as a JSON response for each day in the forecast
        predicted_solar_power = np.exp(predicted_pickle) - 1
        result = {'predictions': []}
        for i in range(len(predicted_solar_power)):
            date = test_features.iloc[i]['date']
            prediction = predicted_solar_power[i]
            result['predictions'].append({'date': date, 'predicted_solar_power': prediction})
        return jsonify(result)
    else:
        return 'Error retrieving weather data.'

if __name__ == '__main__':
    app.run(debug=True)
