from flask import Flask, render_template, jsonify
from keras.models import load_model
import numpy as np
import pandas as pd
import requests
import joblib

API_KEY = '1171402ea5744f1a99713605232104'
MODEL_PATH = 'model.pkl'

app = Flask(__name__)

# Load the trained machine learning model from file
pickled_model = joblib.load(MODEL_PATH)
preprocessing_pipeline = joblib.load('preprocessing.pkl')

def get_weather_data(location):
    url = f'https://api.weatherapi.com/v1/current.json?key={API_KEY}&q={location}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temp = data['current']['temp_c']
        wind = data['current']['wind_kph']
        humidity = data['current']['humidity']
        weather = data['current']['condition']['text']
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
            'Mostly sunny': 'Sunny'
        }
        mapped_weather_condition = weather_mapping.get(weather)
        df = pd.DataFrame({'temp': [temp], 'wind': [wind], 'humidity': [humidity], 'weather': [mapped_weather_condition]})
        return df
    else:
        return None

@app.route('/weather/<location>')
def weather(location):
    test_features = get_weather_data(location)
    prepared_api = preprocessing_pipeline.transform(test_features)
    if test_features is not None:
        # Use the weather data to make a prediction with the trained machine learning model
        predicted_pickle = pickled_model.predict(prepared_api)
        # Return the predicted solar power as a JSON response
        return jsonify({'predicted_solar_power': predicted_pickle[0]})
    else:
        return jsonify({'error': 'Unable to retrieve weather data.'})

if __name__ == '__main__':
    app.run(debug=True)