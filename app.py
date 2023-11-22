import os
import json
import joblib
import calendar
import pandas as pd
from datetime import datetime
import streamlit as st
import altair as alt

# Get the current directory of the script
current_directory = os.path.dirname(os.path.realpath(__file__))

# Define relative paths to models
models_directory = os.path.join(current_directory, "../models")

min_model_path = os.path.join(models_directory, 'minimum.pkl')
median_model_path = os.path.join(models_directory, 'median.pkl')
modal_model_path = os.path.join(models_directory, 'mode.pkl')
mean_model_path = os.path.join(models_directory, 'mean.pkl')
flightroutes_path = os.path.join(models_directory, 'flightroutes.json')

# Load models and data
min_model = joblib.load(min_model_path)
median_model = joblib.load(median_model_path)
modal_model = joblib.load(modal_model_path)
mean_model = joblib.load(mean_model_path)    

with open(flightroutes_path, 'r') as file:
    routes = json.load(file)

# Airport lists
origin_airports = ['OAK', 'IAD', 'DEN', 'LGA', 'LAX', 'ONT', 'ATL', 'DFW', 'FLL',
                   'CLT', 'PHL', 'TTN', 'DTW', 'JFK', 'DAL', 'BOS', 'EWR', 'SFO', 'ORD', 'MIA']

destination_airports = ['DEN', 'LAX', 'PHL', 'DTW', 'ORD', 'SFO', 'ATL', 'BOS', 'CLT',
                        'DFW', 'EWR', 'IAD', 'JFK', 'LGA', 'MIA', 'OAK', 'ONT', 'DAL', 'TTN', 'FLL']

# Time categories
departure_times = ['Early Morning', 'Morning', 'Midday', 'Afternoon', 'Evening', 'Night', 'Late Night']

def getfarepredictions(origin, destination, day_of_month, mm, yr, time_category, cabin):
    
    # checking that route exists
    if (origin in routes) and (destination in routes[origin]):

        # generating date features

        mm = {month: index for index, month in enumerate(calendar.month_name) if month}.get(mm, None)

        date = pd.to_datetime(f'{int(yr)}-{mm}-{int(day_of_month)}')
        today = pd.to_datetime('today')

        day_of_week = pd.Series(date).dt.dayofweek
        days_from_flight = (date - today).days

        day_mapping = {
        0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'
        }

        day_name = day_mapping[date.weekday()]

        # Minimum fare prediction

        min_df = pd.DataFrame({'segmentsDepartureAirportCode': [origin],
        'segmentsArrivalAirportCode': [destination],
        'day_of_month': [day_of_month],
        'day_of_week': [day_of_week],
        'month': [mm],
        'year': [yr],
        'time_category': [time_category],
        'segmentsCabinCode': [cabin],
        'days_from_flight': [days_from_flight]
        })

        min_fare = round(min_model.predict(min_df)[0], 2)

        # Median fare prediction
        median_df = pd.DataFrame({'segmentsDepartureAirportCode': [origin],
        'segmentsArrivalAirportCode': [destination],
        'day_of_month': [day_of_month],
        'day_of_week': [day_of_week],
        'month': [mm],
        'year': [yr],
        'time_category': [time_category],
        'segmentsCabinCode': [cabin],
        'days_from_flight': [days_from_flight]
        })

        median_fare = round(median_model.predict(median_df)[0], 2)

        # Mean Fare Prediction
        mean_df = pd.DataFrame({'segmentsDepartureAirportCode': [origin],
        'segmentsArrivalAirportCode': [destination],
        'day_of_month': [day_of_month],
        'day_of_week': [day_of_week],
        'month': [mm],
        'year': [yr],
        'time_category': [time_category],
        'segmentsCabinCode': [cabin],
        'days_from_flight': [days_from_flight]
        })

        
        mean_fare = round(mean_model.predict(mean_df)[0], 2)

        # Modal fare prediction
        modal_df = pd.DataFrame({'segmentsDepartureAirportCode': [origin],
        'segmentsArrivalAirportCode': [destination],
        'day_of_month': [day_of_month],
        'day_of_week': [day_of_week],
        'month': [mm],
        'year': [yr],
        'time_category': [time_category],
        'segmentsCabinCode': [cabin],
        'days_from_flight': [days_from_flight]
        })

        modal_fare = round(modal_model.predict(modal_df)[0], 2)

        return min_fare, median_fare, mean_fare, modal_fare
    else:
        raise st.Error('Flight route not found! Please try different airports')

# Streamlit interface
st.title("Flight Fare Prediction for US flights")

origin = st.selectbox("Origin Airport", origin_airports)
destination = st.selectbox("Destination Airport", destination_airports)
cabin = st.radio("Cabin", ['coach', 'premium coach', 'first', 'business'])

day_of_month = st.number_input("Day", min_value=1, max_value=31)
mm = st.selectbox("Month", list(calendar.month_name)[1:])
yr = st.number_input("Year", min_value=2020, max_value=2030)
time_category = st.selectbox("Departure Time", departure_times)

if st.button("Submit"):
    predictions = getfarepredictions(origin, destination, day_of_month, mm, yr, time_category, cabin)
    if predictions:
        min_fare, median_fare, mean_fare, modal_fare = predictions
        st.write(f"Minimum Prediction: {min_fare}")
        st.write(f"Median Prediction: {median_fare}")
        st.write(f"Mean Prediction: {mean_fare}")
        st.write(f"Modal Prediction: {modal_fare}")
