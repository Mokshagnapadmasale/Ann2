import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import streamlit as st

with open("label.pkl", "rb") as file:
    label = pickle.load(file)
with open("preprocess.pkl", "rb") as file:
    preprocess = pickle.load(file)

st.title("Weather Classification")

Temperature = st.slider("Temperature (°C)", -100, 100, step=1)
Humidity = st.number_input("Humidity (%)", min_value=1, max_value=100, step=1)
WindSpeed = st.number_input("Wind Speed (km/h)", min_value=0.0, step=0.1)
Precipitation = st.number_input("Precipitation (mm)", min_value=0.0, step=0.1)
CloudCover = st.selectbox("Cloud Cover", ["overcast", "partly cloudy", "clear", "cloudy"])
AtmosphericPressure = st.number_input("Atmospheric Pressure (hPa)", min_value=900.0, step=0.1)
UVIndex = st.slider("UV Index", 0, 15, step=1)
Season = st.selectbox("Season", ["Winter", "Spring", "Autumn", "Summer"])
Visibility = st.number_input("Visibility (km)", min_value=0.0, step=0.1)
Location = st.selectbox("Location", ["inland", "mountain", "coastal"])

input_data = pd.DataFrame({
    "Temperature": [Temperature],
    "Humidity": [Humidity],
    "WindSpeed": [WindSpeed],
    "Precipitation": [Precipitation],
    "CloudCover": [CloudCover],
    "AtmosphericPressure": [AtmosphericPressure],
    "UVIndex": [UVIndex],
    "Season": [Season],
    "Visibility": [Visibility],
    "Location": [Location],
})

input_data["Season"] = label.transform(input_data["Season"])

user_df_encoded = preprocess.transform(input_data)

model = load_model("model.h5")

if st.button("Predict Weather"):
    prediction = model.predict(user_df_encoded)
    predicted_label = label.inverse_transform([np.argmax(prediction)])
    st.success(f"Predicted Weather Type: {predicted_label[0]}")
