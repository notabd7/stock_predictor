import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plot

model = load_model('C:\\Users\\muham\\Desktop\\personal_projects\\stock_predictor\\Stock Predictor Model.keras')
st.header('Stock Price Predictor')

stock  = st.text_input('Enter Stock Symbol', 'GOOG')

start_date = '2013-01-01'
end_date = '2023-12-25'


data = yf.download(stock, start_date, end_date)

st.subheader('Raw Stock Data')
st.write(data)

x = int(len(data)*0.8)
data_train = pd.DataFrame(data.Close[0:x])
data_test = pd.DataFrame(data.Close[x: len(data)])


scaler = MinMaxScaler(feature_range=(0,1))

#past 100 days of data dfrom training data
pas_100_data = data_train.tail(100)
data_test = pd.concat([pas_100_data, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader("Average of 50 days (Moving Average)")
#moving average(mean) of 50 days
ma_50_days = data.Close.rolling(50).mean()
fig1 = plot.figure(figsize=(10,8))
plot.plot(ma_50_days,  'b')
plot.plot(data.Close, 'g')
st.pyplot(fig1)

st.subheader("Average of 100 days (Moving Average)")
#moving average(mean) of 100 days
ma_100_days = data.Close.rolling(100).mean()
fig2 = plot.figure(figsize=(10,8))
plot.plot(ma_100_days,  'r')
plot.plot(data.Close, 'g')
st.pyplot(fig2)

st.subheader("Average of 200 days (Moving Average)")
#moving average(mean) of 200 days
ma_200_days = data.Close.rolling(200).mean()
fig3 = plot.figure(figsize=(10,8))
plot.plot(ma_200_days,  'y')
plot.plot(data.Close, 'g')
st.pyplot(fig3)

st.subheader("Stock Price vs 50 day vs 100 day vs 200 day")
fig4 =plot.figure(figsize=(10,8))
plot.plot(ma_50_days,  'b')
plot.plot(ma_100_days,  'r')
plot.plot(ma_200_days,  'y')
plot.plot(data.Close, 'g')
st.pyplot(fig4)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x, y = np.array(x), np.array(y)


#displaying statistical visualizations


predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale

y=y* scale


st.subheader("Predicted vs Original")
fig5 = plot.figure(figsize=(10,8))
plot.plot(predict,  'b', label = "Original Price")
plot.plot(y,  'r', label = "Predicted Price" )
plot.xlabel("Time")
plot.ylabel("Price")
st.pyplot(fig5)