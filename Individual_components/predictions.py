from typing import Final
from numpy.core.fromnumeric import shape
import yfinance as yf

from datetime import datetime, timedelta

import numpy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

#-- to plot data 
import mplfinance as mpf

#-- to crate directories
import os
from shutil import rmtree


import shutil #to delete directories
import csv

#-------------------  to save and load models --------------
import tensorflow as tf
from tensorflow import keras

#Get the yesterday date
yesterday=datetime.strftime(datetime.now() - timedelta(1), '%Y-%m-%d')
#print("Fecha de ayer:", yesterday)
#get the stock quote

df= yf.download('TWTR',start='2013-11-18',end=yesterday)

df_topop=yf.download('TWTR',start='2013-11-18',end=yesterday)
#df= yf.download('TWTR',start='2013-11-18',end='2021-01-01')
#Get the number of rows and colums in the data set
#print(df["Close"].head())
#print("End")
print(df.head()) #7 columns, including the Date. 
print(type(df))

#---------------------------------------------------

#Separate dates for future plotting
train_dates = df.index
train_date=pd.to_datetime(train_dates)
#train_dates = pd.to_datetime(df)
#train_dates = train_dates.index
#print(train_dates.tail(15)) #Check last few dates. 
print(train_dates.shape)
#--------------------------------------
#Variables for training
cols = list(df)[0:6]
#Date and volume columns are not used in training. 
print(cols) #['Open', 'High', 'Low', 'Close', 'Adj Close']

#----------------------------------------
#New dataframe with only training data - 5 columns
df_for_training = df[cols].astype(float)

"""df_for_training_Open=yOpen.astype(float)
df_for_training_High=yHigh.astype(float)
df_for_training_Low=yLow.astype(float)
df_for_training_Close=yClose.astype(float)
df_for_training_Adj_Close=yAdj_Close.astype(float)
df_for_training_Volume=yVolume.astype(float)"""

#Were here 8/23/21
"""print(type(df_for_training_Close))
print(type(df_for_training))"""

print(len(df_for_training))
# df_for_plot=df_for_training.tail(5000)
# df_for_plot.plot.line()
#-------------------------------------------
#LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
#Scaling for input X colums
scaler = StandardScaler()

scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

#Scaling for OutPuts "Y" colums
df_for_training_Open_scaled=df_for_training_scaled[:,[0]] #selecting colums from a numpy array
df_for_training_High_scaled=df_for_training_scaled[:,[1]]
df_for_training_Low_scaled=df_for_training_scaled[:,[2]]
df_for_training_Close_scaled=df_for_training_scaled[:,[3]]
df_for_training_Adj_Close_scaled=df_for_training_scaled[:,[4]]
df_for_training_Volume_scaled=df_for_training_scaled[:,[5]]


#-----------------------------------
#Start with the last day in training date and predict future...
N_Days_to_predict=30#Redefining n_future to extend prediction dates beyond original n_future dates...

predict_period_dates = pd.date_range(list(train_dates)[-1], periods=N_Days_to_predict, freq='1d').tolist()
#predict_period_dates = pd.date_range(list(train_dates)[-1], periods=n_days_for_prediction, freq=us_bd).tolist()
print(predict_period_dates)

#--------------------------------------------------

#--------------------------  load model -----------------------------
Model_Path='/home/eduardo/Desktop/SPP_deep_learning/BatchDataGenerator_Qt_file/Py files Missing to modify/Models/SPP_Model'
model = tf.keras.models.load_model(Model_Path)

#-------------------------- load batch ----------------
PathNumpyPredictionBatch='/home/eduardo/Desktop/SPP_deep_learning/BatchDataGenerator_Qt_file/All_Batches_TWTR/2014-05-12-TO-2014-07-23/trainX.npy'
XToPredict=numpy.load(PathNumpyPredictionBatch)

print(XToPredict.shape)
#-------------------------------Forcasting-----------------------------...


Prediction_Saved=[]
Batch_to_predict=XToPredict[len(XToPredict)-1:]
#print(Batch_to_predict)
#print(Batch_to_predict.shape)
#print("--------------------------------")
for i in range(N_Days_to_predict):
  prediction = model.predict(Batch_to_predict) #the input is a 30 days batch
  prediction_Reshaped=np.reshape(prediction,(1,1,6))
  Batch_to_predict=np.append(Batch_to_predict,prediction_Reshaped, axis=1)
  Batch_to_predict=np.delete(Batch_to_predict,0,1)
  print(Batch_to_predict.shape)
  #print(Batch_to_predict)
  #Batch_to_predict=prediction_Reshaped
  Prediction_Saved.append(prediction_Reshaped)
  #Perform inverse transformation to rescale back to original range
  #Since we used 5 variables for transform, the inverse expects same dimensions
  #Therefore, let us copy our values 5 times and discard them after inverse transform
#print(Batch_to_predict)
#print(Batch_to_predict)
#print(len(Prediction_Saved))#<----------- 20 days predicted
#print(Prediction_Saved)

y_pred_future = scaler.inverse_transform(Prediction_Saved)[:,0]
y_pred_future=y_pred_future
print(y_pred_future[:3])
predict_Open=[]
predict_High=[]
predict_Low=[]
predict_Close=[]
predict_Adj_Close=[]
predict_Volume=[]

for i in range(len(y_pred_future)):
  predict_Open.append(y_pred_future[i][0][0])
  predict_High.append(y_pred_future[i][0][1])
  predict_Low.append(y_pred_future[i][0][2])
  predict_Close.append(y_pred_future[i][0][3])
  predict_Adj_Close.append(y_pred_future[i][0][4])
  predict_Volume.append(y_pred_future[i][0][5])

print(type(predict_Close))

#-----------------------------------------------

#---Generating the generating <class 'pandas.core.frame.DataFrame'> of prediction

#--------  data shape it's (x days, 6 columns)
# Convert timestamp to date
forecast_dates = []
print(len(predict_period_dates))
for time_i in predict_period_dates:
    forecast_dates.append(time_i.date())

forecast_dates=pd.DatetimeIndex(forecast_dates)


df_forecast = pd.DataFrame({'Open':predict_Open,'High':predict_High, 'Low':predict_Low,'Close':predict_Close,'Adj Close':predict_Adj_Close,'Volume':predict_Volume}, index=forecast_dates)

df_forecast.index.name="Date"

print(df_forecast.head())


#_---------------- geting the candle chart  -

mpf.plot(df_forecast, type='candle',title='TWTR data ', style='charles')


"""

df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Open':predict_Open})
df_forecast['Date']=pd.to_datetime(df_forecast['Date'])

print(original[len(original)-90:])
print(df_forecast)

plt.plot(original[len(original)-90:])

plt.plot(df_forecast['Date'], df_forecast['Close'])
sns.lineplot(original[len(original)-90:])
sns.lineplot(df_forecast['Date'], df_forecast['Close'])
"""


