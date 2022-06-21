from datetime import datetime, timedelta
import numpy
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

from tensorflow import keras

#-------------------  to save and load models --------------
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers.core import Activation

#-------------------- Define functional model simplified. ----------------------------------

keras.backend.clear_session()  # Reseteo sencillo
#we were here 8/25/2021 need to be inproved the model 
#Choose 
#https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/
"""VIdeo Multiple outputs min 6:30
https://www.youtube.com/watch?v=JN08CqZKKkA&ab_channel=PythonEngineer"""

#https://www.tensorflow.org/guide/keras/functional

#---------Layes are created

inputs=keras.Input(shape=(30,5))

x=LSTM_Layer1=keras.layers.LSTM(90, input_shape=(30,5), return_sequences=True, activation='sigmoid')(inputs)
x=LSTM_Layer1=keras.layers.LSTM(90, input_shape=(30,5), return_sequences=True, activation='sigmoid')(inputs)
#x=Dropout_layer1=keras.layers.Dropout(0.2)(x)

x=LSTM_Layer4=keras.layers.LSTM(90, return_sequences=False)(x)
dense=Dropout_layer4=keras.layers.Dropout(0.2)(x)# modify

#dense=keras.layers.Dense(80,activation='relu')(x)


#---------------------------Outputs
dense2=keras.layers.Dense(1)(dense)
dense2_2=keras.layers.Dense(1)(dense)
dense2_3=keras.layers.Dense(1)(dense)
dense2_4=keras.layers.Dense(1)(dense)
dense2_5=keras.layers.Dense(1)(dense)


#-------Layers outputs are linked

outputs=dense2
outputs2=dense2_2
outputs3=dense2_3
outputs4=dense2_4
outputs5=dense2_5


#-----The model it's created

model=keras.Model(inputs=inputs, outputs=[outputs,outputs2,outputs3,outputs4,outputs5], name='Prices_Prediction')
#model=keras.Model(inputs=[inputs,None], outputs=[outputs,outputs2,outputs3,outputs4,outputs5,outputs6], name='Prices_Prediction')

#print(model.summary())

#keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)


#------------------- Loss and optimizer ----------------------------------------
#got to ensure MeanAbsoluteError it's the good one for our data
loss1 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss2 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss3 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss4 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss5 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
optim=keras.optimizers.Adam(1e-3)
metrics=["accuracy"]

losses={
    "dense": loss1,
    "dense_1": loss2,
    "dense_2": loss3,
    "dense_3": loss4,
    "dense_4": loss5,
}

model.compile(loss=losses, optimizer=optim, metrics=metrics)

print(model.summary())

keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)


#------------------- Save model ---------------------------------


Model_Path='Individual_components/Models/SPP_Model'


# datetime object containing current date and time
now = datetime.now()
 
print("now =", now)

# dd/mm/YY H:M:S
dt_string = now.strftime("%d_%m_%Y %H:%M:%S")
print("date and time =", dt_string)	

#------------------- To save the model with
Model_Path=Model_Path+"_"+dt_string

model.save(Model_Path)

#-----------   Loding model
"""print("-------Loding model")
new_model = tf.keras.models.load_model(Model_Path)

# Check its architecture
new_model.summary()"""

