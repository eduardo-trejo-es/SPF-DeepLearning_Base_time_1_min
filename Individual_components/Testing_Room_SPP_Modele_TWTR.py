from typing import Final
import yfinance as yf


from datetime import datetime, timedelta

import numpy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

#-- to crate directories
import os
from shutil import rmtree


import shutil #to delete directories
import csv

#-------------------  to save and load models --------------
import tensorflow as tf
from tensorflow import keras

################################
#    Read the index file       #
################################
NameDirectoryPathsFIles="/home/eduardo/Desktop/SPP_deep_learning/BatchDataGenerator_Qt_file/All_Batches_TWTR/IndexFile.csv"

#_-------------- To read the files ----------------------
with open(NameDirectoryPathsFIles, newline='') as csvfile:
        TestFIleCSV= csv.reader(csvfile, delimiter=' ', quotechar='|')
        TestFIleCSV=list(TestFIleCSV)
        
# ----- Retrieving numpyArrays test ------
print(type(TestFIleCSV))
index_N=0
for i in TestFIleCSV:
        print(index_N)
        if index_N == 8:
                index_N=0
                print(i[0])
        if index_N == 0:
                trainX=numpy.load(i[0])
        if index_N == 1:
                trainY_Open=numpy.load(i[0])
        if index_N == 2:
                trainY_High=numpy.load(i[0])
        if index_N == 3:
                trainY_Low=numpy.load(i[0])
        if index_N == 4:
                trainY_Close=numpy.load(i[0])
        if index_N == 5:
                trainY_Adj_Close=numpy.load(i[0])
        if index_N == 6:
                trainY_Volume=numpy.load(i[0])
        
        if index_N == 7:
                print('trainX shape == {}.'.format(trainX.shape))
                print('trainY_Open shape == {}.'.format(trainY_Open.shape))
                print('trainY_High shape == {}.'.format(trainY_High.shape))
                print('trainY_Low shape == {}.'.format(trainY_Low.shape))
                print('trainY_Close shape == {}.'.format(trainY_Close.shape))
                print('trainY_Adj_Close shape == {}.'.format(trainY_Adj_Close.shape))
                print('trainY_Volume shape == {}.'.format(trainY_Volume.shape))
                print("---------------------- Another one ----------------------------")
        


                ###################################
                #         To train models         #
                ###################################
                """Checkpoint callback usage
                Create a tf.keras.callbacks.ModelCheckpoint callback that saves weights only during training:
                https://www.tensorflow.org/tutorials/keras/save_and_load """

                Model_Path='/home/eduardo/Desktop/SPP_deep_learning/BatchDataGenerator_Qt_file/Py files Missing to modify/Models/SPP_Model'

                #-----------   Loding model

                model = tf.keras.models.load_model(Model_Path)
                # Check its architecture
                #model.summary()
                #--------------------------- Assing Y data to losses dictionary -----
                y_data={ 
                "dense": trainY_Open,
                "dense_1": trainY_High,
                "dense_2": trainY_Low,
                "dense_3": trainY_Close,
                "dense_4": trainY_Adj_Close,
                "dense_5": trainY_Volume,
                }
                #------------------------- Training model --------------------------------
                test_scores = model.evaluate(trainX,y=y_data, verbose=2)
                #history = model.fit(trainX,y=y_data, epochs=30, batch_size=16, validation_split=0.1, verbose=1)
                print("-------------------------------------------------------")
                print("Test loss:", test_scores[0])
                print("Test accuracy:", test_scores[1])


                """plt.plot(history.history['loss'], label='Training loss')
                plt.legend()"""


                #------------------- Save model ---------------------------------

                # datetime object containing current date and time
                now = datetime.now()
                
                print("now =", now)

                # dd/mm/YY H:M:S
                dt_string = now.strftime("%d_%m_%Y %H:%M:%S")
                print("date and time =", dt_string)	

                #------------------- To save the model with date --------------------
                #Model_Path=Model_Path+"_"+dt_string
                rmtree(Model_Path)
                #Model_Path=Model_Path+"_"+dt_string
                model.save(Model_Path)

                """loaded_array = np.load(PathFile)
                print('trainX shape == {}.'.format(loaded_array.shape))"""
        index_N=index_N+1