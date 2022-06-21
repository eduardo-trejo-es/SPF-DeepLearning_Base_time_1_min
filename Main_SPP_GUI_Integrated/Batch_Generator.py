#from typing import Final
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
import shutil #to delete directories
import csv

#To make possible the Threading
from PyQt5.QtCore import *


#_--------------------------------------------------------
class  WorkerThread_Batch_Gen(QThread):
    Update_Progress= pyqtSignal(int)
    Update_Progress_Description= pyqtSignal(str)
    LastBatchPathPredict=pyqtSignal(str)

    def __int_(self):
        pass
        #############################################
        #  To Update label and progress bar at GUI  #
        #############################################
        

    def GettingParametersForBatches(self,StartDate,ToDate,N_past,SelectedCompany,DirectoryPath, KindBatch):
        ##################################
        #  parameters to create batches  #
        ##################################
        self.startDate = StartDate
        self.toDate = ToDate

        self.n_past = N_past  # Number of past days we want to use to predict the future.

        self.selectedCompany = SelectedCompany

        self.directoryPath = DirectoryPath
        self.kindBatch=KindBatch

    def run(self):
        yesterday=datetime.strftime(datetime.now() - timedelta(1), '%Y-%m-%d')

        self.Update_Progress.emit(5)
        
        self.Update_Progress_Description.emit("Getting data from Yahoo Finances")

        #df_forscale= yf.download(self.selectedCompany,start='2014-11-18',end=yesterday)
        df_forscale= yf.download(self.selectedCompany,start='2014-11-18',end=self.toDate)
        df_forscale=df_forscale.drop('Adj Close', axis=1)
        print(df_forscale.head)
        df= yf.download(self.selectedCompany,start=self.startDate,end=self.toDate)
        df=df.drop('Adj Close', axis=1)

        #----------------------------------------------------------

        #Separate dates for future plotting
        train_dates = df.index
        train_dates=pd.to_datetime(train_dates)
        #print(train_dates.shape)

        #-------------------------------------------------------------
        self.Update_Progress.emit(15)
        self.Update_Progress_Description.emit("Data from Yahoo Finances gotten")
        #Variables for training
        cols = list(df)[0:5]
        #Date and volume columns are not used in training. 
        #print(cols) #['Open', 'High', 'Low', 'Close', 'Adj Close']

        #--------------------------------------------------------------
        self.Update_Progress.emit(25)
        self.Update_Progress_Description.emit("Extracting columns from data")
        #New dataframe with only training data - 5 columns
        df_for_training = df[cols].astype(float)

        #print(len(df_for_training))

        #-------------------------------------------------------------

        #LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
        # normalize the dataset
        #Scaling for input X colums
        scaler = StandardScaler()

        scaler = scaler.fit(df_forscale)
        df_for_training_scaled = scaler.transform(df_for_training)

        #Scaling for OutPuts "Y" colums
        df_for_training_Open_scaled=df_for_training_scaled[:,[0]] #selecting colums from a numpy array
        df_for_training_High_scaled=df_for_training_scaled[:,[1]]
        df_for_training_Low_scaled=df_for_training_scaled[:,[2]]
        df_for_training_Close_scaled=df_for_training_scaled[:,[3]]
        df_for_training_Volume_scaled=df_for_training_scaled[:,[4]]

        #------------------------------------------------------------------------------

        #Empty lists to be populated using formatted training data
        trainX = []
        trainX_sub=[]
        trainY_Open = []
        trainY_High = []
        trainY_Low = []
        trainY_Close = []
        trainY_Volume= []

        n_future = 1   # Number of days we want to look into the future based on the past days.


        #Reformat input data into a shape: (n_samples x timesteps x n_features)
        #In my example, my df_for_training_scaled has a shape (12823, 5)
        #12823 refers to the number of data points and 5 refers to the columns (multi-variables).
        self.Update_Progress.emit(50)
        self.Update_Progress_Description.emit("Creating Batches")

        print(self.kindBatch)

        if (self.kindBatch == 'Training'):
            for i in range(self.n_past, len(df_for_training_scaled) - n_future +1):
                trainX.append(df_for_training_scaled[i - self.n_past:i, 0:df_for_training.shape[1]])
                #trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])
                #trainY_Open.append(df_for_training_scaled[i:i+1, 0:df_for_training.shape[1]])
                trainY_Open.append(df_for_training_Open_scaled[i:i+1, 0:df_for_training.shape[1]])
                trainY_High.append(df_for_training_High_scaled[i:i+1, 0:df_for_training.shape[1]])
                trainY_Low.append(df_for_training_Low_scaled[i:i+1, 0:df_for_training.shape[1]])
                trainY_Close.append(df_for_training_Close_scaled[i:i+1, 0:df_for_training.shape[1]])
                trainY_Volume.append(df_for_training_Volume_scaled[i:i+1, 0:df_for_training.shape[1]])
            
            

            trainX, trainY_Open, trainY_High, trainY_Low =  np.array(trainX), np.array(trainY_Open), np.array(trainY_High), np.array(trainY_Low)
            trainY_Close, trainY_Volume = np.array(trainY_Close), np.array(trainY_Volume)

            trainX=trainX[:len(trainX)]#Why -3
            trainY_Open=trainY_Open[:len(trainY_Open)]#Why -3
            trainY_High=trainY_High[:len(trainY_High)]#Why -3
            trainY_Low=trainY_Low[:len(trainY_Low)]#Why -3
            trainY_Close=trainY_Close[:len(trainY_Close)]#Why -3
            trainY_Volume=trainY_Volume[:len(trainY_Volume)]#Why -3

            print('trainX shape == {}.'.format(trainX.shape))
            print('trainY_Open shape == {}.'.format(trainY_Open.shape))
            print('trainY_High shape == {}.'.format(trainY_High.shape))
            print('trainY_Low shape == {}.'.format(trainY_Low.shape))
            print('trainY_Close shape == {}.'.format(trainY_Close.shape))
            print('trainY_Volume shape == {}.'.format(trainY_Volume.shape))

            self.Update_Progress.emit(70)
            self.Update_Progress_Description.emit("Batches created")
        
        elif (self.kindBatch =='Predict'):
            for i in range(len(df_for_training_scaled)):
                if i >= len(df_for_training_scaled)-self.n_past:
                    trainX_sub.append(df_for_training_scaled[i])
                
            trainX.append(trainX_sub)
            
            trainX=np.array(trainX)
            trainX=trainX[:len(trainX)]
            print('trainX shape == {}.'.format(trainX.shape))
            

        #Creating the numpy files
        #parent_directory="/home/eduardo/Desktop/SPP_deep_learning/BatchDataGenerator_Qt_file/All_Batches_TWTR/"
        parent_directory=self.directoryPath
        directory=self.startDate+"-TO-"+self.toDate

        #Index file's paths

        #--------verify if already exist, or create a new one

        NameIndexDirectoryPathsFIles=parent_directory+"IndexFile.csv"


        try:
            with open(NameIndexDirectoryPathsFIles, newline='') as csvfile:
                TestFIleCSV= csv.reader(csvfile, delimiter=' ', quotechar='|')
                TestFIleCSV=list(TestFIleCSV)
                print(type(TestFIleCSV))
        except:
            print("Was an exeption")
            Indexdf=numpy.asarray([])
            numpy.savetxt(NameIndexDirectoryPathsFIles, Indexdf, delimiter=",")
            with open(NameIndexDirectoryPathsFIles, newline='') as csvfile:
                TestFIleCSV= csv.reader(csvfile, delimiter=' ', quotechar='|')
                TestFIleCSV=list(TestFIleCSV)
                print(type(TestFIleCSV))

        #---- Creating new folder -----
        #Folders name must contain the periodes time the batches are made of
        def creating_directory(parent_dir,directory):
            # Path
            path = os.path.join(parent_dir, directory)
            access_rights = 0o755
            # Create the directory
            # 'GeeksForGeeks' in
            # '/home / User / Documents'
            os.mkdir(path,access_rights)
            print("Directory '% s' created" % directory)

            return path

        

        #----Saving numpy files ----------------------
        def savingNumpyArrays(Path,NameNumpyArray, NumpyArray):
            Final_Path=Path+"/"+NameNumpyArray+".npy"
            np.save(Final_Path, NumpyArray)
            return Final_Path

        #  Deleting directory
        def DeleteFolder(parent_dir,directory):
            path = parent_dir+directory
            try:
                shutil.rmtree(path)
            except:
                print ("Deletion of the directory %s failed" % path)
            else:
                print ("Successfully deleted the directory %s" % path)


        self.Update_Progress.emit(80)
        self.Update_Progress_Description.emit("verifying directories and creating them")
        #Before to create a new Folder, need to ensure there is an existing with same name
        DeleteFolder(parent_directory,directory)# Can not deleted if contains some thing

        #Now it's possible to create a directory
        newpath=creating_directory(parent_directory,directory)#Function is called
        print(newpath)

        #Creating numpyfiles
        self.Update_Progress.emit(90)
        self.Update_Progress_Description.emit("Saving data batches")

        # --------- trainX numpy array
        if (self.kindBatch == 'Training'):
            PathFile_TrainX=savingNumpyArrays(newpath,"trainX",trainX)

            # --------- trainY_Open numpy array

            

            PathFile_TrainY_Open=savingNumpyArrays(newpath,"trainY_Open",trainY_Open)

            # ---------trainY_High numpy array

            PathFile_TrainY_High=savingNumpyArrays(newpath,"trainY_High",trainY_High)

            # ---------trainY_Low numpy array

            PathFile_TrainY_Low=savingNumpyArrays(newpath,"trainY_Low",trainY_Low)


            # --------- trainY_Close numpy array

            PathFile_TrainY_Close=savingNumpyArrays(newpath,"trainY_Close",trainY_Close)


            # --------- trainY_Volume numpy array

            PathFile_TrainY_Volume=savingNumpyArrays(newpath,"trainY_Volume",trainY_Volume)
        if (self.kindBatch =='Predict'):
            PathFile_TrainX_Predict=savingNumpyArrays(newpath,"trainX_Predict",trainX)
            self.LastBatchPathPredict.emit(PathFile_TrainX_Predict)


        
        #-------------- Save PathFile ------------------------------
        # open the file in the write mode

        self.Update_Progress.emit(95)
        self.Update_Progress_Description.emit("Saving index paths data batches files")
        if (self.kindBatch == 'Training'):
            TestFIleCSV.append([PathFile_TrainX])
            TestFIleCSV.append([PathFile_TrainY_Open])
            TestFIleCSV.append([PathFile_TrainY_High])
            TestFIleCSV.append([PathFile_TrainY_Low])
            TestFIleCSV.append([PathFile_TrainY_Close])
            TestFIleCSV.append([PathFile_TrainY_Volume])
            
        if (self.kindBatch =='Predict'):
            TestFIleCSV.append([PathFile_TrainX_Predict])
        
        TestFIleCSV.append(["@"])



        with open(NameIndexDirectoryPathsFIles, 'w') as f:
            # create the csv writer
            writer = csv.writer(f)
            # write a row to the csv file
            writer.writerows(TestFIleCSV)

        self.Update_Progress.emit(100)
        self.Update_Progress_Description.emit("Done")
