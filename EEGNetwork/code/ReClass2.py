#!/usr/bin/env python
# coding: utf-8

# In[2]:

#Contains classes used for object-oriented programming
#Classes contain data on each subject and train models for ML
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import tensorflow as tf
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import scipy.stats as st
import multiprocessing as mp
import gc
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session


#Class for patient data
class Subject:
    
    def __init__(self, cogState = 'Unknown', data = [], index = 0):
        self.cogState = cogState #Cognitive state for subject i
        self.data = data #continuous data for subject i
        self.index = index #index value of subject i in the data set
    
    def setPredState(self, predState):
        self.predState = predState #predicted state for subject i based on regression model


#Class for model training
class TSModel:
    
    def __init__(self, cogStates):
        self.cogStates = cogStates #list of subjects and their info
        self.chModel = [] #list of models -> Used for Approach1, i.e. channel-to-channel reconstruction
        self.predictions = [] #list of predictions -> Used for Approach1, i.e. channel-to-channel reconstruction
        self.matrixModel = None #matrix of models -> Used for Approach2, i.e. network reconstruction
        self.matrixPred = None #matrix of prediction -> Used for Approach2, i.e. network reconstruction 
     
    #Compact channels in cross validation into one continuous signal
    def compactChannels(self, i = None):
        self.channels = np.asarray([np.concatenate([x.data[:,z] for x in self.cogStates], axis=None)\
                                    for z in range(30)]).transpose() 
    
    #The following 4 functions are used for Approach1: channel-to-channel reconstruction
    #Create model for Approach1 -> channel-to-channel reconstruction
    #Input layer is equal to the amount of channels one subject has
    #The hidden layer consists of 15 neurons and the output layer has 1 neuron
    def createModel(self):
        classifier = Sequential()
        classifier.add(Dense(units = 15, kernel_initializer = 'normal', activation = 'relu',\
                             input_dim = len(self.channels[0,:])-1))
        classifier.add(Dense(units = 1, activation = 'linear'))

        classifier.compile(optimizer = 'adadelta', loss='mean_squared_error', metrics=['mse'])
        return classifier
    
    #Train the model 
    #Use a right shifting method to create 29 input variables for the model
    #Store the data in a list to be used for predictions
    def trainModel(self):
        self.compactChannels()
        print("Training....")
        
        for i in range(30):
            
            x_train = self.channels[:, i]
            y_train = self.channels[:, [x for x in range(30) if x != i]]
            tmp=x_train
            for m in range(28):
                x_train= np.pad(x_train,(1,0),mode='constant')[:-1]
                tmp=np.vstack((tmp,x_train))
            x_train = tmp
            self.classifier = self.createModel()

            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            y_train = scaler.fit_transform(y_train.reshape(-1,1))

            self.classifier.fit(x_train, y_train, batch_size = y_train.size, epochs=25, verbose=0)
            self.chModel.append(self.classifier)
        print("Done Training.")
           
    #Reconstruct the data and store the results in a list  
    #List is used for calculating correlation coefficients
    def predictModels(self, testData):
        print("Predicting......")
        for i in range(len(self.chModel)):
            test = testData.data[:, [x for x in range(30) if x != i]]
            self.predictions.append(self.chModel[i].predict(x=test))
        print("Done predicting.")
     
    #Calculate the correlation coefficient betweent the original and reconstructed data
    #Save the data in a text file for later use -> data is sequential and not formatted
    #into a matrix
    def calc_CCof(self, testData, x):
        self.cof=[]
        for i in range(len(self.chModel)):
            
            self.cof.append(st.pearsonr(testData.data[:, i].transpose(),\
                                        self.predictions[i].transpose().flatten('C')))
            
        with open("Sub-%d.txt" % (testData.index+1), "a+") as f:
            f.write("These are correlation coefficients based on subject %d for the %s-Model\n\n" % \
                    (testData.index+1, x))
            for item in self.cof:
                f.write("-\t (%f, %f)\n" % (item[0], item[1]))
            f.write("\n\n")
    
    
    #The following 4 functions are used for Approach2: network reconstruction
    #Using tensorflow instead of Keras to speed up training
    #Similar to the createModel() function above, but has 50 input neurons
    def createMatrixModel(self):
        
        classifier = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(50,)),
            tf.keras.layers.Dense(15, kernel_initializer=tf.initializers.random_normal, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)])
        
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.0001)
        classifier.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])
        return classifier
    """ This function is used when one has access to the GPU
        Flushes out memory to speed up model training
        
    def reset_keras(self):
        sess = get_session()
        clear_session()
        sess.close()
        sess = get_session()
  
        try:
            del classifier
        except:
            pass
  
        print(gc.collect())

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1
        config.gpu_options.visible_device_list = "0"
        set_session(tf.Session(config=config))"""
    
    #Creates our causality matrix of 30x30 models
    #When a model_type is specified, causality matrix for that 
    #cognitive group is saved in a .json and .h5py file
    #This is to avoid rebuilding the matrix with every run of the code
    #Otherwise, if no model_type is given, a causality matrix for one 
    #subject is created
    def trainMatrixModel(self, model_type=None):
        
        self.compactChannels()
        print("Training....")
        self.count = 0
        self.l = []
        for i in range(30):
            self.y_train = self.channels[:, i]
            for j in range(30):           
                self.x_train = self.channels[:, j]
                self.tmp=self.x_train
                for m in range(49):
                    self.x_train= np.pad(self.x_train,(1,0),mode='constant')[:-1]
                    self.tmp=np.vstack((self.tmp,self.x_train))
                self.x_train = self.tmp
                scaler = StandardScaler()
                self.x_train = scaler.fit_transform(np.transpose(self.x_train))
                self.y_train = scaler.fit_transform(self.y_train.reshape(-1,1))
                
                
                self.classifier = self.createMatrixModel()
                self.classifier.fit(self.x_train, self.y_train, batch_size = self.y_train.size, epochs=20, verbose=0)
                self.l.append(self.classifier)
                
                if model_type != None:
                    model = self.classifier
                    model_json = model.to_json()
                    with open("~/spark-2019/data/model-%s-%d-%d.json" % (model_type,i,j), "a+") as json_file:
                        json_file.write(model_json)
                    # serialize weights to HDF5
                    model.save_weights("~/spark-2019/data/model-%s-%d-%d.h5py" % (model_type,i,j))

            if self.count == 0:
                self.matrixModel = [np.asarray(self.l)]
                self.count = 1
            else:
                self.tmp = np.asarray(self.l)
                self.matrixModel = np.append(self.matrixModel, [self.tmp], axis=0)
            self.l.clear()

        print("Done Training.")
     
    #Similar to predictModels() above, but organizes the data into a matrix instead of a list
    def predictMatrixModels(self, matmod, testData):
        print("Predicting......")
        self.k = 0
        
        for i in range(30):
            for j in range(30):
                self.x_train = testData.data[:, i]
                self.tmp=x_train
                for m in range(49):
                    self.x_train= np.pad(self.x_train,(1,0),mode='constant')[:-1]
                    self.tmp=np.vstack((self.tmp,self.x_train))
                self.x_train = self.tmp
                
                self.pred = matmod[i][j].predict(x=self.x_train.transpose())
                self.predictions.append(self.pred)
            if self.k == 0:
                self.matrixPred = [np.asarray(self.predictions)]
                self.k = 1
            else:
                self.tmp = np.asarray(self.predictions)
                self.matrixPred = np.append(self.matrixPred, [self.tmp], axis=0)
            self.predictions.clear()
        print("Done predicting.")

    #Similar to calc_CCof() above, but is for the causality matrix
    def calcMatrix_CCof(self, testData, x):
        self.cof=[]
        for i in range(30):
            for j in range(30):
                self.cof.append(st.pearsonr(testData.data[:, j].transpose(),\
                                        self.matrixPred[i, j].transpose().flatten('C')))
            
        with open("Sub-new-%d.txt" % (testData.index+1), "a+") as f:
            f.write("These are the new correlation coefficients based on subject %d for the %s-Model\n\n" % \
                    (testData.index+1, x))
            for item in range(len(self.cof)):
                f.write("-\t (%f)\n" % (self.cof[item][0]))
                if (item+1) % 30 == 0:
                  
                    f.write("\n\n")

    
    
    
    
    
    
    
