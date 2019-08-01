#Code for Approach2: network reconstruction
import sys
import json
import numpy as np
from scipy.io import loadmat
from ReClass_for_right_shift import Subject, TSModel
import tensorflow as tf
import glob
import keras
from keras.models import model_from_json
import keras.backend as K

if __name__ == '__main__':
  
    subjects = [] #Creating a list of Subject classes to keep track of participant information
    sN = int(sys.argv[1]) #Get ID of subject for testing -> ID for training set
    print(sys.argv[1]) #check if it's the right ID

    #separate the .mat files needed from the directories
    #import subject data files
    subjectData = sorted(glob.glob('/home/user1/spark-2019/data/rest1 1min/rec*.mat'))
    subjectData = [x[39:] for x in subjectData]
    
    
    #Loop through each file and save the subjects data, state, and channels in a Class Object
    for index,value in enumerate(subjectData):
        
        subjects.append(Subject(loadmat('/home/user1/spark-2019/data/headerInfo/'+value)['dx'][0],\
                                loadmat('/home/user1/spark-2019/data/rest1 1min/'+value)['data'],\
                                index))

    
    #Separate the subjects into three training sets based on cognitive state
    models = []

    models.append(TSModel([x for x in subjects if x.cogState == 'N']))
    models.append(TSModel([x for x in subjects if x.cogState == 'MCI']))
    models.append(TSModel([x for x in subjects if x.cogState == 'AD']))

    #Import a list of network models in numerical order
    #Import the corresponding weights for each model into a separate list
    MCI_models = []
    MCI_weights = []
    AD_models=[]
    AD_weights = []
    NC_models =[]
    NC_weights =[]

    for i in range(30):
        MCI_models += sorted(glob.glob("/home/user1/spark-2019/data/MCI_models/model-MCI-%d-?.json" %i))+sorted(glob.glob("/home/user1/spark-2019/data/MCI_models/model-MCI-%d-??.json"%i))
        MCI_weights += sorted(glob.glob("/home/user1/spark-2019/data/MCI_models/model-MCI-%d-?.h5py"%i))+sorted(glob.glob("/home/user1/spark-2019/data/MCI_models/model-MCI-%d-??.h5py"%i))
    
    for i in range(30):
        AD_models += sorted(glob.glob("/home/user1/spark-2019/data/AD_model/model-AD-%d-?.json" %i))+sorted(glob.glob("/home/user1/spark-2019/data/AD_model/model-AD-%d-??.json"%i))
        AD_weights += sorted(glob.glob("/home/user1/spark-2019/data/AD_model/model-AD-%d-?.h5py"%i))+sorted(glob.glob("/home/user1/spark-2019/data/AD_model/model-AD-%d-??.h5py"%i))
    
    for i in range(30):
        NC_models += sorted(glob.glob("/home/user1/spark-2019/data/NC_model/model-%d-?.json" %i))+sorted(glob.glob("/home/user1/spark-2019/data/NC_model/model-%d-??.json"%i))
        NC_weights += sorted(glob.glob("/home/user1/spark-2019/data/NC_model/model-%d-?.h5py"%i))+sorted(glob.glob("/home/user1/spark-2019/data/NC_model/model-%d-??.h5py"%i))

        
    #Load the models obtained from above
    #Since the files are .json, we need to compile the models so that we
    #don't have just the structure of the model
    def recreateMatrix(models, weights, p, state=None,):
        count = 0
        k = 0
        channel = 0
        l = []
        matrix = None
        try:
            for i in range(len(models)):
                count += 1
                
                #Some models were stored differently than others
                #If a state is specified, then that model was stored just fine
                #Otherwise, a different way of the loading the model
                #was need in order to load all of them correctly
                if state != None:
                    with open(models[i], "r") as file: 
                        json_l = json.load(file)
                        json_string = json.dumps(json_l)
                else:
                    with open(models[i], mode="r",encoding="ISO-8859-1") as f:
                        json_l = f.readlines()[-1]
                        start = json_l.index("class_name")
                        json_l = json_l[(start-2):]
                        end = json_l.index("tensorflow")+11
                        json_string = json_l[:end+1]
                model = keras.models.model_from_json(json_string)
                model.load_weights(weights[i])
                l.append(model)
                #Delete the model once loaded to free up memory
                del model
                
                #Reorganize the models into the 30x30 causality matrix
                if count == 30:
                    print("%d for subject %d" % (i/30, p))
                    if k == 0:
                        matrix = [np.asarray(l)]
                        k = 1
                    else:
                        tmp = np.asarray(l)
                        matrix = np.append(matrix, [tmp], axis=0)
                    l.clear()
                    count = 0
                    channel += 1
        #If anything fails, print the channel and model in which it failed for easier debugging            
        except ValueError:
            print(channel)
            print(i)
            print(json_string)
            
        return matrix

    #When the models are loaded, they need to be placed in the same graph in tensorflow in order for them to work
    #Set the default graph so that the loaded models will work
    with tf.Graph().as_default() as graph:
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            
            #Load the models
            AD_matrix = recreateMatrix(AD_models, AD_weights, sN)
            MCI_matrix = recreateMatrix(MCI_models, MCI_weights, sN, "MCI")
            NC_matrix = recreateMatrix(NC_models, NC_weights, sN)
            
            #Reconstruct EEG data and calculate correlation coefficients
            #with the data from the loaded models
            models[1].predictMatrixModels(MCI_matrix, models[0].cogStates[int(sN)])
            models[2].predictMatrixModels(AD_matrix, models[0].cogStates[int(sN)])
            models[1].calcMatrix_CCof(models[0].cogStates[int(sN)], "MCI")
            models[2].calcMatrix_CCof(models[0].cogStates[int(sN)], "AD")
            
          
            #Train a new model for each subject in the training set using the data from the n other subjects in the set
            NC_TR = TSModel([x for x in models[0].cogStates if x.cogState == 'N' and models[0].cogStates.index(x) != sN])
            NC_TR.trainMatrixModel()
            NC_TR.predictMatrixModels(NC_TR.matrixModel, models[0].cogStates[int(sN)])
            NC_TR.calcMatrix_CCof(models[0].cogStates[int(sN)], "NC")


