from inspect import GEN_CLOSED
from json.tool import main
import pandas as pd
import numpy as np
import json
from load import fetchModel
import base64
import os
from tensorflow.keras.utils import to_categorical

def SendModel(modelString):

    try:
        with open("Models/model.h5","wb") as file:
            file.write(base64.b64decode(modelString))
            print("Successfully saved model...")
            return 0
    except Exception as e:
        print(e)
        print("An error occured!")
        return 1

def saveLearntMetrice(file_name,score):

    with open(file_name,'r+') as f:
        trainMetrics = json.load(f)
        trainMetrics['accuracy'].append(score[1])
        trainMetrics['loss'].append(score[0])
        f.seek(0) 
        f.truncate()
        f.write(json.dumps(trainMetrics))

# The function is used for training local model based on private data
def Train():
    

    print("Starting training...")

    continousTrainingBatchSize = 60
    #Reading index to simulate continous learning
    currentIndex = 0
    try :

        with open('data/indexFile.txt', "r+") as f:
            fileIndex = json.load(f)
            currentIndex = fileIndex['index']

        print("Current Index is ", currentIndex)

        data = pd.read_csv('data/data.csv', on_bad_lines='skip')
        totalRowCount = data.shape[0]
        nextIndex = currentIndex + continousTrainingBatchSize if currentIndex + continousTrainingBatchSize < totalRowCount else totalRowCount


        X = data.iloc[currentIndex:nextIndex,1:-1].values
        y = data.iloc[currentIndex:nextIndex,-1].values
        y = to_categorical(y)


        #Updating Index
        if nextIndex == totalRowCount:
            nextIndex = 0
        with open('data/indexFile.txt', "w") as f: 
            index = {'index' : nextIndex}
            f.write(json.dumps(index))

        #get latest model from own directory
        model = fetchModel()
        #model.summary()

        #Printing aggregated global model metrics
        score = model.evaluate(X, y, verbose=0)
        print("Global model loss : {} Global model accuracy : {}".format(score[0], score[1]))
        
        saveLearntMetrice('data/metrics.txt', score)
        

        model.fit(X, y, epochs=16, verbose=0)
            
        #Printing loss and accuracy after training 
        score = model.evaluate(X, y, verbose=0)
        print("Local model loss : {} Local model accuracy : {}".format(score[0], score[1]))
        
        saveLearntMetrice('data/localMetrics.txt', score)

        #Save current model 
        model.save('Models/model.h5')
        with open('Models/model.h5','rb') as file:
            encoded_string = base64.b64encode(file.read())
        
        return encoded_string
    except Exception as e:
        print(e)
        print("An error occured!")
        return 0
def initilizeDevice():

    metric = {'accuracy' : [], 'loss' : []}
    index = {'index' : 0}

    with open('data/indexFile.txt', "w") as f:
        f.write(json.dumps(index))
    with open('data/metrics.txt', "w") as f:
        f.write(json.dumps(metric))
    with open('data/localMetrics.txt', "w") as f:
        f.write(json.dumps(metric))

    
#This function is used fro dataset generation
def GenerateData():
    initilizeDevice()
    print("Devices initilization done")
    return 0
    