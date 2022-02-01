import pandas as pd
import numpy as np
import json
from load import init
import base64
import os
from tensorflow.keras.utils import to_categorical



'''
    This function stores(send) latest model to respective folder.
    It works as model transfer machanism for various participating devices. 
'''
def SendModel(modelString):

    try:
        with open("Models/model.h5","wb") as file:
            file.write(base64.b64decode(modelString))
            print("Successfully saved model...")
            return 1
    except Exception as e:
        print(e)
        print("An error occured!")
        return 0

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
    #get latest model from own directory
    model = init()
    #model.summary()

    #Reading index to simulate continous learning
    currentIndex = 0
    with open('data/indexFile.txt', "r+") as f:
        fileIndex = json.load(f)
        currentIndex = fileIndex['index']

    print("Current Index is ", currentIndex)

    try:
        data = pd.read_csv('data/data.csv')
    except Exception as e:
        print(e)
        return 0
    totalRowCount = data.shape[0]
    nextIndex = currentIndex + continousTrainingBatchSize if currentIndex + continousTrainingBatchSize < totalRowCount else totalRowCount


    X = data.iloc[currentIndex:nextIndex,1:-1].values
    y = data.iloc[currentIndex:nextIndex,-1].values
    y = to_categorical(y)

    #print("Dimension of current data ", X.shape)

    #Updating Index
    if nextIndex == totalRowCount:
        nextIndex = 0
    with open('data/indexFile.txt', "w") as f: 
        index = {'index' : nextIndex}
        f.write(json.dumps(index))


    #Printing aggregated global model metrics
    score = model.evaluate(X, y, verbose=0)
    print("Global model loss : {} Global model accuracy : {}".format(score[0], score[1]))
    
    saveLearntMetrice('data/metrics.txt', score)
    

    model.fit(X, y, epochs=5, verbose=0)
           
    #Printing loss and accuracy after training 
    score = model.evaluate(X, y, verbose=0)
    print("Local model loss : {} Local model accuracy : {}".format(score[0], score[1]))
    
    saveLearntMetrice('data/localMetrics.txt', score)

    #Save current model 
    model.save('Models/model.h5')
    with open('Models/model.h5','rb') as file:
        encoded_string = base64.b64encode(file.read())
    
    return encoded_string

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

    def Sigmoid(z):
        return 1/(1+np.exp(-z))
    
    def eqn(input_vals):

        x1 = input_vals[0]
        x2 = input_vals[1]
        x3 = input_vals[2]
        x4 = input_vals[3]
        
        """Totally random values that the neural network must learn"""
        m1 = 0.5447894
        m2 = 0.007894674
        m3 = 0.012252348#

        y = m1*x1 + m2*x2 + m3*x3 + m4*x4 + b

        return np.round(Sigmoid(y))

    try:
        data = []
        """Generate 1000 data points"""
        for _ in range(10000):
            row = (np.random.sample(4) * np.random.uniform(5,0.0001)) + np.random.uniform(-5,5) # Generate array of 4 numbers
            row = np.append(row, eqn(row))
            data.append(row)
        df = pd.DataFrame(data, columns=['x1','x2','x3','x4','y'])
        df.reset_index()
        df.to_csv('data/data.csv')
        print("Data generated")
        return 1
    except:
        print("An error occured")
        return 0