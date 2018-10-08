# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 01:11:32 2018

@author: Shashank
"""
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import numpy as np

def testModel(model,testX,testY,jump,look_back):
    pred=[]
    perfs=[]
    for i in range(len(testX)):
#        if (i==3500):
#           print("haha") 
        if(i % jump == 0):
            if(i!=0):
                prevInp=inp
            inp=np.reshape(testX[i],(1,1,look_back))
            if(i!=0):
                if(jump<look_back):
                    perfs.append(np.cov(inp[0][0][look_back-jump:],prevInp[0][0][look_back-jump:])[0,1])
                else:
                    perfs.append(np.cov(inp[0][0],prevInp[0][0])[0,1])
        out = model.predict(inp)
        pred.append(out[0][0])
        dum=inp[0][0].tolist()
        dum.append(out[0][0])
        del dum[0]
#        prevInp=inp
        inp=np.reshape(dum,(1,1,look_back))
    return pred,perfs

def getModel(w):
    model = Sequential()
    
    model.add(LSTM(
        input_dim=w,
        output_dim=50,
        return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(
        100,
        return_sequences=False))
    model.add(Dropout(0.2))
    
    model.add(Dense(
        output_dim=1))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer='rmsprop')

    return model

def saveModel(model,name):
    model.save(name+'.h5')