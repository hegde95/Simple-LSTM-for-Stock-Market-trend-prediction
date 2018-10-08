# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 08:22:00 2018

@author: Shashank
"""
from keras.models import load_model
import pandas as pd
import numpy as np

import DataPreprocessing as dp
import model as mod


def load_dataCSV():
    df = pd.read_csv('Data/data2.csv')
    return df

def main():
    data = load_dataCSV()
    
    look_back = 28
    jump=4
    
    train_data, test_data = dp.rescale_data(data)
    trainX, trainY = dp.create_dataset(train_data, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX, testY = dp.create_dataset(test_data, look_back)
    
    savedModel = load_model('lstm3.h5')

    pred,perfs=mod.testModel(savedModel,testX,testY,jump,look_back)
    
    actual_test_data=test_data[len(test_data)-len(pred):]

    print("\n Average Covarance between predicted and actual prices on only predicted days:")
    print(np.mean(perfs))
    
    print("\n Covarance between predicted and actual prices on all days:")    
    print(np.cov(actual_test_data,pred)[1][0])

if __name__ == '__main__':
    main()