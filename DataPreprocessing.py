# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 01:03:28 2018

@author: Shashank
"""
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)


def rescale_data(data):
    train_data = data['mid_prices'][:2800].as_matrix()
    test_data = data['mid_prices'][2800:].as_matrix()
    
    scaler = MinMaxScaler()
    train_data = train_data.reshape(-1,1)
    test_data = test_data.reshape(-1,1)
    smoothing_window_size = 700
    for di in range(0,len(train_data),smoothing_window_size):
        scaler.fit(train_data[di:di+smoothing_window_size,:])
        train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])
    train_data = train_data.reshape(-1)
    test_data = scaler.transform(test_data).reshape(-1)
    EMA = 0.0
    gamma = 0.1
    for ti in range(len(train_data)):
      EMA = gamma*train_data[ti] + (1-gamma)*EMA
      train_data[ti] = EMA
    all_mid_data = np.concatenate([train_data, test_data],axis=0)
    return train_data, test_data