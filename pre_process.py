# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:31:24 2020

@author: Afreen Kazi
"""
import numpy as np 


#Read data
import numpy as np
import pandas as pd
import datetime
import pandas_datareader.data as web

def fetch_close_price(stock_name):
    start = datetime.datetime(2015,1,1)
    end = datetime.datetime(2020,1,1)
    comp = web.DataReader([stock_name], 'yahoo', start, end)
    #getting close columns
    comp = comp['Close']
    comp = np.array(comp)
    return comp

data = fetch_close_price('TSLA')


# FUNCTION TO CREATE 1D DATA INTO TIME SERIES DATASET
def new_dataset(dataset, step_size):
	data_X, data_Y = [], []
	for i in range(len(dataset)-step_size-1):
		a = dataset[i:(i+step_size), 0]
		data_X.append(a)
		data_Y.append(dataset[i + step_size, 0])
	return np.array(data_X), np.array(data_Y)

data_X, data_Y = new_dataset(data, 1)

# THIS FUNCTION CAN BE USED TO CREATE A TIME SERIES DATASET FROM ANY 1D ARRAY	
data = data.reindex(index = data.index[::-1])
def pre_process(data):
    # PREPARATION OF TIME SERIES DATASE
    from sklearn.preprocessing import MinMaxScaler
    data= np.reshape(data, (len(data),1)) # 1664
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    return data
    
preprocess_X = pre_process(data_X)
preprocess_Y = pre_process(data_Y)

def test_train_split(series):
    train = int(len(series) * 0.75)
    test = len(series) - train
    train, test = series[0:train,:],series[train:len(series),:]
    return train, test

train_X, test_X = test_train_split(preprocess_X)
train_Y, test_Y = test_train_split(preprocess_Y)

def reshape(train, test):
    train = np.reshape(train, (train.shape[0], 1, train.shape[1]))
    test = np.reshape(test, (test.shape[0], 1, test.shape[1]))
    return train, test

train_X_r, test_X_r = reshape(train_X, test_X) 
train_Y_r, test_Y_r= reshape(train_Y, test_Y) 

    

    



    