# DL_Package_TimeSeries_LSTM
Dataset:
The dataset is taken from yahoo finace's website in CSV format. The dataset consists of Open, High, Low and Closing Prices of any given stocks from 1 January 2015 to 1 January 2020.

Price Indicator:
Stock traders mainly use three indicators for prediction: OHLC average (average of Open, High, Low and Closing Prices), HLC average (average of High, Low and Closing Prices) and Closing price, In this project, OHLC average has been used.

Data Pre-processing:
After converting the dataset into a single array, it becomes one column data. This has been converted into two column time series data, 1st column consisting stock price of time t, and second column of time t+1. All values have been normalized between 0 and 1.

Model:
Two sequential LSTM layers have been stacked together and one dense layer is used to build the RNN model using Keras deep learning library. Since this is a regression task, 'linear' activation has been used in final layer.

Version:
Python 3.4 and latest versions of all libraries including deep learning library Keras and Tensorflow.

Training:
75% data is used for training. Adagrad (adaptive gradient algorithm) optimizer is used for faster convergence.

Test:
Test accuracy metric is root mean square error (RMSE).
