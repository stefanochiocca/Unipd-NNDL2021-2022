# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 22:01:39 2021

@author: stefa
"""

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

torch.manual_seed(0)

train_df = pd.read_csv('regression_dataset/train_data.csv')
test_df = pd.read_csv('regression_dataset/test_data.csv')

x_train = np.ndarray(len(train_df['input'])) 
y_train = np.ndarray(len(train_df['label']))

x_test = np.ndarray(len(test_df['input'])) 
y_test = np.ndarray(len(test_df['label']))

#store the training /test data into ndarrays

for sample_index in range(len(train_df)):
    x_train[sample_index] = train_df.iloc[sample_index]['input']
    y_train[sample_index] = train_df.iloc[sample_index]['label']


for sample_index in range(len(test_df)):
    x_test[sample_index] = test_df.iloc[sample_index]['input']
    y_test[sample_index] = test_df.iloc[sample_index]['label']


class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.hidden = nn.Linear(in_features = n_input, out_features = n_hidden)
        self.predict = nn.Linear(in_features = n_hidden, out_features = n_output)
        
    def forward(self, x):
        x = nn.Sigmoid(self.hidden(x))
        x = self.predict(x)
        return x
    

net = Net(n_input=len(x_train), n_hidden=10, n_output=1)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
loss_func = nn.MSELoss()

# Network training 

x_train_t = torch.from_numpy(x_train)
y_train_t = torch.from_numpy(y_train)

print(x_train_t.shape)
#x_train_t, y_train_t = torch.autograd.Variable(x_train_t), torch.autograd.Variable(y_train_t)

for t in range(len(x_train_t)):
    
    prediction = net(x_train_t[t][0])
    
    loss = loss_func(prediction, y_train_t)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


