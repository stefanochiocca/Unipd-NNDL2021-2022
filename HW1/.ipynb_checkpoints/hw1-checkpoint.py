# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 22:01:39 2021

@author: stefa
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt

def plot_data(data, label=''):
    fig = plt.figure(figsize=(12,8))
    plt.scatter(data.input, data.label, label=label)
    plt.xlabel('input')
    plt.ylabel('label')
    plt.legend()
    plt.show()


train_data = pd.read_csv('regression_dataset/train_data.csv')
test_data = pd.read_csv('regression_dataset/test_data.csv')

x_train = np.ndarray(len(train_data['input'])) 
y_train = np.ndarray(len(train_data['label']))

x_test = np.ndarray(len(test_data['input'])) 
y_test = np.ndarray(len(test_data['label']))

#store the training /test data into ndarrays

for sample_index in range(len(train_data)):
    x_train[sample_index] = train_data.iloc[sample_index]['input']
    y_train[sample_index] = train_data.iloc[sample_index]['label']

for sample_index in range(len(test_data)):
    x_test[sample_index] = test_data.iloc[sample_index]['input']
    y_test[sample_index] = test_data.iloc[sample_index]['label']
    
    
plot_data(train_data, 'Training data')
plot_data(test_data, 'Test data')

class CsvDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        
        self.transform = transform
        # Read the file and split the lines in a list
        with open(csv_file, 'r') as f:
            lines = f.read().split('\n')
        lines.pop(0)
        lines.pop(-1)
        # Get x and y values from each line and append to self.data
        self.data = []
        for line in lines:
            sample = line.split(',')
            self.data.append((float(sample[0]), float(sample[1])))
            # Now self.data contains all our dataset.
        # Each element of the list self.data is a tuple: (input, output)
    def __len__(self):
        # The length of the dataset is simply the length of the self.data list
        return len(self.data)

    def __getitem__(self, idx):
        # Our sample is the element idx of the list self.data
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
    
class ToTensor():
        """Convert sample to Tensors."""
        def __call__(self, sample):
            x, y = sample
            return (torch.tensor([x]).float(), torch.tensor([y]).float())
    
    
composed_transform = transforms.Compose([ToTensor()])

train_dataset = CsvDataset('regression_dataset/train_data.csv', transform=composed_transform)
test_dataset = CsvDataset('regression_dataset/test_data.csv', transform=composed_transform)
    
print(train_dataset.__getitem__(30))

class Net(nn.Module):
    
    def __init__(self, Ni, Nh1, Nh2, No):
        """
        Ni - Input size
        Nh1 - Neurons in the 1st hidden layer
        Nh2 - Neurons in the 2nd hidden layer
        No - Output size
        """
        super().__init__()
        
        print('Network initialized')
        self.fc1 = nn.Linear(in_features=Ni, out_features=Nh1)
        self.fc2 = nn.Linear(in_features=Nh1, out_features=Nh2)
        self.out = nn.Linear(in_features=Nh2, out_features=No)
        self.act = nn.Sigmoid()
        
    def forward(self, x, additional_out=False):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.out(x)
        return x
    
# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Training device: {device}")

torch.manual_seed(0)
Ni = 1
Nh1 = 128
Nh2 = 256
No = 1
net = Net(Ni, Nh1, Nh2, No)
net.to(device)

loss_fn = nn.MSELoss()  
optimizer = optim.Adam(net.parameters(), lr = 1e-3)

### TRAINING LOOP
num_epochs = 300
train_loss_log = []
val_loss_log = []

my_images = []
fig, ax = plt.subplots(figsize=(12,7))

for epoch_num in range(num_epochs):
    print('#################')
    print(f'# EPOCH {epoch_num}')
    print('#################')

    ### TRAIN
    train_loss= []
    net.train() # Training mode (e.g. enable dropout, batchnorm updates,...)
    for sample_batched in train_dataset:
        # Move data to device
        x_batch = sample_batched[0].to(device)
        label_batch = sample_batched[1].to(device)

        # Forward pass
        out = net(x_batch)

        # Compute loss
        loss = loss_fn(out, label_batch)

        # Backpropagation
        net.zero_grad()
        loss.backward()

        # Update the weights
        optimizer.step()

        # Save train loss for this batch
        loss_batch = loss.detach().cpu().numpy()
        train_loss.append(loss_batch)
        
        plt.cla()
        ax.set_title('Regression Analysis', fontsize=35)
        ax.set_xlabel('Independent variable', fontsize=24)
        ax.set_ylabel('Dependent variable', fontsize=24)
        ax.set_xlim(-1.05, 1.5)
        ax.set_ylim(-0.25, 1.25)
        ax.scatter(x_batch.data.numpy(), label_batch.data.numpy(), color = "orange")
        ax.plot(x_batch.data.numpy(), out.data.numpy(), 'g-', lw=3)
        ax.text(1.0, 0.1, 'Step = %d' % epoch_num, fontdict={'size': 24, 'color':  'red'})
        ax.text(1.0, 0, 'Loss = %.4f' % loss.data.numpy(),
                fontdict={'size': 24, 'color':  'red'})
        
        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    my_images.append(image)

    # Save average train loss
    train_loss = np.mean(train_loss)
    print(f"AVERAGE TRAIN LOSS: {train_loss}")
    train_loss_log.append(train_loss)
