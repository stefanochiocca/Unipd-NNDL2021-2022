# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 10:33:24 2021

@author: stefa
"""

import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import helper



transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.repeat(3,1,1)),
                                transforms.Normalize((0.5, ), (0.5,))])

train_dataset = FashionMNIST('classifier_data', train=True, download=True)
test_dataset  = FashionMNIST('classifier_data', train=False, download=True)

#sample_index = 19
#image = train_dataset[sample_index][0]
#label = train_dataset[sample_index][1]

#fig = plt.figure(figsize=(8,8))
#plt.imshow(image, cmap='Greys')
#print(f"SAMPLE AT INDEX {sample_index}")
#print(f"LABEL: {label}")

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.repeat(3,1,1)),
                                transforms.Normalize((0.5, ), (0.5,))])

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

image, label = next(iter(train_dataloader))
helper.imshow(image[0,:]);




