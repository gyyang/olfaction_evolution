"""Test the performance of a linear oracle."""

import os

import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    """Compute softmax along axis 1."""
    e_x = np.exp(x - np.max(x))
    return (e_x.T / e_x.sum(axis=1)).T

path = '../datasets/proto/test_data'

names = ['train_x', 'train_y', 'val_x', 'val_y', 'prototype']
results = [np.load(os.path.join(path, name + '.npy')) for name in names]
train_x, train_y, val_x, val_y, prototype = results

# the 0 class is not used
assert np.min(train_y) == 1
train_y -= 1
val_y -= 1

# print(np.unique(train_y, return_counts=True))

# Oracle network
data_x, data_y = val_x, val_y
y = 2*np.dot(data_x, prototype.T) - np.diag(np.dot(prototype, prototype.T))


def compute_loss(y):
    y = softmax(y)
    
    output = np.argmax(y, axis=1)
    
    correct = data_y == output
    loss = -np.log(y[np.arange(len(data_y)), data_y])
    
    acc = np.mean(correct)
    loss = np.mean(loss)
    
    return acc, loss


alphas = np.linspace(1, 6, 20)
accs = list()
losses = list()
for alpha in alphas:
    acc, loss = compute_loss(alpha*y)
    accs.append(acc)
    losses.append(loss)
    

plt.figure()
plt.plot(alphas, losses)

plt.figure()
plt.plot(alphas, accs)


