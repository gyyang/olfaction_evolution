"""Test the performance of a linear oracle."""

import os

import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    """Compute softmax along axis 1."""
    e_x = np.exp(x - np.max(x))
    return (e_x.T / e_x.sum(axis=1)).T

path = '../datasets/proto/standard'


class OracleAnalysis():
    
    def __init__(self):
        names = ['train_x', 'train_y', 'val_x', 'val_y', 'prototype']
        results = [np.load(os.path.join(path, name + '.npy')) for name in names]
        train_x, train_y, val_x, val_y, prototype = results
        
        # print(np.unique(train_y, return_counts=True))
        
        # Oracle network
        self.data_x, self.data_y = val_x, val_y

        self.w_oracle = 2 * prototype.T
        self.b_oracle =  -np.diag(np.dot(prototype, prototype.T))

    def compute_loss(self, noise=0, alpha=1, orn_dropout_rate=0):
        data_x = self.data_x + np.random.randn(*self.data_x.shape) * noise

        dropout_mask = np.random.rand(*data_x.shape) > orn_dropout_rate
        data_x = data_x * dropout_mask

        y = np.dot(data_x, self.w_oracle) + self.b_oracle

        y = alpha * y

        y = softmax(y)
        
        output = np.argmax(y, axis=1)
        
        correct = self.data_y == output
        loss = -np.log(y[np.arange(len(self.data_y)), self.data_y])
        
        acc = np.mean(correct)
        loss = np.mean(loss)
        
        return acc, loss

    def get_losses_by_alphas(self, alphas, noise=0):
        # alphas = np.linspace(1, 6, 20)
        accs = list()
        losses = list()
        for alpha in alphas:
            acc, loss = self.compute_loss(noise=noise, alpha=alpha)
            accs.append(acc)
            losses.append(loss)

        plt.figure()
        plt.plot(alphas, losses)
        
        plt.figure()
        plt.plot(alphas, accs)

    def get_losses_by_noise(self, noises, alpha=1):
        accs = list()
        losses = list()
        for noise in noises:
            acc, loss = self.compute_loss(noise=noise, alpha=alpha)
            accs.append(acc)
            losses.append(loss)

        return accs, losses

    def get_losses_by_dropout(self, rates, alpha=1):
        accs = list()
        losses = list()
        for rate in rates:
            acc, loss = self.compute_loss(orn_dropout_rate=rate, alpha=alpha)
            accs.append(acc)
            losses.append(loss)

        return accs, losses

    def get_losses_by_noisealpha(self, noises, alphas):
        accs = list()
        losses = list()
        for noise in noises:
            for alpha in alphas:
                acc, loss = self.compute_loss(noise=noise, alpha=alpha)
                accs.append(acc)
                losses.append(loss)

        accs = np.array(accs)
        losses = np.array(losses)

        shape = (len(noises), len(alphas))
        accs = np.reshape(accs, shape)
        losses = np.reshape(losses, shape)

        return accs, losses


if __name__ == '__main__':
    oa = OracleAnalysis()

    rates = np.linspace(0, 0.9, 100)
    accs_oracle, losses_oracle = oa.get_losses_by_dropout(rates)

    plt.figure()
    plt.plot(rates, accs_oracle, 'o-', color='black')
    plt.xlabel('ORN drop out rate')
    plt.ylabel('Acc')