from scipy.optimize import curve_fit
from scipy.misc import factorial
import pickle
import numpy as np


def approximate_distribution():
    def exponential(x, mu, lamb, A):
        return A*np.exp(-(x-mu) * lamb)

    def gauss(x,mu,sigma,A):
        return A*np.exp(-(x-mu)**2/2/sigma**2)

    def poisson(k, lamb, A):
        return A * (lamb**k/ factorial(k)) * np.exp(-lamb)

    def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
        return gauss(x,mu1,sigma1,A1) + gauss(x,mu2,sigma2,A2)

    def exponential_gauss(x, mu1, sigma1, A1, mu2, sigma2, A2):
        return exponential(x, mu1, sigma1, A1) + gauss(x, mu2, sigma2, A2)

    def estimate_parameters(model, x, y):
        params, cov = curve_fit(model, x, y)
        sigma = np.sqrt(np.diagonal(cov))
        return params, sigma

    range = [0.03, 1]
    bins = 100
    for w in wglo:
        data = w.flatten()
        y, x = np.histogram(data, bins=bins, range=range)
        x = [(a+b)/2 for a, b in zip(x[:-1],x[1:])]
        guess = [0, 0, 0, 0, 0, 0]
        params, sigma = estimate_parameters(gauss, x, y)
        print(params)
        plt.plot(x,gauss(x,*params[:3]),color='red',lw=3,label='model')
        # plt.plot(x,gauss(x,*params[-3:]),color='green',lw=3,label='model')
        plt.hist(data,bins=bins, range=range)
        plt.show()