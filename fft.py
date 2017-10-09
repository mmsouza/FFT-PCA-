import pandas as pd
import numpy as np
from scipy.signal import detrend
from scipy.fftpack import fft
import matplotlib.pyplot as plt


def __reject_outliers__(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def hf_fft(data):
    n = len(data)
    t = 1.0 / (len(data))

    yf = fft(detrend(data))
    xf = np.linspace(0, 1.0 / (2.0 * t), n // 2)
    yff = n * np.abs(yf[0:n // 2])
    return xf, yff, yf


def plot_fft(xf, yff, name, method='plot', save=False):

    p = plt.figure()
    p.canvas.set_window_title(name)

    if str.lower(method) == 'plot':
        plt.plot(xf, yff)
    if str.lower(method) == 'scatter':
        plt.scatter(xf, yff)
    if save:
        plt.savefig(name)
    return