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
    return xf, yf, yff


def plot_fft(xf, yff, name, method='plot', save=False):

    p = plt.figure()
    p.canvas.set_window_title(name)

    if str.lower(method) == 'plot':
        plt.plot(xf, yff)
    if str.lower(method) == 'scatter':
        plt.scatter(xf, yff)
    if save:
        plt.savefig(name)




if __name__ == "__main__":
    df = pd.read_excel("AirQualityUCI.xlsx")


    xf, yf, yff = hf_fft(df.loc[0: 24, "T"])
    fftserie = pd.Series(yff)
    datafff = pd.DataFrame([list(fftserie)])
    xf, yf, yff = hf_fft(df.loc[24: 48, "T"])

    datafff.loc[1]= np.array(yff)
    print(datafff)

'''
    step= 24
    for n in range(0, 48, step):
        subset_df = df.loc[n: n+step, "T"]
        plt.figure()
        plt.plot(subset_df)

        xf, yf, yff = hf_fft(subset_df)
        print(yff)

        plot_fft(xf, yff, "T"+str(n), method='scatter',save= True)
plt.show()
'''
