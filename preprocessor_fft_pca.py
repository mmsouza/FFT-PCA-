from sklearn import decomposition
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from scipy.signal import detrend
import matplotlib.pyplot as plt
from scipy.fftpack import fft



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



def df_fft(data, step):
    # step is the number of observations used to perform fft
    last, next = 0, step
    list_dataframes = []
    list_aux = []
    col = 0

    while col < len(data.columns):

        last, next = 0, step
        while last <= len(data):
            subset = data.iloc[last:next, col]
            if len(subset) > 0:
                xf, yff, yf = hf_fft(subset)
                yff = yff.tolist()
                yff.insert(0, subset.name)
                list_aux.insert(len(list_aux), yff)
            last, next = next, next + step
        list_dataframes.insert(len(list_dataframes), pd.DataFrame(list_aux))
        list_aux = []
        col += 1

    list_aux = []
    for i in range(0, len(list_dataframes)):
        dfaux = (list_dataframes[i].loc[:, 1:step - 1])
        dfaux.dropna(inplace=True)
        dfaux = pd.DataFrame(normalize(dfaux))

        list_aux.insert(len(list_aux), dfaux)

    dftotal = pd.concat(list_aux, ignore_index=True)
    return dftotal, list_aux


def fft_pac_pivoting(list_dataframes, pca):
    print(len(list_dataframes))
    print(len(list_dataframes[1].columns))
    X = []

    for i in range(0, 52):

        dfaux = list_dataframes[i]
        dfaux.dropna(inplace=True)
        names = []

        for j in range(0, pca.n_components):
            names.insert(len(names), "VAR_" + str(i) + "_PC_" + str(j))

        dfresult = pd.DataFrame(data=pca.transform(dfaux), columns=names)
        X.insert(len(X), dfresult)

    return pd.concat(X, ignore_index=True, axis=1)


def df_fft_pca(normadf, faultdf, pcs):
    # frequency domain and the concatenated version

    normal_dfFFT, list_normal_dfFFT = df_fft(normadf, 200)
    fault_dfFFT, list_fault_dfFFT = df_fft(faultdf, 200)
    full_df = normal_dfFFT.append(fault_dfFFT, ignore_index=True)
    names = []
    for j in range(0, 100):
        names.insert(len(names), str(j))
    scaler = StandardScaler().fit(full_df)
    full_df = pd.DataFrame(data=scaler.transform(full_df), columns=names)
    pca = decomposition.PCA(n_components=100)
    pca.fit(full_df)
    normal_piv_df = fft_pac_pivoting(list_normal_dfFFT, pca)
    fault_piv_df = fft_pac_pivoting(list_fault_dfFFT, pca)
    # ---------------
    pca2 = decomposition.PCA(n_components=pcs)
    fftpca_full_df = normal_piv_df.append(fault_piv_df, ignore_index=True)
    pca2.fit(fftpca_full_df)
    names = []
    for j in range(0, pca2.n_components):
        names.insert(len(names), "PC" + str(j))
    fftpca_dfnormal = pd.DataFrame(data=pca2.transform(normal_piv_df), columns=names)
    fftpca_dffailure = pd.DataFrame(data=pca2.transform(fault_piv_df), columns=names)
    # ---------------
    fftpca_dfnormal['failure'] = 0
    fftpca_dffailure['failure'] = 1
    full_df = fftpca_dfnormal.append(fftpca_dffailure, ignore_index=True)
    full_df = full_df.sample(frac=1).reset_index(drop=True)
    # Specify the data
    X = full_df.iloc[:, 0:pcs].astype(float)
    # Specify the target labels and flatten the array
    # y = np_utils.to_categorical(full_df['failure'])
    y = full_df['failure']

    return X, y

if __name__ == "__main__":
    print('main')