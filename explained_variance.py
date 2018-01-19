from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import preprocessor_fft_pca
import pandas as pd
import matplotlib as plt
import numpy as np


def explained_variance_fft_pca(normadf, faultdf, pcs):

    normal_dfFFT, list_normal_dfFFT = preprocessor_fft_pca.df_fft(normadf, 200)
    fault_dfFFT, list_fault_dfFFT = preprocessor_fft_pca.df_fft(faultdf, 200)
    full_df = normal_dfFFT.append(fault_dfFFT, ignore_index=True)
    names = []
    for j in range(0, 100):
        names.insert(len(names), str(j))
    scaler = StandardScaler().fit(full_df)
    full_df = pd.DataFrame(data=scaler.transform(full_df), columns=names)
    pca = decomposition.PCA(n_components=100)
    pca.fit(full_df)
    normal_piv_df = preprocessor_fft_pca.fft_pac_pivoting(list_normal_dfFFT, pca)
    fault_piv_df = preprocessor_fft_pca.fft_pac_pivoting(list_fault_dfFFT, pca)
    # ---------------
    pca2 = decomposition.PCA(n_components=pcs)
    fftpca_full_df = normal_piv_df.append(fault_piv_df, ignore_index=True)
    pca2.fit(fftpca_full_df)

    return np.cumsum(pca2.explained_variance_ratio_)


def explained_variance_pca(normal_data, fault1_df, pcs, colNames):
    full_df = normal_data.append(fault1_df, ignore_index=True)
    scaler = StandardScaler().fit(full_df)  # setup normalizer
    full_df = pd.DataFrame(data=scaler.transform(full_df), columns=colNames)  # formatting result as pandas dataframe
    full_df = full_df.sample(frac=1).reset_index(drop=True)  # data shuffle
    pca = decomposition.PCA(n_components=pcs)  # setting number of principle components
    pca.fit(full_df)

    return np.cumsum(pca.explained_variance_ratio_)


def print_explained_variance(exp_var_array):
    plt.plot(exp_var_array)
    plt.xlabel("N comp")
    plt.ylabel("Cumulative")
    plt.figure()
    return


if __name__ == "__main__":
    print('main')
