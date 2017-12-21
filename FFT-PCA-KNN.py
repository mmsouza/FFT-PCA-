import fft as fft
import data_prep as dp
import time
import pandas as pd
import matplotlib.pyplot as plt
import pre_fft_pca
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize



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
                xf, yff, yf = fft.hf_fft(subset)
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


def pca_fft_knn(n_modes=1, fault_prop=.5, pcs=5200, repetitions=1, filename='FFT-PCA-KNN',  neighbors=5):
    normadf, faultdf = dp.load_df(n_modes, fault_prop)


    pre_process_init = time.time()

    X, y = pre_fft_pca.df_fft_pca(normadf, faultdf, pcs)

    pre_process_finish = time.time()
    pre_proc_time = pre_process_finish - pre_process_init

    # ---------------------------------------------------------------------------------------------------------------------------------
    estimator = KNeighborsClassifier(n_neighbors=neighbors)
    dp.validation(X, y, estimator, repetitions, n_modes, pre_proc_time, fault_prop,filename,pcs=pcs,n_neghbors=neighbors)


# ---------------------------------------------------------------------------------------------------------------------------------

pca_fft_knn()
plt.show()


# plt.plot(np.cumsum(pca2.explained_variance_ratio_))
# plt.xlabel("N comp")
# plt.ylabel("Cumulative")
# plt.figure()
