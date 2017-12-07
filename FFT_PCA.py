import fft as fft
from sklearn import decomposition
from sklearn.preprocessing import normalize
import data_prep as dp
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np
import ann
import sklearn.metrics as metrics
from sklearn.metrics import make_scorer

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

    for i in range(0, 40):

        dfaux = list_dataframes[i]
        dfaux.dropna(inplace=True)
        names = []

        for j in range(0, pca.n_components):
            names.insert(len(names), "VAR_" + str(i) + "_PC_" + str(j))

        dfresult = pd.DataFrame(data=pca.transform(dfaux), columns=names)
        X.insert(len(X), dfresult)

    return pd.concat(X, ignore_index=True, axis=1)




if __name__ == "__main__":

    start = time.time()



    normadf, faultdf = dp.import_data('C:/Users/Lais-WHart/Google Drive/UFRGS/Mestrado/Data mining/Data/')

    # df_fft return the list of all dataframes of features in
    # frequency domain and the concatenated version
    tic = time.clock()
    normal_dfFFT, list_normal_dfFFT = df_fft(normadf, 200)
    fault_dfFFT, list_fault_dfFFT = df_fft(faultdf, 200)

    full_df = normal_dfFFT.append(fault_dfFFT, ignore_index=True)


    names = []
    for j in range(0, 100):
        names.insert(len(names), str(j))
    scaler = StandardScaler().fit(full_df)
    full_df = pd.DataFrame(data=scaler.transform(full_df), columns=names)
    pca = decomposition.PCA(n_components=50)
    pca.fit(full_df)
    normal_piv_df = fft_pac_pivoting(list_normal_dfFFT, pca)
    fault_piv_df = fft_pac_pivoting(list_fault_dfFFT, pca)
#---------------
    pca2 = decomposition.PCA(n_components=15)
    fftpca_full_df = normal_piv_df.append(fault_piv_df, ignore_index=True)
    pca2.fit(fftpca_full_df)
    names = []
    for j in range(0, pca2.n_components):
        names.insert(len(names), "PC" + str(j))
    fftpca_dfnormal = pd.DataFrame(data=pca2.transform(normal_piv_df), columns=names)
    fftpca_dffailure = pd.DataFrame(data=pca2.transform(fault_piv_df), columns=names)

# ---------------
    fftpca_dfnormal['normal'] = 1
    fftpca_dfnormal['failure'] = 0
    fftpca_dffailure['normal'] = 0
    fftpca_dffailure['failure'] = 1

    full_df = fftpca_dfnormal.append(fftpca_dffailure, ignore_index=True)
    full_df = full_df.sample(frac=1).reset_index(drop=True)

    # Specify the data
    X = full_df.iloc[:, 0:15].astype(float)
    # Specify the target labels and flatten the array
    y = np_utils.to_categorical(full_df.iloc[:, 16:17])
    #y=full_df['failure']
    ann.inputsize=15
    estimator = KerasClassifier(build_fn=ann.baseline_model, epochs=20, batch_size=3, verbose=1)

    estimator.fit(np.array(X), np.array(y))

    seed = 7
    np.random.seed(seed)
    #np.random.seed()
    kfold = KFold(n_splits=5, shuffle=True, random_state=np.random)




    results = cross_val_score(estimator, np.array(X), np.array(y), cv=kfold, scoring= make_scorer(metrics.precision_recall_fscore_support) )
    print(results)

    #print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    toc = time.clock()
    print(toc - tic)

