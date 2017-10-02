import fft
from sklearn import decomposition
import pandas as pd
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import Data_prep as dp
from matplotlib.axes import Axes
import time


import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import numpy as np




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

    seed = 7
    np.random.seed(seed)


    normadf, faultdf = dp.import_data()

    normadf = normadf.iloc[:, 0:40]
    faultdf = faultdf.iloc[:, 0:40]

    # df_fft return the list of all dataframes of features in
    # frequency domain and the concatenated version
    normal_dfFFT, list_normal_dfFFT = df_fft(normadf, 100)
    fault_dfFFT, list_fault_dfFFT = df_fft(faultdf, 100)

    full_df = normal_dfFFT.append(fault_dfFFT, ignore_index=True)
    pca = decomposition.PCA(n_components=1)
    pca.fit(full_df)

    normal_piv_df= fft_pac_pivoting(list_normal_dfFFT, pca)
    normal_piv_df['normal'] = 1
    normal_piv_df['failure'] = 0

    fault_piv_df = fft_pac_pivoting(list_fault_dfFFT, pca)
    fault_piv_df['normal'] = 0
    fault_piv_df['failure'] = 1

    end = time.time()
    print("Elapsed Time " + str(end - start))

    full_df = normal_piv_df.append(fault_piv_df, ignore_index=True)
    full_df = full_df.sample(frac=1).reset_index(drop=True)

    print(full_df.describe())
    # Specify the data
    X = full_df.iloc[:, 0:40].astype(float)

    # Specify the target labels and flatten the array
    y = np_utils.to_categorical(full_df.iloc[:, 41:42])


    def baseline_model():
        # create model
        model = Sequential()
        model.add(Dense(40, input_dim=40, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


    estimator = KerasClassifier(build_fn=baseline_model, epochs=20, batch_size=5, verbose=1)
    estimator.fit(np.array(X), np.array(y))

    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

    results = cross_val_score(estimator, np.array(X), np.array(y), cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))



    # print(data.describe())

    # print(data2.head())

'''
pca = decomposition.PCA(n_components=2)
pca.fit(dftotal)

for i in range(0, 41):
    dfaux = list_dataframes[i].loc[:, 1:199]
    dfaux.dropna(inplace=True)
    dfaux = normalize(dfaux)

    # print(type(dfaux))
    X = pca.transform(dfaux)
    # print(pca.components_)

    plt.figure(i)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.scatter(X[:, 0], X[:, 1])

plt.show()
'''