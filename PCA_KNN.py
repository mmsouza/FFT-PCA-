from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import data_prep as dp
import time
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import Confusion_matrix as CFM


def pca_KNN():
#if __name__ == "__main__":

    # -------------loading data-----------
    fault_list = []
    for i in range(1, 20):
        fault_list.insert(len(fault_list),dp.import_data('C:/Users/Lais-WHart/Google Drive/UFRGS/Mestrado/Data mining/full2/', 'Fault_','base_mode', i, 24, 696))
        print(i)

    Fault1_df = pd.concat(fault_list, ignore_index=True)
    #print(faultdf)
    normal_data = dp.import_data('C:/Users/Lais-WHart/Google Drive/UFRGS/Mestrado/Data mining/full2/', 'Normal',
                             'base_mode', 0, 24, 14352)


    full_df = normal_data.append(Fault1_df, ignore_index=True)
    scaler = StandardScaler().fit(full_df)  # setup normalizer
    full_df = pd.DataFrame(data=scaler.transform(full_df), columns=dp.colNames)  # formatting result as pandas dataframe
    full_df = full_df.sample(frac=1).reset_index(drop=True)  # data shuffle

    tic = time.clock()  # time counter

    pca = decomposition.PCA(n_components=52)  # setting number of principle components
    pca.fit(full_df)

    # Applying pca and formatting as pandas dataframe
    names = []
    for j in range(0, pca.n_components):
        names.insert(len(names), "PC" + str(j))

    dfnormal = pd.DataFrame(data=pca.transform(normal_data), columns=names)
    dffailure = pd.DataFrame(data=pca.transform(Fault1_df), columns=names)


    # Adding dummy data, labels that mark if a given occurrence is normal or a failure
    #dffailure['normal'] = 0
    dffailure['failure'] = 1
    #dfnormal['normal'] = 1
    dfnormal['failure'] = 0

    # join both data classes
    full_df = dfnormal.append(dffailure, ignore_index=True)
    full_df = full_df.sample(frac=1).reset_index(drop=True)

    # Specify the data
    X = full_df.iloc[:, 0:52].astype(float)
    # Specify the target labels and flatten the array
    #y = np_utils.to_categorical(full_df.iloc[:, 13:14])
    y=full_df['failure']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)




    # setup classifier


    estimator = KNeighborsClassifier(n_neighbors=5)
    estimator.fit(np.array(X_train), np.array(y_train))
    ypred = estimator.predict(np.array(X_test))

    cnf_matrix = confusion_matrix(y_test, ypred)

    print("\n Acurácia  " + str(metrics.accuracy_score(y_test, ypred)))
    # Plot non-normalized confusion matrix

    CFM.plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Falha'], title=" Matriz de Confusão PCA-KNN ")
    plt.savefig('PCA-KNN',bbox_inches='tight')
    plt.figure()
    CFM.plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Falha'], title=" Matriz de Confusão  PCA-KNN", normalize=True)
    plt.savefig('PCA-KNN_Norm',bbox_inches='tight')
    #plt.show()

    print(cnf_matrix)

    toc = time.clock()
    print(toc - tic)
