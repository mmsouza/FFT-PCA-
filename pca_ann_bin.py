from sklearn.metrics import confusion_matrix
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
import Confusion_matrix as CFM
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import data_prep as dp
import time
import ann
import matplotlib.pyplot as plt
import sklearn.metrics as metrics



def pca_ann():
#if __name__ == "__main__":

    # -------------loading data-----------

    fault_list=[]
    ncopm =50
    for i in range(1, 20):
        fault_list.insert(len(fault_list),dp.import_data('C:/Users/Lais-WHart/Google Drive/UFRGS/Mestrado/Data mining/full2/', 'Fault_','base_mode', i, 24, 696))

    Fault1_df = pd.concat(fault_list, ignore_index=True)
    print(str(len(Fault1_df))+ 'len fault')
    normal_data = dp.import_data('C:/Users/Lais-WHart/Google Drive/UFRGS/Mestrado/Data mining/full2/', 'Normal',
                             'base_mode', 0, 24, 14352)


    full_df = normal_data.append(Fault1_df, ignore_index=True)
    scaler = StandardScaler().fit(full_df)  # setup normalizer
    full_df = pd.DataFrame(data=scaler.transform(full_df), columns=dp.colNames)  # formatting result as pandas dataframe
    full_df = full_df.sample(frac=1).reset_index(drop=True)  # data shuffle

    tic = time.clock()  # time counter

    pca = decomposition.PCA(n_components=50)  # setting number of principle components
    pca.fit(full_df)

    # Applying pca and formatting as pandas dataframe
    names = []
    for j in range(0, pca.n_components):
        names.insert(len(names), "PC" + str(j))

    dfnormal = pd.DataFrame(data=pca.transform(normal_data), columns=names)
    dffailure = pd.DataFrame(data=pca.transform(Fault1_df), columns=names)


    # Adding dummy data, labels that mark if a given occurrence is normal or a failure

    dffailure['failure'] = 1
    dfnormal['failure'] = 0

    # join both data classes
    full_df = dfnormal.append(dffailure, ignore_index=True)
    #full_df = full_df.sample(frac=1).reset_index(drop=True)
    #full_df=dffailure
    # Specify the data
    X = full_df.iloc[:, 0:ncopm].astype(float)
    # Specify the target labels and flatten the array
    #y = np_utils.to_categorical(full_df.iloc[:, 13:14])
    y=full_df['failure']
    # setup classifier
    ann.inputsize=ncopm
    estimator = KerasClassifier(build_fn=ann.bin_baseline_model, epochs=20, batch_size=32, verbose=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)
    print(pd.DataFrame(data=y_test).describe())


    estimator.fit(np.array(X_train), np.array(y_train))
    ypred = estimator.predict(np.array(X_test))
    print(pd.DataFrame(data=ypred).describe())
    cnf_matrix = confusion_matrix(y_test,ypred)
    print("\n Acurácia  " + str(metrics.accuracy_score(y_test, ypred)))
    # Plot non-normalized confusion matrix

    CFM.plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Falha'], title=" Matriz de Confusão PCA-ANN")
    plt.savefig('PCA-ANN', bbox_inches='tight')
    plt.figure()
    CFM.plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Falha'], title=" Matriz de Confusão PCA-ANN", normalize=True)
    plt.savefig('PCA-ANN_Norm', bbox_inches='tight')
    plt.show()

    #plt.show()

    print(cnf_matrix)
    toc = time.clock()
    print(toc - tic)




