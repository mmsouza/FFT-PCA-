import data_prep as dp
import time
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
import ann
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import Confusion_matrix as CFM





def ANN():
#if __name__ == "__main__":
    ncomp = 52
    fault_list=[]

    for i in range(1, 20):
        fault_list.insert(len(fault_list),dp.import_data('C:/Users/Lais-WHart/Google Drive/UFRGS/Mestrado/Data mining/full2/', 'Fault_','base_mode', i, 24, 696))
        print(i)

        Fault1_df = pd.concat(fault_list, ignore_index=True)

    normal_data = dp.import_data('C:/Users/Lais-WHart/Google Drive/UFRGS/Mestrado/Data mining/full2/', 'Normal',
                                 'base_mode', 0, 24, 14352)

    tic = time.clock()  # time counter

    normal_data['failure'] = 0.0
    Fault1_df['failure'] = 1.0



    full_df = normal_data.append(Fault1_df, ignore_index=True)
    #full_df =Fault1_df
    #full_df = full_df.sample(frac=1).reset_index(drop=True)

    # Specify the data
    X = full_df.iloc[:, 0:ncomp].astype(float)

    # Specify the target labels and flatten the array
    y = full_df['failure']


    ann.inputsize=ncomp
    estimator = KerasClassifier(build_fn=ann.bin_baseline_model, epochs=20, batch_size=32, verbose=1)
    X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.20, shuffle=True)

    #print(y_test.describe())


    estimator.fit(np.array(X_train), np.array(y_train))

    ypred=estimator.predict(np.array(X_test))

    print(pd.DataFrame(data=ypred).describe())
    cnf_matrix = confusion_matrix(y_test,ypred)

    print("\n Acurácia  " + str(metrics.accuracy_score(y_test, ypred)))
    # Plot non-normalized confusion matrix

    CFM.plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Falha'], title=" Matriz de Confusão ANN ")
    plt.savefig('ANN', bbox_inches='tight')
    plt.figure()
    CFM.plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Falha'], title=" Matriz de Confusão ANN", normalize=True)
    plt.savefig('ANN_Norm', bbox_inches='tight')
    #plt.show()

    #plt.show()

    print(cnf_matrix)
    # print elapsed time
    toc = time.clock()
    print(toc - tic)
