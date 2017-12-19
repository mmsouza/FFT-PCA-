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


def ANN(n_modes=1, fault_prop=.5, pcs=52, repetitions=1, filename='ANN', batchsize=32):

    normadf, faultdf = dp.load_df(n_modes, fault_prop)

    file = open(filename + '.csv', 'a')
    file.write('#test;n_modes;pre_proc_time;trainig_time;fault_prop;pcs;precision;recall;f1;tp;fp;tn;fn \n')

    # Adding dummy data, labels that mark if a given occurrence is normal or a failure
    pre_process_init = time.time()
    faultdf['failure'] = 1
    normadf['failure'] = 0
    # join both data classes
    full_df = normadf.append(faultdf, ignore_index=True)
    full_df = full_df.sample(frac=1).reset_index(drop=True)

    # Specify the data
    X = full_df.iloc[:, 0:52].astype(float)
    # Specify the target labels and flatten the array
    # y = np_utils.to_categorical(full_df.iloc[:, 13:14])
    y = full_df['failure']

    pre_process_finish = time.time()
    pre_proc_time = pre_process_finish - pre_process_init



    ann.inputsize = pcs
    estimator = KerasClassifier(build_fn=ann.bin_baseline_model, epochs=20, batch_size=batchsize, verbose=1)
    dp.validation(X, y, estimator, repetitions, n_modes, pre_proc_time, fault_prop, pcs, file)
    file.close()
