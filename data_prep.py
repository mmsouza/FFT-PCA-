import pandas as pd
from sklearn.metrics import confusion_matrix
import time
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

colNames = ["Xmv_1", "Xmv_2", "Xmv_3", "Xmv_4", "Xmv_5", "Xmv_6", "Xmv_7", "Xmv_8", "Xmv_9", "Xmv_10", "Xmv_11",
            "Xmv_12", "XMEAS_1", "XMEAS_2", "XMEAS_3", "XMEAS_4", "XMEAS_5", "XMEAS_6", "XMEAS_7",
            "XMEAS_8", "XMEAS_9", "XMEAS_10",
            "XMEAS_11", "XMEAS_12", "XMEAS_13", "XMEAS_14", "XMEAS_15", "XMEAS_16", "XMEAS_17",
            "XMEAS_18", "XMEAS_19",
            "XMEAS_20", "XMEAS_21", "XMEAS_22", "XMEAS_23", "XMEAS_24", "XMEAS_25", "XMEAS_26",
            "XMEAS_27", "XMEAS_28",
            "XMEAS_29", "XMEAS_30", "XMEAS_31", "XMEAS_32", "XMEAS_33", "XMEAS_34", "XMEAS_35",
            "XMEAS_36", "XMEAS_37",
            "XMEAS_38", "XMEAS_39", "XMEAS_40", "XMEAS_41"]


def import_file(data_path, condition, mode, fault_id, step, final):
    list_aux = []

    for x in range(step, final + step, step):
        df_aux = pd.read_csv(data_path + condition + mode + '_ID_' + str(fault_id) + '_' + str(x) + '.csv',
                             names=colNames)

        list_aux.insert(len(list_aux), df_aux)

    df = pd.concat(list_aux, ignore_index=True)

    return df


def load_df(n_modes, fault_proportion):
    fault_list = []
    normal_list = []

    for i in range(1, 20):
        fault_list.insert(len(fault_list),
                          import_file('C:/Users/Lais-WHart/Google Drive/UFRGS/Mestrado/Data mining/full2/', 'Fault_',
                                      'base_mode', i, 24, 696))
        # print(i)

    if n_modes > 1 & n_modes < 8:
        for j in range(1, n_modes):
            for i in range(1, 20):
                fault_list.insert(len(fault_list),
                                  import_file('C:/Users/Lais-WHart/Google Drive/UFRGS/Mestrado/Data mining/full2/',
                                              'Fault_',
                                              'mode_' + str(j), i, 24, 696))
                print(j)

                # print(faultdf)

    normal_list.insert(len(normal_list),
                       import_file('C:/Users/Lais-WHart/Google Drive/UFRGS/Mestrado/Data mining/full2/', 'Normal',
                                   'base_mode', 0, 24, 6528))

    if n_modes > 1 & n_modes < 8:
        for j in range(1, n_modes):
            normal_list.insert(len(normal_list),
                               import_file('C:/Users/Lais-WHart/Google Drive/UFRGS/Mestrado/Data mining/full2/',
                                           'Normal',
                                           'mode_' + str(j), 0, 24, 6528))  # 6528 value for fault and normal balance
            print(j)

    normal_data = pd.concat(normal_list, ignore_index=True)
    fault1_df = pd.concat(fault_list, ignore_index=True)
    fault1_df = fault1_df[0:len(normal_data.index)]

    print(round((((1 - fault_proportion) * len(fault1_df)) / fault_proportion)))

    if fault_proportion > .5:
        new_normal_lengh = int(((1 - fault_proportion) * len(fault1_df)) / fault_proportion)
        print(new_normal_lengh)
        normal_data = normal_data[0: new_normal_lengh]
    if fault_proportion < .5:
        new_fault_lengh = int((fault_proportion * len(normal_data)) / (1 - fault_proportion))
        fault1_df = fault1_df[0:new_fault_lengh]
        print(new_fault_lengh)

    return normal_data, fault1_df


def validation(X, y, estimator, repetitions, n_modes, pre_proc_time, fault_prop, filename,pcs=-1, batchsize=-1,
               n_neghbors=-1):
    file = open(filename + '.csv', 'a')
    file.write(
        '#test;n_modes;pre_proc_time;trainig_time;fault_prop;pcs;precision;recall;f1;tp;fp;tn;fn;batchsize;n_neghbors \n')

    for j in range(1, repetitions + 1):
        process_init = time.time()

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)
        estimator.fit(np.array(x_train), np.array(y_train))
        ypred = estimator.predict(np.array(x_test))

        process_finish = time.time()
        process_time = process_finish - process_init

        cnf_matrix = confusion_matrix(y_test, ypred)

        tp = np.array(cnf_matrix).item((1, 1))
        fp = np.array(cnf_matrix).item((0, 1))
        tn = np.array(cnf_matrix).item((0, 0))
        fn = np.array(cnf_matrix).item((1, 0))

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * ((precision * recall) / (precision + recall))

        # test;n_modes;pre_proc_time;trainig_time;fault_prop;pcs;precision;recall;f1;tp;fp;tn;fn \n
        file.write(str(j) + ';' + str(n_modes) + ';' + str(round(pre_proc_time / 60, 2)) + ';' + str(
            round(process_time / 60, 2)) + ';' + str(fault_prop) + ';' + str(pcs) + ';' + str(precision) + ';' + str(
            recall) + ';' + str(f1) + ';' + str(tp) + ';' + str(fp) + ';' + str(tn) + ';' + str(fn) + ';' + str(
            batchsize) + ';' + str(n_neghbors) + '\n')

    file.close()


if __name__ == "__main__":
    print()
