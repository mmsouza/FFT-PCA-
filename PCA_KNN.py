import data_prep as dp
import time
from sklearn.neighbors import KNeighborsClassifier
import pca


def pca_KNN(n_modes=1, fault_prop=.5, pcs=52, repetitions=1, filename='PCA-KNN', batchsize=32, neighbors=5):
    file = open(filename + '.csv', 'a')
    file.write('#test;n_modes;pre_proc_time;trainig_time;fault_prop;pcs;precision;recall;f1;tp;fp;tn;fn \n')

    normal_data, fault1_df = dp.load_df(n_modes, fault_prop)
    pre_process_init = time.time()

    # -------------loading data-----------

    X, y = pca.df_pca(normal_data, fault1_df, pcs, dp.colNames)

    pre_process_finish = time.time()
    pre_proc_time = pre_process_finish - pre_process_init

    # setup classifier
    estimator = KNeighborsClassifier(n_neighbors=neighbors)
    dp.validation(X, y, estimator, repetitions, n_modes, pre_proc_time, fault_prop, pcs, file)
    file.close()


if __name__ == "__main__":
    pca_KNN()
