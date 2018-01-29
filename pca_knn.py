import data_handler as dp
import time
import preprocessor_pca
from sklearn.neighbors import KNeighborsClassifier


def run(n_modes=1, fault_prop=.5, pcs=52, repetitions=1, filename='PCA-KNN', batchsize=32, neighbors=5):


    normal_data, fault1_df = dp.load_df(n_modes, fault_prop)
    pre_process_init = time.time()

    # -------------loading data-----------

    X, y = preprocessor_pca.df_pca(normal_data, fault1_df, pcs, dp.colNames)

    pre_process_finish = time.time()
    pre_proc_time = pre_process_finish - pre_process_init

    # setup classifier
    estimator = KNeighborsClassifier(n_neighbors=neighbors)
    dp.validation(X, y, estimator, repetitions, n_modes, pre_proc_time, fault_prop, filename,pcs=pcs,n_neghbors=neighbors)



if __name__ == "__main__":
    run()
