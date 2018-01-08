import data_prep as dp
import time
from sklearn.svm import LinearSVC
import pca


def run_pca_svm(n_modes=1, fault_prop=.5,pcs=52, repetitions=1, filename='fft_pca_svm_'):
    normal_data, fault1_df = dp.load_df(n_modes, fault_prop)
    pre_process_init = time.time()

    # -------------loading data-----------

    X, y = pca.df_pca(normal_data, fault1_df, pcs, dp.colNames)


    pre_process_finish = time.time()
    pre_proc_time = pre_process_finish - pre_process_init

    print(filename +' pre-process finished')

    #setup classifier
    estimator = LinearSVC(dual=False,verbose=True)
    dp.validation(X, y, estimator, repetitions, n_modes, pre_proc_time, fault_prop,filename)




if __name__ == "__main__":
    print('main')