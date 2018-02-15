import data_handler as dp
import time
from sklearn.svm import LinearSVC
import preprocessor_pca
import preprocessor_fft_pca

def run(n_modes=1, fault_prop=.5, pcs=5200, repetitions=1, filename='pca_svm_'):
    normal_data, fault1_df = dp.load_df(n_modes, fault_prop)
    pre_process_init =time.perf_counter()

    # -------------loading data-----------

    X, y =preprocessor_fft_pca.df_fft_pca(normal_data, fault1_df, pcs)
    pre_process_finish =time.perf_counter()
    pre_proc_time = pre_process_finish - pre_process_init

    print(filename +' pre-process finished')

    #setup classifier
    estimator = LinearSVC(dual=False,verbose=True)
    dp.validation(X, y, estimator, repetitions, n_modes, pre_proc_time, fault_prop,filename)




if __name__ == "__main__":
    print('main')