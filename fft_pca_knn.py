import data_handler as dp
import time
import preprocessor_fft_pca
from sklearn.neighbors import KNeighborsClassifier





def run_pca_fft_knn(n_modes=1, fault_prop=.5, pcs=5200, repetitions=1, filename='FFT-PCA-KNN', neighbors=5):

    normadf, faultdf = dp.load_df(n_modes, fault_prop)


    pre_process_init = time.time()

    X, y = preprocessor_fft_pca.df_fft_pca(normadf, faultdf, pcs)

    pre_process_finish = time.time()
    pre_proc_time = pre_process_finish - pre_process_init

    # ---------------------------------------------------------------------------------------------------------------------------------
    estimator = KNeighborsClassifier(n_neighbors=neighbors)
    dp.validation(X, y, estimator, repetitions, n_modes, pre_proc_time, fault_prop,filename,pcs=pcs,n_neghbors=neighbors)


# ---------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    print('main')

#run_pca_fft_knn()
#plt.show()


# plt.plot(np.cumsum(pca2.explained_variance_ratio_))
# plt.xlabel("N comp")
# plt.ylabel("Cumulative")
# plt.figure()
