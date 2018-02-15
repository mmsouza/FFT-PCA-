import data_handler as dp
import time
import ann_settings
import preprocessor_fft_pca
from keras.wrappers.scikit_learn import KerasClassifier





def run(n_modes=1, fault_prop=.5, pcs=5200, repetitions=1, filename='FFT-PCA-ANN', batchsize=512):
    normadf, faultdf = dp.load_df(n_modes, fault_prop)
    pre_process_init =time.perf_counter()

    X, y = preprocessor_fft_pca.df_fft_pca(normadf, faultdf, pcs)

    pre_process_finish =time.perf_counter()
    pre_proc_time = pre_process_finish - pre_process_init

    # ---------------------------------------------------------------------------------------------------------------------------------

    ann_settings.inputsize = pcs
    estimator = KerasClassifier(build_fn=ann_settings.bin_baseline_model, epochs=20, batch_size=batchsize, verbose=0)
    dp.validation(X,y,estimator,repetitions,n_modes,pre_proc_time,fault_prop,filename,pcs=pcs,batchsize=batchsize)

    # ---------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    run(repetitions=1, batchsize=600)
    # plt.show()
    # CFM.plot_confusion_matrix(cnf_matrix, classes=['Normal','Falha'],title= " Matriz de Confusão FFT_PCA_ANN")
    # plt.savefig('FFT_PCA', bbox_inches='tight')
    # plt.figure()
    # CFM.plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Falha'], title=" Matriz de Confusão FFT_PCA_AN ", normalize= True)
    # plt.savefig('FFT_PCA_Norm', bbox_inches='tight')
    # print(cnf_matrix)