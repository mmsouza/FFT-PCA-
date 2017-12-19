import data_prep as dp
import time
from sklearn.model_selection import train_test_split

from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
import ann
from sklearn.metrics import confusion_matrix
import pre_fft_pca




def pca_fft_ann(n_modes=1, fault_prop=.5, pcs=5200, repetitions=1, filename='FFT-PCA-ANN', batchsize=32):
    normadf, faultdf = dp.load_df(n_modes, fault_prop)

    file = open(filename + '.csv', 'a')
    file.write('#test;n_modes;pre_proc_time;trainig_time;fault_prop;pcs;precision;recall;f1;tp;fp;tn;fn \n')

    pre_process_init = time.time()

    X , y = pre_fft_pca.df_fft_pca(normadf,faultdf,pcs)

    pre_process_finish = time.time()
    pre_proc_time = pre_process_finish - pre_process_init

    # ---------------------------------------------------------------------------------------------------------------------------------

    ann.inputsize = pcs
    estimator = KerasClassifier(build_fn=ann.bin_baseline_model, epochs=20, batch_size=batchsize, verbose=1)
    dp.validation(X, y ,estimator,repetitions,n_modes,pre_proc_time,fault_prop,pcs,file)

    # ---------------------------------------------------------------------------------------------------------------------------------
    file.close()


# ---------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pca_fft_ann(repetitions=1,batchsize=600)
    # plt.show()
    # CFM.plot_confusion_matrix(cnf_matrix, classes=['Normal','Falha'],title= " Matriz de Confusão FFT_PCA_ANN")
    # plt.savefig('FFT_PCA', bbox_inches='tight')
    # plt.figure()
    # CFM.plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Falha'], title=" Matriz de Confusão FFT_PCA_AN ", normalize= True)
    # plt.savefig('FFT_PCA_Norm', bbox_inches='tight')
    # print(cnf_matrix)