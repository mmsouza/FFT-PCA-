from keras.wrappers.scikit_learn import KerasClassifier
import data_prep as dp
import time
import ann
import pca


def pca_ann(n_modes=1, fault_prop=.5, pcs=52, repetitions=1, filename='PCA-ANN', batchsize=32):
    file = open(filename + '.csv', 'a')
    file.write('#test;n_modes;pre_proc_time;trainig_time;fault_prop;pcs;precision;recall;f1;tp;fp;tn;fn \n')

    normal_data, fault1_df = dp.load_df(n_modes, fault_prop)
    pre_process_init = time.time()
    # -------------loading data-----------


    # Applying PCA
    X, y = pca.df_pca(normal_data, fault1_df, pcs, dp.colNames)

    pre_process_finish = time.time()
    pre_proc_time = pre_process_finish - pre_process_init

    # setup classifier
    ann.inputsize = pcs
    estimator = KerasClassifier(build_fn=ann.bin_baseline_model, epochs=20, batch_size=batchsize, verbose=1)
    dp.validation(X, y, estimator, repetitions, n_modes, pre_proc_time, fault_prop, pcs, file)

    file.close()


if __name__ == "__main__":
    pca_ann()
    #    CFM.plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Falha'], title=" Matriz de Confusão PCA-ANN")
    # plt.savefig('PCA-ANN', bbox_inches='tight')
    # plt.figure()
    # CFM.plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Falha'], title=" Matriz de Confusão PCA-ANN",normalize=True)
    # plt.savefig('PCA-ANN_Norm', bbox_inches='tight')
    # plt.show()
    # plt.show()
