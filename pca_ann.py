import data_handler as dp
import time
import ann_settings
import preprocessor_pca
from keras.wrappers.scikit_learn import KerasClassifier


def run(n_modes=1, fault_prop=.5, pcs=52, repetitions=1, filename='PCA-ANN', batchsize=512):


    normal_data, fault1_df = dp.load_df(n_modes, fault_prop)
    pre_process_init = time.time()
    # -------------loading data-----------


    # Applying PCA
    X, y = preprocessor_pca.df_pca(normal_data, fault1_df, pcs, dp.colNames)

    pre_process_finish = time.time()
    pre_proc_time = pre_process_finish - pre_process_init

    # setup classifier
    ann_settings.inputsize = pcs
    estimator = KerasClassifier(build_fn=ann_settings.bin_baseline_model, epochs=20, batch_size=batchsize, verbose=0)
    dp.validation(X, y, estimator, repetitions, n_modes, pre_proc_time, fault_prop, filename, pcs=pcs, batchsize=batchsize)



if __name__ == "__main__":
    run()
    #    CFM.plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Falha'], title=" Matriz de Confusão PCA-ANN")
    # plt.savefig('PCA-ANN', bbox_inches='tight')
    # plt.figure()
    # CFM.plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Falha'], title=" Matriz de Confusão PCA-ANN",normalize=True)
    # plt.savefig('PCA-ANN_Norm', bbox_inches='tight')
    # plt.show()
    # plt.show()
