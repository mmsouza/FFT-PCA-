import data_handler as dp
import time
from keras.wrappers.scikit_learn import KerasClassifier
import ann_settings


def run(n_modes=1, fault_prop=.5, pcs=52, repetitions=1, filename='ANN', batchsize=512):
    normadf, faultdf = dp.load_df(n_modes, fault_prop)

    # Adding dummy data, labels that mark if a given occurrence is normal or a failure
    pre_process_init =time.perf_counter()
    faultdf['failure'] = 1
    normadf['failure'] = 0
    # join both data classes
    full_df = normadf.append(faultdf, ignore_index=True)
    full_df = full_df.sample(frac=1).reset_index(drop=True)

    # Specify the data
    X = full_df.iloc[:, 0:52].astype(float)
    # Specify the target labels and flatten the array
    # y = np_utils.to_categorical(full_df.iloc[:, 13:14])
    y = full_df['failure']

    # capture pre-process time
    pre_process_finish =time.perf_counter()
    pre_proc_time = pre_process_finish - pre_process_init

    ann_settings.inputsize = pcs  # set input_size as the number of principle components
    estimator = KerasClassifier(build_fn=ann_settings.bin_baseline_model, epochs=20, batch_size=batchsize, verbose=0)
    dp.validation(X, y, estimator, repetitions, n_modes, pre_proc_time, fault_prop, filename, batchsize=batchsize)

if __name__ == "__main__":
    print('main')