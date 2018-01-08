import data_prep as dp
import time
from sklearn import svm


def run_svm(n_modes=1, fault_prop=.5, repetitions=1, filename='svm_'):

    normadf, faultdf = dp.load_df(n_modes, fault_prop)

    # Adding dummy data, labels that mark if a given occurrence is normal or a failure
    pre_process_init = time.time()
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

    pre_process_finish = time.time()
    pre_proc_time = pre_process_finish - pre_process_init


    #setup classifier
    estimator = svm.SVC()
    dp.validation(X, y, estimator, repetitions, n_modes, pre_proc_time,fault_prop,filename)




if __name__ == "__main__":
    print('main')