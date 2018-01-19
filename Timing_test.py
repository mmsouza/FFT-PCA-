import data_prep as dp
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np


def timing_test_KNN(neighbors=5):
    print("entrei")
    normadf, faultdf = dp.load_df(1, 0.5)

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

    # setup classifier
    estimator = KNeighborsClassifier(n_neighbors=neighbors)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)
    estimator.fit(np.array(x_train), np.array(y_train))

    predict_time_init = time.time()
    ypred = estimator.predict(np.array(x_test[1, :]))
    predict_time = time.time() - predict_time_init

    print("Prediction KNN time : {1}".format(str(predict_time)))


if __name__ == "__main__":
    timing_test_KNN(5)
