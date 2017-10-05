import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

import time



seed = 7
np.random.seed(seed)

normal_data = pd.read_csv('C:/Users/Lais-WHart/Google Drive/UFRGS/Mestrado/Data mining/Normal_lessdata/normal_output.csv',
                          names=["XMEAS_1", "XMEAS_2", "XMEAS_3", "XMEAS_4", "XMEAS_5", "XMEAS_6", "XMEAS_7", "XMEAS_8",
                                 "XMEAS_9", "XMEAS_10",
                                 "XMEAS_11", "XMEAS_12", "XMEAS_13", "XMEAS_14", "XMEAS_15", "XMEAS_16", "XMEAS_17",
                                 "XMEAS_18", "XMEAS_19",
                                 "XMEAS_20", "XMEAS_21", "XMEAS_22", "XMEAS_23", "XMEAS_24", "XMEAS_25", "XMEAS_26",
                                 "XMEAS_27", "XMEAS_28",
                                 "XMEAS_29", "XMEAS_30", "XMEAS_31", "XMEAS_32", "XMEAS_33", "XMEAS_34", "XMEAS_35",
                                 "XMEAS_36", "XMEAS_37",
                                 "XMEAS_38", "XMEAS_39", "XMEAS_40", "XMEAS_41"])



list_aux = []

for x in range(24, 696 + 24, 24):
    df_aux = pd.read_csv('C:/Users/Lais-WHart/Google Drive/UFRGS/Mestrado/Data mining/Failure/Fault2_' + str(x) + '.csv',
                         names=["XMEAS_1", "XMEAS_2", "XMEAS_3", "XMEAS_4", "XMEAS_5", "XMEAS_6", "XMEAS_7", "XMEAS_8",
                                "XMEAS_9", "XMEAS_10",
                                "XMEAS_11", "XMEAS_12", "XMEAS_13", "XMEAS_14", "XMEAS_15", "XMEAS_16", "XMEAS_17",
                                "XMEAS_18", "XMEAS_19",
                                "XMEAS_20", "XMEAS_21", "XMEAS_22", "XMEAS_23", "XMEAS_24", "XMEAS_25", "XMEAS_26",
                                "XMEAS_27", "XMEAS_28",
                                "XMEAS_29", "XMEAS_30", "XMEAS_31", "XMEAS_32", "XMEAS_33", "XMEAS_34", "XMEAS_35",
                                "XMEAS_36", "XMEAS_37",
                                "XMEAS_38", "XMEAS_39", "XMEAS_40", "XMEAS_41"])

    list_aux.insert(len(list_aux), df_aux)
    print('C:/Users/matheus/Desktop/Fault_TEP/Fault1' + str(x))

Fault1_df = pd.concat(list_aux, ignore_index=True)



full_df = normal_data.append(Fault1_df, ignore_index=True)
scaler  = StandardScaler().fit(full_df)


full_df = pd.DataFrame(data=scaler.transform(full_df), columns=["XMEAS_1", "XMEAS_2", "XMEAS_3", "XMEAS_4", "XMEAS_5", "XMEAS_6", "XMEAS_7", "XMEAS_8",
                                "XMEAS_9", "XMEAS_10",
                                "XMEAS_11", "XMEAS_12", "XMEAS_13", "XMEAS_14", "XMEAS_15", "XMEAS_16", "XMEAS_17",
                                "XMEAS_18", "XMEAS_19",
                                "XMEAS_20", "XMEAS_21", "XMEAS_22", "XMEAS_23", "XMEAS_24", "XMEAS_25", "XMEAS_26",
                                "XMEAS_27", "XMEAS_28",
                                "XMEAS_29", "XMEAS_30", "XMEAS_31", "XMEAS_32", "XMEAS_33", "XMEAS_34", "XMEAS_35",
                                "XMEAS_36", "XMEAS_37",
                                "XMEAS_38", "XMEAS_39", "XMEAS_40", "XMEAS_41"])




full_df = full_df.sample(frac=1).reset_index(drop=True)

tic = time.clock()


pca = decomposition.PCA(n_components=12)
pca.fit(full_df)

names = []

for j in range(0, pca.n_components):
    names.insert(len(names), "PC" + str(j))


dfnormal= pd.DataFrame(data=pca.transform(normal_data), columns=names)
dffailure= pd.DataFrame(data=pca.transform(Fault1_df), columns=names)

dffailure['normal'] = 0
dffailure['failure'] = 1
dfnormal['normal'] = 1
dfnormal['failure'] = 0

full_df = dfnormal.append(dffailure, ignore_index=True)

full_df = full_df.sample(frac=1).reset_index(drop=True)


# Specify the data
X = full_df.iloc[:, 0:12].astype(float)

# Specify the target labels and flatten the array
y = np_utils.to_categorical(full_df.iloc[:, 13:14])


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(70, input_dim=12, activation='tanh'))
    model.add(Dense(40, activation='tanh'))
    model.add(Dense(15, activation='tanh'))
    model.add(Dense(2, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=20, batch_size=32, verbose=0)
estimator.fit(np.array(X), np.array(y))


#y_pred = estimator.predict(X_test)


kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, np.array(X), np.array(y), cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

toc = time.clock()
print(toc - tic)

'''
print(precision_score(y_test, dfyf))
#Recall
print(recall_score(y_test, dfyf))

# F1 score
#f1_score(y_test, y_pred)

# Cohen's kappa
#cohen_kappa_score(y_test, y_pred)
'''
