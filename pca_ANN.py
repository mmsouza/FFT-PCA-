from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import data_prep as dp
import time
import ann

def baseline_model():
    model = Sequential()
    model.add(Dense(70, input_dim=12, activation='tanh'))
    model.add(Dense(40, activation='tanh'))
    model.add(Dense(15, activation='tanh'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# -------------loading data-----------
normal_data, Fault1_df = dp.import_data('C:/Users/Lais-WHart/Google Drive/UFRGS/Mestrado/Data mining/Data/')

full_df = normal_data.append(Fault1_df, ignore_index=True)
scaler = StandardScaler().fit(full_df)  # setup normalizer
full_df = pd.DataFrame(data=scaler.transform(full_df), columns=dp.colNames)  # formatting result as pandas dataframe
full_df = full_df.sample(frac=1).reset_index(drop=True)  # data shuffle

tic = time.clock()  # time counter

pca = decomposition.PCA(n_components=12)  # setting number of principle components
pca.fit(full_df)

# Applying pca and formatting as pandas dataframe
names = []
for j in range(0, pca.n_components):
    names.insert(len(names), "PC" + str(j))

dfnormal = pd.DataFrame(data=pca.transform(normal_data), columns=names)
dffailure = pd.DataFrame(data=pca.transform(Fault1_df), columns=names)


# Adding dummy data, labels that mark if a given occurrence is normal or a failure
dffailure['normal'] = 0
dffailure['failure'] = 1
dfnormal['normal'] = 1
dfnormal['failure'] = 0

# join both data classes
full_df = dfnormal.append(dffailure, ignore_index=True)
full_df = full_df.sample(frac=1).reset_index(drop=True)

# Specify the data
X = full_df.iloc[:, 0:12].astype(float)
# Specify the target labels and flatten the array
y = np_utils.to_categorical(full_df.iloc[:, 13:14])
#y=full_df['failure']
# setup classifier
ann.inputsize=12
estimator = KerasClassifier(build_fn=ann.baseline_model, epochs=20, batch_size=200, verbose=0)
estimator.fit(np.array(X), np.array(y))

# setup cross folder validation
seed = 7
np.random.seed(seed)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
#results = cross_val_score(estimator, np.array(X), np.array(y), cv=kfold)
results = cross_val_score(estimator, np.array(X), np.array(y), cv=kfold, scoring= 'precision' )
print(results)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

# print elapsed time
toc = time.clock()
print(toc - tic)
