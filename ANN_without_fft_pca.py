import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import Data_prep as dp

def shallow_ann():
    # create model
    model = Sequential()
    model.add(Dense(40, input_dim=40, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



normal_data, Fault1_df = dp.import_data('C:/Users/matheus/Google Drive/UFRGS/Mestrado/Data mining/Data/')

normal_data['normal'] = 1
normal_data['failure'] = 0
Fault1_df['normal'] = 0
Fault1_df['failure'] = 1



full_df = normal_data.append(Fault1_df, ignore_index=True)

full_df = full_df.sample(frac=1).reset_index(drop=True)

# Specify the data
X = full_df.iloc[:, 0:40].astype(float)

# Specify the target labels and flatten the array
y = np_utils.to_categorical(full_df.iloc[:, 41:42])

estimator = KerasClassifier(build_fn=shallow_ann, epochs=20, batch_size=100, verbose=1)
estimator.fit(np.array(X), np.array(y))

seed = 7
np.random.seed(seed)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, np.array(X), np.array(y), cv=kfold)
print("Rusults: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
