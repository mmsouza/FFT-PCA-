from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

inputsize = 40


def shallow_ann():
    # create model
    model = Sequential()
    model.add(Dense(40, input_dim=inputsize, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    # Compile model

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def baseline_model():
    model = Sequential()
    model.add(Dense(70, input_dim=inputsize, activation='tanh'))
    model.add(Dense(40, activation='tanh'))
    model.add(Dense(15, activation='tanh'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def bin_shallow_ann():
    # create model
    model = Sequential()
    model.add(Dense(40, input_dim=inputsize, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def bin_baseline_model():
    model = Sequential()
    model.add(Dense(70, kernel_initializer='normal', input_dim=inputsize, activation='relu'))
    model.add(Dense(40, kernel_initializer='normal', activation='relu'))
    model.add(Dense(15, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model
