from keras.models import Sequential
from keras.layers import Dense

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
