from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from keras import optimizers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.HDFStore('store.h5')['df']

model = Sequential()
model.add(Dense(7, input_shape=(7,), name='input', activation='sigmoid'))
model.add(Dense(1, activation='sigmoid', kernel_initializer='normal'))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

## build callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=50),
    ModelCheckpoint('devTest.hdf5', monitor='val_loss',
                    verbose=0, save_best_only=True,
                    save_weights_only=False, mode='auto',
                    period=1
                    )
]

## get the data for the two-classes to discriminate
training_processes = data[
    (data['sample_names'] == 'ZTT.root') | (data['sample_names'] == 'VBF125.root')
]

## do event selection
selected_events = training_processes[
    (training_processes['el_iso'] < 0.1) &
    (training_processes['t1_tightIso'] > 0) &
    (training_processes['nbjets'] == 0) &
    (training_processes['el_charge'] + training_processes['t1_charge'] == 0)
]

training_dataframe = selected_events[
    ['Q2V1', 'Q2V2', 'Phi', 'Phi1', 'costheta1',
        'costheta2', 'costhetastar','isSignal', 'evtwt']
]

training_data, testing_data, training_meta, testing_meta = train_test_split(
    training_dataframe.values[:, :7], training_dataframe.values[:, 7:], test_size=0.1, random_state=7
)

training_labels = training_meta[:, 0]
training_weights = training_meta[:, 1]

## train that there model, my dude
history = model.fit(training_data, training_labels, shuffle=True,
                    epochs=10000, batch_size=1024, verbose=True,
                    callbacks=callbacks, validation_split=0.25, sample_weight=training_weights
                    )


