import pandas as pd
import numpy as np
import tensorflow.keras
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split

def loadData():
    # This function pulls in the unfolding data, which is from the IAEA tecdoc.
    #  It contains 251 detector responses that correspond to various spectra,
    #  which are noted in the unfolding data key: 'Spectrum Name'. The neural
    #  network will take as an input the variable X, which is the detector
    #  responses of a Bonner sphere system. The corresponding spectrum that 
    #  generates each detector response is in the variable Y.
    # Variables:
        # unfolding_data    - pandas dataframe
        # num_data          - number of data entries
        # X                 - detector responses
        # Y                 - corresponding spectra
    # This function returns the variables X and Y
    unfolding_data = pd.read_pickle("unfolding_data.pkl")
    num_data = len(unfolding_data)
    X = np.zeros((num_data,15))
    Y = np.zeros((num_data,60))
    for i in range(num_data):
        X[i,:] = unfolding_data['Detector Response'][i]
        Y[i,:] = unfolding_data['Spectrum'][i]
    return pd.DataFrame(X),pd.DataFrame(Y)

###-------------------------------------------------------------------------###
#                              LOAD DATA                                      #
###-------------------------------------------------------------------------###
# This portion of the code loads in the data, which consists of spectra and   #
#  detector responses. It splits the data into a training set and validation  #
#  set. Following the split, the X data (detector responses) is pre-processed #
#  using make_column_transformer to transform.                                #
###-------------------------------------------------------------------------###
[X,Y] = loadData()

X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y,test_size=0.3,random_state=42)

# preprocessor = make_column_transformer(
#     (StandardScaler(), make_column_selector(dtype_include=np.number)))
# X_train = preprocessor.fit_transform(X_train)
# X_valid = preprocessor.transform(X_valid)
# Y_train = preprocessor.fit_transform(Y_train)
# Y_valid = preprocessor.transform(Y_valid)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_valid = sc.transform(X_valid)

###-------------------------------------------------------------------------###
#                           BUILDING THE MODEL                                #
###-------------------------------------------------------------------------###
# This portion of the code builds the neural network using tensorflow and     #
#  Keras.
# early_stopping is used to terminate the training once there are no more     #
#  significant gains in the training. The min_delta relates to the change in  #
#  loss over each epoch. The variable patience refers to the number of epochs #
#  that will run after the min_delta has been reached. The variable           #
#  restore_best_weights will restore the weights at which the best min_delta  #
#  was reached.                                                               #
# model defines the neural network.                                           #
###-------------------------------------------------------------------------###
early_stopping = callbacks.EarlyStopping(
    min_delta=0.000001,
    patience=20,
    restore_best_weights=True
    )

model = keras.Sequential([
    layers.Dense(units=512, activation='relu',input_shape=[15]),
    layers.Dense(units=256, activation='relu'),
    layers.Dense(units=128, activation='relu'),
    layers.Dense(units=60)
    ])

model.compile(optimizer='adam', loss='mse')

###-------------------------------------------------------------------------###
#                           FIT/TRAIN THE MODEL                               #
###-------------------------------------------------------------------------###
# This portion performs the actual training and fitting for the neural network#
#  The variables X_train and Y_train are used to make the network and the     #
#  X_valid, Y_valid variables are used to test the model at each epoch. The   #
#  variable epochs limits the total number of epochs that this network will be#
#  allowed to train on. In reality, the callback early_stopping will terminate#
#  the training before 500 epochs is reached. The batch size refers to the    #
#  total number of datasets that will be used to train the network at each    #
#  point in the epoch.                                                        #
###-------------------------------------------------------------------------###
history = model.fit(
    X_train,Y_train,
    validation_data=(X_valid, Y_valid),
    epochs=500,
    batch_size=50,
    verbose=1,
    callbacks = [early_stopping]
    )

history_df = pd.DataFrame(history.history)
# history_df.loc[:,['loss']].plot()
history_df.loc[5:,['loss', 'val_loss']].plot()
# # fig.title('title')
tensorflow.keras.backend.clear_session()