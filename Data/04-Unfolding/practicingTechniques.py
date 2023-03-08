import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split

plt.rcParams.update({'font.size': 12})

def loadData_IAEA():
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

def loadData_FRUIT():
    X = np.load('C:\\Users\\zacht\\OneDrive\\PhD\\Data\\05-FRUIT_Spectra\\fruitDetResponse.npy',allow_pickle=(True))
    Y = np.load('C:\\Users\\zacht\\OneDrive\\PhD\\Data\\05-FRUIT_Spectra\\fruitSpectra.npy',allow_pickle=(True))
    return pd.DataFrame(X),pd.DataFrame(Y)

###-------------------------------------------------------------------------###
#                              LOAD DATA                                      #
###-------------------------------------------------------------------------###
# This portion of the code loads in the data, which consists of spectra and   #
#  detector responses. It splits the data into a training set and validation  #
#  set. Following the split, the X data (detector responses) is pre-processed #
#  using make_column_transformer to transform.                                #
###-------------------------------------------------------------------------###
# [X,Y] = loadData_IAEA()
[X,Y] = loadData_FRUIT()
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
    min_delta=0.00000001,
    patience=20,
    restore_best_weights=True
    )

model = keras.Sequential([
    layers.Dense(units=512, activation='relu',input_shape=[10]),
    layers.Dropout(rate=0.3),
    layers.Dense(units=256, activation='relu'),
    layers.Dropout(rate=0.3),
    layers.Dense(units=128, activation='relu'),
    layers.Dropout(rate=0.3),
    layers.Dense(units=75)
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
    epochs=10000,
    batch_size=1000,
    verbose=1,
    callbacks = [early_stopping]
    )

history_df = pd.DataFrame(history.history)
# history_df.loc[:,['loss']].plot()
history_df.loc[5:,['loss', 'val_loss']].plot()
# fig.title('title')
tensorflow.keras.backend.clear_session()

AWE_detres = np.array([10.89,15.28,22.76,25.63,27.68,29.59,28.3,26.94,25.11,23.56]).reshape(-1,1)
AWE_detres = AWE_detres/np.linalg.norm(AWE_detres)
pred_spec = model.predict(AWE_detres.T)
Ebins = np.array([1e-9,1.58e-9,2.51e-9,3.98e-9,6.31e-9,1e-8,1.58e-8,2.51e-8,3.98e-8,
         6.31e-8,1e-7,1.58e-7,2.51e-7,3.98e-7,6.31e-7,1e-6,1.58e-6,2.51e-6,
         3.98e-6,6.31e-6,1e-5,1.58e-5,2.51e-5,3.98e-5,6.31e-5,1e-4,1.58e-4,
         2.51e-4,3.98e-4,6.31e-4,1e-3,1.58e-3,2.51e-3,3.98e-3,6.31e-3,1e-2,
         1.58e-2,2.51e-2,3.98e-2,6.31e-2,1e-1,1.26e-1,1.58e-1,2e-1,2.51e-1,
         3.16e-1,3.98e-1,5.01e-1,6.31e-1,7.94e-1,1e0,1.12e0,1.26e0,1.41e0,
         1.58e0,1.78e0,2e0,2.24e0,2.51e0,2.82e0,3.16e0,3.55e0,3.98e0,4.47e0,
         5.01e0,5.62e0,6.31e0,7.08e0,7.94e0,8.91e0,1e1,1.12e1,1.26e1,1.41e1,
         1.58e1])
plt.semilogx(Ebins,pred_spec.T)
plt.xlabel('Energy (keV)')
plt.ylabel('')