# This script will use the generated FRUIT spectra to calculate some detector
#  responses using the detector response function/matrix for the PNS.
import pandas as pd
import numpy as np

# The below DRM is from the plane source version
DRM_df = pd.read_csv('C:\\Users\\zacht\\OneDrive\\PhD\\Data\\01-Detector_Response_Matrices\\DRM_PlaneSource_1e10nps_Li6_averagedMeanTallies.csv')
DRM = DRM_df.to_numpy()[:,1:76].T

spectra = np.load('fruitSpectra.npy')

detResponse = np.dot(spectra,DRM)
np.save('fruitDetResponse.npy',detResponse)

spectra_1e5 = np.load('fruitSpectra_1e5.npy')
detResponse_1e5 = np.dot(spectra_1e5,DRM)
np.save('fruitDetResponse_1e5.npy',detResponse_1e5)