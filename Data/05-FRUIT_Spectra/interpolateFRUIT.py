# This code will interpolate the generated FRUIT spectra from James McGreivy.

import numpy as np

spectra_og = np.load('fruitSpectra_original.npy')
spectra_og1e5 = np.load('fruitSpectra_original1e5.npy')
Ebins_og = np.load('fruitEbins_original.npy') *1e-6 # To convert from Mev to eV

# Ebins = np.array([1e-9,1.58e-9,2.51e-9,3.98e-9,6.31e-9,1e-8,1.58e-8,2.51e-8,3.98e-8,
#          6.31e-8,1e-7,1.58e-7,2.51e-7,3.98e-7,6.31e-7,1e-6,1.58e-6,2.51e-6,
#          3.98e-6,6.31e-6,1e-5,1.58e-5,2.51e-5,3.98e-5,6.31e-5,1e-4,1.58e-4,
#          2.51e-4,3.98e-4,6.31e-4,1e-3,1.58e-3,2.51e-3,3.98e-3,6.31e-3,1e-2,
#          1.58e-2,2.51e-2,3.98e-2,6.31e-2,1e-1,1.26e-1,1.58e-1,2e-1,2.51e-1,
#          3.16e-1,3.98e-1,5.01e-1,6.31e-1,7.94e-1,1e0,1.12e0,1.26e0,1.41e0,
#          1.58e0,1.78e0,2e0,2.24e0,2.51e0,2.82e0,3.16e0,3.55e0,3.98e0,4.47e0,
#          5.01e0,5.62e0,6.31e0,7.08e0,7.94e0,8.91e0,1e1,1.12e1,1.26e1,1.41e1,
#          1.58e1,1.78e1,2e1,2.51e1,3.16e1,3.98e1,5.01e1,6.31e1,7.94e1,1e2])

Ebins = np.array([1e-9,1.58e-9,2.51e-9,3.98e-9,6.31e-9,1e-8,1.58e-8,2.51e-8,3.98e-8,
         6.31e-8,1e-7,1.58e-7,2.51e-7,3.98e-7,6.31e-7,1e-6,1.58e-6,2.51e-6,
         3.98e-6,6.31e-6,1e-5,1.58e-5,2.51e-5,3.98e-5,6.31e-5,1e-4,1.58e-4,
         2.51e-4,3.98e-4,6.31e-4,1e-3,1.58e-3,2.51e-3,3.98e-3,6.31e-3,1e-2,
         1.58e-2,2.51e-2,3.98e-2,6.31e-2,1e-1,1.26e-1,1.58e-1,2e-1,2.51e-1,
         3.16e-1,3.98e-1,5.01e-1,6.31e-1,7.94e-1,1e0,1.12e0,1.26e0,1.41e0,
         1.58e0,1.78e0,2e0,2.24e0,2.51e0,2.82e0,3.16e0,3.55e0,3.98e0,4.47e0,
         5.01e0,5.62e0,6.31e0,7.08e0,7.94e0,8.91e0,1e1,1.12e1,1.26e1,1.41e1,
         1.58e1]) # Until I generate FRUIT spectra myself, I'm limited to this maximum energy

# 10000 different spectra
spectra = np.zeros((spectra_og.shape[0],len(Ebins)))
for i in range(spectra_og.shape[0]):
    spectra[i] = np.interp(Ebins,Ebins_og,spectra_og[i])
np.save('fruitSpectra',spectra)

# 100000 different spectra
spectra = np.zeros((spectra_og1e5.shape[0],len(Ebins)))
for i in range(spectra_og1e5.shape[0]):
    spectra[i] = np.interp(Ebins,Ebins_og,spectra_og1e5[i])
np.save('fruitSpectra_1e5',spectra)