# The script will process reported neutron spectra and save them to a csv file
#  that will match the energy bins (in MeV) and interpolate the spectrum
#  emission profile. I have a file, called IAEA_Spectra_original.csv, that I 
#  read in using Pandas and then return out a file with all of the same spectra
#  in my energy bins.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import csv

def interp_spec(spec_name,Ebins_o,E_bins,IAEA_spec_o,spec_names):
    interped_spec = []
    index_spec_o = np.where(spec_names == spec_name)[0][0]
    spec_o = IAEA_spec_o[:,index_spec_o]
    for E in E_bins:
        for i in range(len(Ebins_o)):
            if Ebins_o[i] == E:
                interped_spec.append(spec_o[i]) 
            else:
                if E > Ebins_o[i] and E < Ebins_o[i+1]:
                    interped_spec.append(spec_o[i] +(E-Ebins_o[i])*(spec_o[i+1]-spec_o[i])/(Ebins_o[i+1]-Ebins_o[i]))
                    break
    return np.array(interped_spec)

IAEA_Spec_original_df = pd.read_csv(r'C:\Users\zacht\OneDrive\PhD\Data\03-IAEA_Spectra\IAEA_Spectra_original.csv')
E_bins_original = IAEA_Spec_original_df.to_numpy()[:,0]
IAEA_Spec_original = IAEA_Spec_original_df.to_numpy()[:,1:]
spec_names = IAEA_Spec_original_df.keys()[1:]

E_bins = np.array([1e-9,1.58e-9,2.51e-9,3.98e-9,6.31e-9,
          1e-8,1.58e-8,2.51e-8,3.98e-8,6.31e-8,
          1e-7,1.58e-7,2.51e-7,3.98e-7,6.31e-7,
          1e-6,1.58e-6,2.51e-6,3.98e-6,6.31e-6,
          1e-5,1.58e-5,2.51e-5,3.98e-5,6.31e-5,
          1e-4,1.58e-4,2.51e-4,3.98e-4,6.31e-4,
          1e-3,1.58e-3,2.51e-3,3.98e-3,6.31e-3,
          1e-2,1.58e-2,2.51e-2,3.98e-2,6.31e-2,
          1e-1,1.26e-1,1.58e-1,2e-1,2.51e-1,3.16e-1,3.98e-1,5.01e-1,6.31e-1,7.94e-1,
          1e0,1.12e0,1.26e0,1.41e0,1.58e0,1.78e0,2e0,2.24e0,2.51e0,2.82e0,
          3.16e0,3.55e0,3.98e0,4.47e0,5.01e0,5.62e0,6.31e0,7.08e0,7.94e0,8.91e0,
          1e1,1.12e1,1.26e1,1.41e1,1.58e1,1.78e1,2e1,2.51e1,3.16e1,3.98e1,
          5.01e1,6.31e1,7.94e1,1e2])

num_spectra = 11
spectra = np.zeros((len(E_bins),num_spectra))

for k in range(num_spectra):
    spectra[:,k] = interp_spec(spec_names[k],E_bins_original,E_bins,IAEA_Spec_original,spec_names)

    fig,ax = plt.subplots()
    ax.semilogx(E_bins,spectra[:,k],label='Interpolated Spectrum')
    ax.semilogx(E_bins_original,IAEA_Spec_original[:,k], label='IAEA Spectrum')
    plt.legend()
    plt.title(f'{spec_names[k]}')
    fig.savefig(spec_names[k],dpi=300)

csv_spectra = np.zeros((len(E_bins),num_spectra+1))
csv_spectra[:,0] = E_bins
csv_spectra[:,1:] = spectra

filename = 'IAEA_spectra_84_EBins.csv'
with open(filename,'w',newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(spec_names.insert(0,'Energy'))
    csvwriter.writerows(csv_spectra)

# t_matrix = np.zeros((num_spectra,num_spectra))
# # chi_matrix = np.zeros((num_spectra,num_spectra))
# for i in range(num_spectra):
#     for k in range(i+1):
#         # chi_matrix[i,k] = (spectra[:,i]-spectra[:,k]).T.dot(np.linalg.inv(np.diag(spectra[:,i]*.1))).dot(spectra[:,i]-spectra[:,k])
#         t_matrix[i,k] = round(ttest_ind(spectra[:,i],spectra[:,k])[0],2)
        
# fig,ax = plt.subplots()
# c = plt.imshow(t_matrix)
# plt.colorbar(c)