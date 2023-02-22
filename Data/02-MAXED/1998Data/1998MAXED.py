import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing
from scipy.stats import ttest_ind
from sklearn.preprocessing import normalize

def Z(lamb, *params):
    # This function gets passed into the dual annealing function from scipy. 
    # This function
    # guess_spec     = the guess spectrum for the dual_annealing process
    # det_res_matrix = the response function that will be unfolded
    # std            = the standard uncertainty of the detector response
    # det_res        = actual measurement from the detectors
    # lamb           = hyperparameters for minimization
    guess_spec, det_res_matrix, std, det_res = params
    omega = det_res.size
    # print(f'det_res_matrix.shape[0] is {det_res_matrix.shape[0]}')
    Z = np.abs(-guess_spec.dot(np.exp(-lamb.dot(det_res_matrix)))
               -(omega*(lamb.dot(std))**2)**0.5
               -det_res.dot(lamb))
    return Z

def unfold_Spectrum(guess_spec, lamb, det_res_matrix):
    # print(lamb.dot(det_res_matrix))
    print(-lamb.dot(det_res_matrix))
    return guess_spec*np.exp(-lamb.dot(det_res_matrix))

def run_case(gs, sm, drm, dr, dre, ts, mi, which_drm, which_spec, im_num, fig_names):
    # Variables:
        # gs  - guess spectrum
        # sm  - spectrum modifier
        # drm - detector response matrix
        # dr  - detector response
        # dre - detector response error
        # ts  - true spectrum
        # mi  - maximum iterations for dual annealing
    params = (gs*sm, drm, dr*.1, dr)
    m = len(dr) # number of detectors in the detector response.
    bounds = list(zip([-10]*m, [10]*m))
    res = dual_annealing(Z, bounds, args=params, maxiter=mi)  # minimize Z
    unfolded_spec = unfold_Spectrum(gs, res.x, drm)   # unfold Z with lambdas (res.x)
    print(f't-test of spectrum: stat={ttest_ind(unfolded_spec,gs)[0]}, p={ttest_ind(unfolded_spec,gs)[1]}')
    det_res_unfolded = drm.dot(unfolded_spec)
    chi2_dr = (det_res_unfolded-dr).T.dot(np.linalg.inv(np.diag(dr*.1))).dot(det_res_unfolded-dr)
    MAC = np.dot(unfolded_spec,ts)**2/(np.dot(unfolded_spec,unfolded_spec)*np.dot(ts,ts))
    # chi2_spec = (unfolded_spec-ts).T.dot(np.linalg.inv(np.diag((ts+1e5)*.1))).dot(unfolded_spec-ts)

    print('The best results led to:')
    print(f'       For the Detector Response: chi2 = {chi2_dr}, chi2 per DOF = {chi2_dr/m}')
    print(f'       For the Spectrum: MAC = {MAC}')
    fig,(ax0,ax1) = plt.subplots(1,2)
    fig.suptitle('MAXED Unfolding Spectra Results\n' + f'DRM: {which_drm}\n' + f'Guess Spectrum: {which_spec}*{sm}')
    
    ax0.step(Ebins, unfolded_spec, label='Unfolded', color='#1f77b4')
    ax0.step(Ebins, ts, label='Real', linestyle=':', color='#ff7f0e')
    ax0.step(Ebins, gs*sm, 'k', label='Guess', linestyle='--')
    ax0.semilogx()
    ax0.set_ylabel('Fluence per Unit \nLog Energy (Lethargy)')
    ax0.set_xlabel('Energy (MeV)')
    # ax0.set_ylim((0,1))
    ax0.legend(loc='upper left', fontsize=8)
    # ax0.text(1e-9,.2,f'$\chi^2$ = {round(chi2_dr,2)}')
    # ax0.text(1e-9,.2,f'MAC = {round(MAC,5)}')
    # ax0.text(1e-9,.1,f't={round(ttest_ind(unfolded_spec,gs)[0],2)}',fontweight='normal')
    ax0.set_title('Spectrum Comparison')
    
    radii = [3,5,7,8,9,10,11,12]
    ax1.scatter(radii, det_res_unfolded, s=10, label='Unfolded', color='#1f77b4')
    ax1.scatter(radii, dr, s=10, label='Real', color='#ff7f0e')
    ax1.errorbar(radii, dr, yerr=dre, capsize=5, linestyle=None, color='#ff7f0e')
    ax1.set_ylabel('Fluence Response (-)')
    ax1.set_xlabel('PNS depth (cm)')
    # ax1.set_ylim((0.01,0.2))
    ax1.legend(loc='upper left', fontsize=8)
    # ax1.text(8,.15,f'$\chi^2$ = {round(chi2_dr,2)}')
    # ax1.text(6,.1,f't={round(ttest_ind(det_res_unfolded,dr)[0],2)}',fontweight='normal')
    ax1.set_title('Detector Response Comparison')

    fig.tight_layout()
    # fig.show()
    fig_name = f'{which_drm}_{which_spec}_{im_num}.png'
    fig.savefig(fig_name,dpi=300)
    fig_names.append(fig_name)
    return fig_names

###-------------------------------------------------------------------------###
#                       READ IN THE DETECTOR RESPONSE                         #
# The csv file detectorResponse.txt contains the information for the detector #
#  response as well as the error associated with that detection. The original #
#  MAXED authors used 8 out of 12 detectors for their code. The first column  #
#  are the numbers associated with the detectors, the second column contains  #
#  the counts for each detector, and the third column contains the error      #
#  associated with each count (the authors just used the sqrt of counts).     #
# DR_df is the pandas dataframe after reading in the csv file                 #
# DR is the counts column from the dataframe                                  #
# DR_error is the error column from the dataframe                             #
###-------------------------------------------------------------------------###
DR_df = pd.read_csv(r'C:\Users\zacht\OneDrive\PhD\Data\02-MAXED\1998Data\detectorResponse.txt')
DR = DR_df.to_numpy()[:,1].astype('float64')
x = sum(DR)
DR = DR/sum(DR)
DR_error = DR_df.to_numpy()[:,2].astype('float64')
DR_error= DR_error/x

###-------------------------------------------------------------------------###
#                  READ IN THE DETECTOR RESPONSE FUNCTION                     #
# The csv file ResponseFunction.txt contains the information for the detector #
#  response function. The column headers give information for each column.    #
#  Column 1 contains the energy bin information and the other columns contain #
#  the detector response for each of the original 12 detectors. To match with #
#  the work that the original authors did, I pulled the detector information  #
#  for the 8 detectors they used (3,5,7,8,9,10,11,12). These numbers can be   #
#  seen in the first column of DR_df.                                         #
# DRF_df is the pandas dataframe after reading in the csv file                #
# DRF is detector response function with the aforementioned detectors         #
# Ebins is the energy bin information                                         #
###-------------------------------------------------------------------------###
DRF_df = pd.read_csv(r'C:\Users\zacht\OneDrive\PhD\Data\02-MAXED\1998Data\ResponseFunction.txt')
DRF = DRF_df.to_numpy()[:,[3,5,7,8,9,10,11,12]].astype('float64')
DRF = DRF.T
DRF = normalize(DRF,axis=1,norm='l1')
Ebins = DRF_df.to_numpy()[:,0].astype('float64')

###-------------------------------------------------------------------------###
#                       READ IN THE DEFAULT SPECTRUM                          #
# The csv file defaultSpectrum.txt contains the information for the default   #
#  spectrum that is used as the initial guess for MAXED. This spectrum was    #
#  obtained by the authors and the reference is in the 1998 paper. There are  #
#  only 20 energy bins in this spectrum, compared to 227 in the detector      #
#  response function. The authors used linear interpolation to expand the     #
#  default spectrum to fit the detector response function.                    #
# DRF_df is the pandas dataframe after reading in the csv file                #
# DRF is detector response function with the aforementioned detectors         #
# Ebins is the energy bin information                                         #
###-------------------------------------------------------------------------###
d_spec_df = pd.read_csv(r'C:\Users\zacht\OneDrive\PhD\Data\02-MAXED\1998Data\defaultSpectrum.txt')
d_spec_eBins = d_spec_df.to_numpy()[:,0].astype('float64')
d_spec = d_spec_df.to_numpy()[:,1].astype('float64')
d_spec_pL = np.zeros((21))
for i in range(20):
    d_spec_pL[i] = d_spec[i]/np.log(max(d_spec_eBins)/d_spec_eBins[i])
d_spec_pL = d_spec_pL/sum(d_spec_pL)
###-------------------------------------------------------------------------###
#                     INTERPOLATE THE REQUIRED VALUES                         #
# 
# DRF_df is the pandas dataframe after reading in the csv file                #
# DRF is detector response function with the aforementioned detectors         #
# Ebins is the energy bin information                                         #
###-------------------------------------------------------------------------###
d_spec_interp = np.interp(Ebins,d_spec_eBins,d_spec_pL)

run_case(d_spec_interp,1,DRF,DR,DR_error,d_spec_interp,1000,'MAXED','MAXED_spec',0,[])