# This will run all of the case studies for my MAXED portion of my paper.

import numpy as np
import pandas as pd
from scipy.optimize import dual_annealing
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import normalize
import imageio
from scipy.stats import ttest_ind

font = {'size': 12}
matplotlib.rc('font',**font)

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
    best_chi = 1e6
    res = dual_annealing(Z, bounds, args=params, maxiter=mi)  # minimize Z
    unfolded_spec = unfold_Spectrum(gs, res.x, drm)   # unfold Z with lambdas (res.x)
    print(f't-test of spectrum: stat={ttest_ind(unfolded_spec,gs)[0]}, p={ttest_ind(unfolded_spec,gs)[1]}')
    det_res_unfolded = drm.dot(unfolded_spec)
    chi2 = (det_res_unfolded-dr).T.dot(np.linalg.inv(np.diag(dr*.1))).dot(det_res_unfolded-dr)
    if chi2 < best_chi:
        best_chi = chi2
        best_unfolded_spec = unfolded_spec
        best_det_res_unfolded = det_res_unfolded
        best_res = res
        # print(f'The current best chi is: {best_chi} at iteration: {i}')

    unfolded_spec = best_unfolded_spec
    print(f'The best results led to: chi2 = {best_chi}, chi2 per DOF = {best_chi/m}')
    fig,(ax0,ax1) = plt.subplots(1,2)
    fig.suptitle('MAXED Unfolding Spectra Results\n' + f'DRM: {which_drm}\n' + f'Guess Spectrum: {which_spec}*{sm}')
    
    ax0.step(E_bins, unfolded_spec, label='Unfolded')
    ax0.step(E_bins, ts, label='Real', linestyle=':')
    ax0.step(E_bins, gs*sm, 'k', label='Guess', linestyle='--')
    ax0.semilogx()
    ax0.set_ylabel('Fluence per Unit \nLog Energy (Lethargy)')
    ax0.set_xlabel('Energy (MeV)')
    ax0.set_ylim((0,1))
    ax0.legend(loc='upper left', fontsize=8)
    ax0.text(1e-9,.2,f'$\chi^2$ = {round(chi2,2)}')
    ax0.text(1e-9,.1,f't={round(ttest_ind(unfolded_spec,gs)[0],2)}',fontweight='normal')
    ax0.set_title('Spectrum Comparison')
    
    # radii = [14, 13, 12, 11, 10, 9, 8, 6, 3, 0]
    if len(dr) == 10:
        radii = [0, 3, 6, 8, 9, 10, 11, 12, 13, 14]
    else:
        radii = [14,13,12,11,10,9,8,6,3,0,-3,-6,-8,-9,-10,-11,-12,-13,-14,
                 14,13,12,11,10,9,8,6,3,0,-3,-6,-8,-9,-10,-11,-12,-13,-14,
                 14,13,12,11,10,9,8,6,3,0,-3,-6,-8,-9,-10,-11,-12,-13,-14]
    ax1.scatter(radii, dr, s=10, label='Real')
    ax1.errorbar(radii, dr, yerr=dre, capsize=5, linestyle=None)
    ax1.scatter(radii, det_res_unfolded, s=10, label='Unfolded')
    ax1.set_ylabel('Fluence Response (-)')
    ax1.set_xlabel('PNS depth (cm)')
    ax1.set_ylim((0.05,0.5))
    ax1.legend(loc='upper left', fontsize=8)
    ax1.text(6,.1,f't={round(ttest_ind(best_det_res_unfolded,dr)[0],2)}',fontweight='normal')
    ax1.set_title('Detector Response Comparison')

    fig.tight_layout()
    # fig.show()
    fig_name = f'{which_drm}_{which_spec}_{im_num}.png'
    fig.savefig(fig_name,dpi=300)
    fig_names.append(fig_name)
    return fig_names

def make_gif(path,filenames):
    with imageio.get_writer(path, mode='I', duration=0.5) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
###-------------------------------------------------------------------------###
#                 Convert IAEA Cf-252 spectrum into AWE energy bins           #
###-------------------------------------------------------------------------###
Cf_spec_IAEA_df = pd.read_csv(r'C:\Users\zacht\OneDrive\OSU\Research\MyMaxed\CaseStudy\IAEA_Cf252_Bare.csv')
Cf_spec_IAEA = Cf_spec_IAEA_df.to_numpy()[0]
Cf_spec_IAEA_E_bins = Cf_spec_IAEA_df.columns.to_numpy().astype('float64')
E_bins = np.array([1e-9,1.58e-9,2.51e-9,3.98e-9,6.31e-9,
          1e-8,1.58e-8,2.51e-8,3.98e-8,6.31e-8,
          1e-7,1.58e-7,2.51e-7,3.98e-7,6.31e-7,
          1e-6,1.58e-6,2.51e-6,3.98e-6,6.31e-6,
          1e-5,1.58e-5,2.51e-5,3.98e-5,6.31e-5,
          1e-4,1.58e-4,2.51e-4,3.98e-4,6.31e-4,
          1e-3,1.58e-3,2.51e-3,3.98e-3,6.31e-3,
          1e-2,1.58e-2,2.51e-2,3.98e-2,6.31e-2,
          1e-1,1.26e-1,1.58e-1,2e-1,2.51e-1,3.16e-1,3.98e-1,5.01e-1,6.31e-1,7.94e-1,
          1e0,1.12e0,1.26e0,1.41e0,1.58e0,1.78e0,2e0,2.24e0,2.51e0,2.82e0,3.16e0,
          3.55e0,3.98e0,4.47e0,5.01e0,5.62e0,6.31e0,7.08e0,7.94e0,8.91e0,
          1e1,1.12e1,1.26e1,1.41e1,1.58e1,1.78e1,2e1,2.51e1,3.16e1,3.98e1,5.01e1,6.31e1,7.94e1,1e2])
Cf_spec = []
for E in E_bins:
    for i in range(len(Cf_spec_IAEA_E_bins)):
        if Cf_spec_IAEA_E_bins[i] == E:
            Cf_spec.append(Cf_spec_IAEA[i]) 
        else:
            if E > Cf_spec_IAEA_E_bins[i] and E < Cf_spec_IAEA_E_bins[i+1]:
                Cf_spec.append(Cf_spec_IAEA[i] +(E-Cf_spec_IAEA_E_bins[i])*(Cf_spec_IAEA[i+1]-Cf_spec_IAEA[i])/(Cf_spec_IAEA_E_bins[i+1]-Cf_spec_IAEA_E_bins[i]))
                break
Cf_spec = np.array(Cf_spec)

###-------------------------------------------------------------------------###
#                      Import detector response matrices                      #
###-------------------------------------------------------------------------###
# Detector response matrix for the plane source, depth-averaged tallies:
plane_avg_drm_df = pd.read_csv(r'C:\Users\zacht\OneDrive\PhD\Data\02-MAXED\DRM_PlaneSource_1e10nps_Li6_averagedMeanTallies.csv')
plane_avg_drm = plane_avg_drm_df.to_numpy()[0:10,1:].astype('float64')
plane_avg_drm = normalize(plane_avg_drm,axis=1,norm='l1')

# Detector response matrix for the spherical source, depth-averaged tallies:
spheric_avg_drm_df = pd.read_csv(r'C:\Users\zacht\OneDrive\PhD\Data\02-MAXED\DRM_SphericalShellSource_1e10nps_Li6_averagedMeanTallies.csv')
spheric_avg_drm = spheric_avg_drm_df.to_numpy()[0:10,1:].astype('float64')
spheric_avg_drm = normalize(spheric_avg_drm,axis=1,norm='l1')

# Detector response matrix for the plane source, all tallies separate:
plane_drm_df = pd.read_csv(r'C:\Users\zacht\OneDrive\PhD\Data\02-MAXED\DRM_PlaneSource_1e10nps_Li6_meanTallies.csv')
plane_drm = plane_drm_df.to_numpy()[0:57,1:].astype('float64')
plane_drm = normalize(plane_drm,axis=1,norm='l1')

# Detector response matrix for the spherical source, all tallies separate:
spheric_drm_df = pd.read_csv(r'C:\Users\zacht\OneDrive\PhD\Data\02-MAXED\DRM_SphericalShellSource_1e10nps_Li6_meanTallies.csv')
spheric_drm = spheric_drm_df.to_numpy()[0:57,1:].astype('float64')
spheric_drm = normalize(spheric_drm,axis=1,norm='l1')

# Random detector response matrix:
random_drm = np.random.random((10,84))
random_drm = normalize(random_drm,axis=1,norm='l1')

###-------------------------------------------------------------------------###
#                         Import detector responses                           #
###-------------------------------------------------------------------------###
# Detector response from LLNL's run 1, Cf-252 source at 300cm from PNS. I used
#  only the Li6 values rather than subtracting out the Li7 measurement.
dr_LLNL_run1_df = pd.read_csv(r'C:\Users\zacht\OneDrive\PhD\Data\02-MAXED\DR_Cf252_300cm_Li6.csv')
dr_LLNL_run1 = dr_LLNL_run1_df.to_numpy()[0:57,1].astype('float64')

dr_LLNL_run1_avg = np.zeros((10))
for i in range(10):
    dr_LLNL_run1_avg[i] = (dr_LLNL_run1[i]+dr_LLNL_run1[18-i]+dr_LLNL_run1[19+i]+dr_LLNL_run1[37-i]+dr_LLNL_run1[38+i]+dr_LLNL_run1[56-i])/6
dr_LLNL_run1_avg = dr_LLNL_run1_avg/np.linalg.norm(dr_LLNL_run1_avg)
dr_LLNL_run1 = dr_LLNL_run1/np.linalg.norm(dr_LLNL_run1)

# # AWE detector response:
# AWE_dr_df= pd.read_csv(r'C:\Users\zacht\OneDrive\OSU\Research\MyMaxed\CaseStudy\AWE_det_res_info.csv')
# AWE_dr = AWE_dr_df.to_numpy()[:,1]
# AWE_norm = np.linalg.norm(AWE_dr)
# AWE_dr = AWE_dr/AWE_norm
# AWE_dr_error = AWE_dr_df.to_numpy()[:,2]
# AWE_dr_error = AWE_dr_error/AWE_norm

# # Flat guess spectrum:
# flat_spec = np.ones(84)

###-------------------------------------------------------------------------###
#                       Case 1                                                #
#                       DRM: Planar Source                                    #
#                       Detector Response: LLNL                               #
#                       Guess Spectrum: IAEA Cf-252                           #
###-------------------------------------------------------------------------###
spec_mod = 1
maxiter = 1000 # for the maximum iterations that dual_annealing will allow
which_drm = 'Planar_Source_DRM_avg'
which_spec = 'IAEA Cf-252 Spectrum'
fig_names = []
print('Case 1:')
for i in range(5):
    run_case(Cf_spec,spec_mod,plane_avg_drm,dr_LLNL_run1_avg,0.1*dr_LLNL_run1_avg,Cf_spec,maxiter,which_drm,which_spec,i,fig_names)
make_gif(r'C:\Users\zacht\OneDrive\PhD\Data\02-MAXED\gif_planeDRMavg_LLNLrun1.gif',fig_names)

# ###-------------------------------------------------------------------------###
# #                       Case 2                                                #
# #                       DRM: AWE                                              #
# #                       Detector Response: AWE                                #
# #                       Guess Spectrum: flat spectrum                         #
# ###-------------------------------------------------------------------------###
spec_mod = 1
maxiter = 1000 # for the maximum iterations that dual_annealing will allow
which_drm = 'Planar_Source_DRM'
which_spec = 'IAEA Cf-252 Spectrum'
fig_names = []
print('Case 1:')
for i in range(5):
    run_case(Cf_spec,spec_mod,plane_drm,dr_LLNL_run1,0.1*dr_LLNL_run1,Cf_spec,maxiter,which_drm,which_spec,i,fig_names)
make_gif(r'C:\Users\zacht\OneDrive\PhD\Data\02-MAXED\gif_planeDRM_LLNLrun1',fig_names)

# spec_mod = 0.4
# maxiter = 1000
# which_drm = 'AWE_drm'
# which_spec = 'Flat Spectrum'
# fig_names = []
# print('Case 2:')
# for i in range(10):
#     run_case(flat_spec,spec_mod,AWE_drm,AWE_dr,AWE_dr_error,Cf_spec,maxiter,which_drm,which_spec,i,fig_names)
# make_gif(r'C:\Users\zacht\OneDrive\OSU\Research\MyMaxed\CaseStudy\gif_AWEdrm_Flatspec.gif',fig_names)
# ###-------------------------------------------------------------------------###
# #                       Case 3                                                #
# #                       DRM: Paige                                            #
# #                       Detector Response: AWE                                #
# #                       Guess Spectrum: IAEA Cf-252                           #
# ###-------------------------------------------------------------------------###
# spec_mod = 0.9
# maxiter = 1000
# which_drm = 'Paige_drm'
# which_spec = 'IAEA Cf-252 Spectrum'
# fig_names = []
# print('Case 3:')
# for i in range(0):
#     run_case(Cf_spec,spec_mod,paige_drm,AWE_dr,AWE_dr_error,Cf_spec,maxiter,which_drm,which_spec,i,fig_names)
# make_gif(r'C:\Users\zacht\OneDrive\OSU\Research\MyMaxed\CaseStudy\gif_Paigedrm_IAEAspec.gif',fig_names)
# ###-------------------------------------------------------------------------###
# #                       Case 4                                                #
# #                       DRM: Zach                                             #
# #                       Detector Response: AWE                                #
# #                       Guess Spectrum: IAEA Cf-252                           #
# ###-------------------------------------------------------------------------###
# spec_mod = 0.9
# maxiter = 1000
# which_drm = 'Zach_drm'
# which_spec = 'IAEA Cf-252 Spectrum'
# fig_names = []
# print('Case 4:')
# for i in range(0):
#     run_case(Cf_spec,spec_mod,zach_drm,AWE_dr,AWE_dr_error,Cf_spec,maxiter,which_drm,which_spec,i,fig_names)
# print(fig_names)
# make_gif(r'C:\Users\zacht\OneDrive\OSU\Research\MyMaxed\CaseStudy\gif_Zachdrm_IAEAspec.gif',fig_names)
# ###-------------------------------------------------------------------------###
# #                       Case 5                                                #
# #                       DRM: random                                           #
# #                       Detector Response: AWE                                #
# #                       Guess Spectrum: IAEA Cf-252                           #
# ###-------------------------------------------------------------------------###
# spec_mod = 0.9
# maxiter = 1000
# which_drm = 'Random_drm'
# which_spec = 'IAEA Cf-252 Spectrum'
# fig_names = []
# print('Case 5:')
# for i in range(0):
#     run_case(Cf_spec,spec_mod,random_drm,AWE_dr,AWE_dr_error,Cf_spec,maxiter,which_drm,which_spec,i,fig_names)
# make_gif(r'C:\Users\zacht\OneDrive\OSU\Research\MyMaxed\CaseStudy\gif_Randdrm_IAEAspec.gif',fig_names)
# ###-------------------------------------------------------------------------###
# #                       Case 6                                                #
# #                       DRM: random                                           #
# #                       Detector Response: AWE                                #
# #                       Guess Spectrum: Flat spectrum                         #
# ###-------------------------------------------------------------------------###
# spec_mod = 0.3
# maxiter = 1000
# which_drm = 'Random_drm'
# which_spec = 'Flat Spectrum'
# fig_names = []
# print('Case 6:')
# for i in range(10):
#     run_case(flat_spec,spec_mod,random_drm,AWE_dr,AWE_dr_error,Cf_spec,maxiter,which_drm,which_spec,i,fig_names)
# make_gif(r'C:\Users\zacht\OneDrive\OSU\Research\MyMaxed\CaseStudy\gif_Randdrm_Flatspec.gif',fig_names)
# ###-------------------------------------------------------------------------###
# #                       Case 7                                                #
# #                       DRM: Zach                                             #
# #                       Detector Response: AWE                                #
# #                       Guess Spectrum: permuted IAEA Cf-252                  #
# ###-------------------------------------------------------------------------###
# # spec_mod = 0.9
# # maxiter = 1000
# # which_drm = 'Zach_drm'
# # which_spec = 'IAEA Cf-252 Spectrum'
# # fig_names = []
# # permute_spec = np.zeros(len(E_bins))
# # permute_amount = 30
# # permute_spec[0:-permute_amount] = Cf_spec[permute_amount:]
# # permute_spec[-permute_amount:] = Cf_spec[0:permute_amount]
# # print('Case 7:')
# # for i in range(0):
# #     run_case(permute_spec,spec_mod,zach_drm,AWE_dr,AWE_dr_error,Cf_spec,maxiter,which_drm,which_spec,i,fig_names)
# # make_gif(r'C:\Users\zacht\OneDrive\OSU\Research\MyMaxed\CaseStudy\gif_Zachdrm_IAEAspec.gif',fig_names)