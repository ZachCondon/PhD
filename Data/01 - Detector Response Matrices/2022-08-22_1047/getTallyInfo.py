# This code will save the data from my 1e10 run that will compare to my old
#  6e8 run and Paige's 5e9 run. This saves the data to a CSV file so that the 
#  files don't need to be scraped again.

import dataManipulation as dm

E_bin_names = ["1e-9MeV","158e-9MeV","251e-9MeV","398e-9MeV","631e-9MeV",
               "1e-8MeV","158e-8MeV","251e-8MeV","398e-8MeV","631e-8MeV",
               "1e-7MeV","158e-7MeV","251e-7MeV","398e-7MeV","631e-7MeV",
               "1e-6MeV","158e-6MeV","251e-6MeV","398e-6MeV","631e-6MeV",
               "1e-5MeV","158e-5MeV","251e-5MeV","398e-5MeV","631e-5MeV",
               "1e-4MeV","158e-4MeV","251e-4MeV","398e-4MeV","631e-4MeV",
               "1e-3MeV","158e-3MeV","251e-3MeV","398e-3MeV","631e-3MeV",
               "1e-2MeV","158e-2MeV","251e-2MeV","398e-2MeV","631e-2MeV",
               "1e-1MeV","126e-1MeV","158e-1MeV","2e-1MeV","251e-1MeV",
               "316e-1MeV","398e-1MeV","501e-1MeV","631e-1MeV","794e-1MeV",
               "1e0MeV","112e0MeV","126e0MeV","141e0MeV","158e0MeV",
               "178e0MeV","2e0MeV","224e0MeV","251e0MeV","282e0MeV",
               "316e0MeV","355e0MeV","398e0MeV","447e0MeV","501e0MeV",
               "562e0MeV","631e0MeV","708e0MeV","794e0MeV","891e0MeV",
               "1e1MeV","112e1MeV","126e1MeV","141e1MeV","158e1MeV",
               "178e1MeV","2e1MeV","251e1MeV","316e1MeV","398e1MeV",
               "501e1MeV","631e1MeV","794e1MeV","1e2MeV"]

tally_names = ['4006','4016','4026','4036','4046','4056','4066','4076','4086',
               '4096','4106','4116','4126','4136','4146','4156','4166','4176',
               '4186','4196','4206','4216','4226','4236','4246','4256','4266',
               '4276','4286','4296','4306','4316','4326','4336','4346','4356',
               '4366','4376','4386','4396','4406','4416','4426','4436','4446',
               '4456','4466','4476','4486','4496','4506','4516','4526','4536',
               '4546']

# The following code reads in all of my output files to the 2D list that
#  contains all the mean tally information from my output files. See the
#  dataManipulation function for how that works. The first index for the 
#  variable "mean_tallys_zach_1e10" is associated with the energy of the output
#  file and the second index is associated with the tally number. There are 84
#  energy bins
mean_tallys_zach_e10 = []
error_tallys_zach_e10 = []
vov_tallys_zach_e10 = []
slope_tallys_zach_e10 = []
for E_bin in E_bin_names:
    mean_list, error_list, vov_list, slope_list = dm.get_all_tally_info('out_PNS_'+E_bin, 10000000000)
    mean_tallys_zach_e10.append(mean_list)
    error_tallys_zach_e10.append(error_list)
    vov_tallys_zach_e10.append(vov_list)
    slope_tallys_zach_e10.append(slope_list)

dm.save_data('mean_tallys_zach_e10.csv',tally_names,mean_tallys_zach_e10)
dm.save_data('error_tallys_zach_e10.csv',tally_names,error_tallys_zach_e10)
dm.save_data('vov_tallys_zach_e10.csv',tally_names,vov_tallys_zach_e10)
dm.save_data('slope_tallys_zach_e10.csv',tally_names,slope_tallys_zach_e10)