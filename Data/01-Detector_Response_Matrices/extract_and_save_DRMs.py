# The object of this script is to take the pickle files from the scraped MCNP
#  files and save the detector response information as .csv files.
# Instructions for using this script:
    # 1. Run this script (nothing will occur at this point). There is now a 
    #     variable called "data" that will be used to extract the DRMs
    # 2. For each detector response matrix, there are two commands:
        # A. data.save_tallies()
            # After running this command, you will be given a list and a 
            #  prompt to choose a file. The file name tells you, in order,
            #  it's for a DRM, what the source of neutrons was, the nps, and
            #  the detector material. Choose an option based on the number
            #  and hit enter.
            # NOTE: It will ask multiple times, choose the same option each
            #        time. To choose a different option, run this command
            #        again once you are finished with the first time.
        # B. data.save_averaged_tallies()
            # After running this command, you will be given a list and a 
            #  prompt to choose a file. The file name tells you, in order,
            #  it's for a DRM, what the source of neutrons was, the nps, and
            #  the detector material. Choose an option based on the number
            #  and hit enter.
            # NOTE: It will ask multiple times, choose the same option each
            #        time. To choose a different option, run this command
            #        again once you are finished with the first time.
    # 3. After running these commands, there will be a meanTallies csv and an
    #     errorTallies csv. The first row of each is the energy associated with
    #     each detector response. To make these compatible with the MAXED code
    #     that I have, you will need to manually add a column to the csv file.
    #    The information to add for the averagedMeanTallies.csv and
    #     averagedErrorTallies.csv is (this needs to be inserted as a column):
    # [Energy (MeV)/Depth, 1cm, 2cm, 3cm, 4cm, 5cm, 6cm, 7cm, 9cm, 12cm, 15cm]^T
    #    The information to add for the meanTallies.csv and errorTallies.csv
    #     is (again, inserted as a column):
    # [Energy (MeV)/Depth, 1cm, 2cm, 3cm, 4cm, 5cm, 6cm, 7cm, 9cm, 12cm, 15cm,
    #  18cm, 21cm, 23cm, 24cm, 25cm, 26cm, 27cm, 28cm, 29cm, 1cm, 2cm, 3cm, 
    #  4cm, 5cm, 6cm, 7cm, 9cm, 12cm, 15cm, 18cm, 21cm, 23cm, 24cm, 25cm, 26cm,
    #  27cm, 28cm, 29cm, 1cm, 2cm, 3cm, 4cm, 5cm, 6cm, 7cm, 9cm, 12cm, 15cm, 
    #  18cm, 21cm, 23cm, 24cm, 25cm, 26cm, 27cm, 28cm, 29cm]^T 
 
import glob
import pickle
import numpy as np
import csv

class allData:
    def __init__(self):
        
        pickle_dict = {}
        pickle_files = glob.glob("*.pickle")
        for file in pickle_files:
            file_ID = file[0:-7]
            with open(file,'rb') as f:
                data = pickle.load(f)
            pickle_dict[file_ID] = data
        allData.data = pickle_dict
        allData.X_tlds = ['4186','4166','4146','4126','4106','4086','4066','4046',
                          '4026','4006','4016','4036','4056','4076','4096','4116',
                          '4136','4156','4176']
        allData.Y_tlds = ['4366','4346','4326','4306','4286','4266','4246','4226',
                          '4206','4006','4196','4216','4236','4256','4276','4296',
                          '4316','4336','4356']
        allData.Z_tlds = ['4546','4526','4506','4486','4466','4446','4426','4406',
                          '4386','4006','4376','4396','4416','4436','4456','4476',
                          '4496','4516','4536']
        allData.numerical_E_bins = ['1e-9MeV','1.58e-9MeV','2.51e-9MeV','3.98e-9MeV','6.31e-9MeV',
          '1e-8MeV','1.58e-8MeV','2.51e-8MeV','3.98e-8MeV','6.31e-8MeV',
          '1e-7MeV','1.58e-7MeV','2.51e-7MeV','3.98e-7MeV','6.31e-7MeV',
          '1e-6MeV','1.58e-6MeV','2.51e-6MeV','3.98e-6MeV','6.31e-6MeV',
          '1e-5MeV','1.58e-5MeV','2.51e-5MeV','3.98e-5MeV','6.31e-5MeV',
          '1e-4MeV','1.58e-4MeV','2.51e-4MeV','3.98e-4MeV','6.31e-4MeV',
          '1e-3MeV','1.58e-3MeV','2.51e-3MeV','3.98e-3MeV','6.31e-3MeV',
          '1e-2MeV','1.58e-2MeV','2.51e-2MeV','3.98e-2MeV','6.31e-2MeV',
          '1e-1MeV','1.26e-1MeV','1.58e-1MeV','2e-1MeV','2.51e-1MeV',
          '3.16e-1MeV','3.98e-1MeV','5.01e-1MeV','6.31e-1MeV','7.94e-1MeV',
          '1e0MeV','1.12e0MeV','1.26e0MeV','1.41e0MeV','1.58e0MeV','1.78e0MeV',
          '2e0MeV','2.24e0MeV','2.51e0MeV','2.82e0MeV','3.16e0MeV','3.55e0MeV',
          '3.98e0MeV','4.47e0MeV','5.01e0MeV','5.62e0MeV','6.31e0MeV',
          '7.08e0MeV','7.94e0MeV','8.91e0MeV','1e1MeV','1.12e1MeV','1.26e1MeV',
          '1.41e1MeV','1.58e1MeV','1.78e1MeV','2e1MeV','2.51e1MeV','3.16e1MeV',
          '3.98e1MeV','5.01e1MeV','6.31e1MeV','7.94e1MeV','1e2MeV']
        allData.E_bins = [1e-9,1.58e-9,2.51e-9,3.98e-9,6.31e-9,
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
          5.01e1,6.31e1,7.94e1,1e2]
        allData.equal_distance_TLDs = [['4186','4176','4366','4356','4546','4536'], # 14 cm from center
                                       ['4166','4156','4346','4336','4526','4516'], # 13 cm from center
                                       ['4146','4136','4326','4316','4506','4496'], # 12 cm
                                       ['4126','4116','4306','4296','4486','4476'], # 11 cm
                                       ['4106','4096','4286','4276','4466','4456'], # 10 cm
                                       ['4086','4076','4266','4256','4446','4436'], #  9 cm
                                       ['4066','4056','4246','4236','4426','4416'], #  8 cm
                                       ['4046','4036','4226','4216','4406','4396'], #  6 cm
                                       ['4026','4016','4206','4196','4386','4376'], #  3 cm
                                       ['4006']]   
        
    def print_available_datasets(self):
        print('Available datasets:')
        [print(f'{i}. {list(self.data.keys())[i]}') for i in range(len(list(self.data.keys())))]
        selection = int(input('Which dataset do you want? '))
        return list(self.data.keys())[selection]
    
    def get_averaged_DRF(self):
        dataset = self.print_available_datasets()
        averageTallies = np.zeros((10,84))
        for i in range(84):
            for edTLDs in range(len(self.equal_distance_TLDs)):
                for edTLD in self.equal_distance_TLDs[edTLDs]:
                    averageTallies[edTLDs,i] += self.data[dataset][self.numerical_E_bins[i]]['Mean'][edTLD]
                averageTallies[edTLDs,i] = averageTallies[edTLDs,i]/len(self.equal_distance_TLDs[edTLDs])
        return averageTallies, dataset
    
    def get_averaged_error(self):
        dataset = self.print_available_datasets()
        relErrorTallies = np.zeros((10,84))
        for i in range(84):
            for edTLDs in range(len(self.equal_distance_TLDs)):
                for edTLD in self.equal_distance_TLDs[edTLDs]:
                    relErrorTallies[edTLDs,i] += self.data[dataset][self.numerical_E_bins[i]]['Error'][edTLD]*self.data[dataset][self.numerical_E_bins[i]]['Mean'][edTLD]
                relErrorTallies[edTLDs,i] = relErrorTallies[edTLDs,i]/len(self.equal_distance_TLDs[edTLDs])
        return relErrorTallies, dataset
    
    def get_axis_DRF(self,axisTLDs):
        dataset = self.print_available_datasets()
        axisTallies = np.zeros((19,84))
        for TLD in axisTLDs:
            for energy in allData.numerical_E_bins:
                axisTallies[axisTLDs.index(TLD),allData.numerical_E_bins.index(energy)] = self.data[dataset][energy]['Mean'][TLD]
        return axisTallies
    
    def get_axis_error(self,axisTLDs):
        dataset = self.print_available_datasets()
        axisTallies = np.zeros((19,84))
        for TLD in axisTLDs:
            for energy in allData.numerical_E_bins:
                axisTallies[axisTLDs.index(TLD),allData.numerical_E_bins.index(energy)] = self.data[dataset][energy]['Error'][TLD]
        return axisTallies
    
    def save_averaged_tallies(self):
        averageTallies, dataset = self.get_averaged_DRF()
        averageTalliesName = dataset + '_averagedMeanTallies.csv'
        with open(averageTalliesName,'w',newline='') as f:
            write = csv.writer(f)
            write.writerow(self.E_bins)
            write.writerows(averageTallies)
        
        averageErrorTallies, dataset = self.get_averaged_error()
        averageErrorName = dataset + '_averagedErrorTallies.csv'
        with open(averageErrorName,'w',newline='') as f:
            write = csv.writer(f)
            write.writerow(self.E_bins)
            write.writerows(averageErrorTallies)
    
    def save_tallies(self):
        dataset = self.print_available_datasets()
        xTallies = self.get_axis_DRF(self.X_tlds)
        yTallies = self.get_axis_DRF(self.Y_tlds)
        zTallies = self.get_axis_DRF(self.Z_tlds)
        xError = self.get_axis_error(self.X_tlds)
        yError = self.get_axis_error(self.Y_tlds)
        zError = self.get_axis_error(self.Z_tlds)
        
        allTallies = np.concatenate((xTallies,yTallies,zTallies),axis=0)
        allErrors = np.concatenate((xError,yError,zError),axis=0)
        
        with open(dataset + 'meanTallies.csv', 'w', newline='') as f:
            write = csv.writer(f)
            write.writerow(self.E_bins)
            write.writerows(allTallies)
            
        with open(dataset + 'errorTallies.csv', 'w', newline='') as f:
            write = csv.writer(f)
            write.writerow(self.E_bins)
            write.writerows(allErrors)
        
data = allData()