# import utils
import os
import shutil
# GBM data tools
from gbm.finder import ContinuousFtp
from gbm.data import Ctime, Cspec
from gbm.binning.binned import rebin_by_time
from gbm.data import PosHist
from gbm import coords
# Standard packages
import pandas as pd
import datetime, calendar
from connections.utils.config import PATH_TO_SAVE
import numpy as np
data_path = PATH_TO_SAVE
# Define range of energy
erange = {}
erange['n'] = [(28, 50), (50, 300), (300, 500)]
erange['b'] = [(756, 5025), (5025, 50000)]

grb_top = pd.DataFrame({'id': [], 'tStart': []})
list_year = list(2009 + np.array(range(0, 13)))
list_month = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
for year in list_year:
    for month in list_month:
        num_days = calendar.monthrange(year, month)[1]
        days = [datetime.date(year, month, day).strftime("%y%m%d") for day in range(1, num_days+1)]
        tStart = [datetime.datetime(year, month, day, 12).strftime('%Y-%m-%dT%H:%M:%S.00') for day in range(1, num_days+1)]
        # Dataframe with the list of days
        grb_top = grb_top.append(pd.DataFrame({'id': days, 'tStart': tStart}), ignore_index=True)
print('End list days bkg.')

# Overwrite csv dataframe
bool_overwrite = False

# Run 4 times the download to be sure that a day is downloaded
for round in [0]:
    # List csv files
    list_csv = [i for i in os.listdir(data_path+'cspec')]
    # Cycle for each Burst day
    for _, row in grb_top.iterrows():
        try:
            # Check if dataset csv is already computed
            if np.sum([row['id'][0:6] in i for i in list_csv]) < 15 and not bool_overwrite:
                # Define what day download
                # ftp_daily = ContinuousFtp(met=row['tStart'])
                print('Initialise connection FTP for time UTC: ', row['tStart'])
                ftp_daily = ContinuousFtp(utc=row['tStart'], gps=None)
                # Download in the folder chosen
                ftp_daily.get_cspec(data_path + 'cspec')
                # Download poshist
                ftp_daily.get_poshist(data_path + 'cspec')
                # # Sort list file to have poshist at the end
                # list_file = os.listdir(data_path + 'tmp')
                # # Initialise the data dictionary
                # dic_data = {}
                # for file_tmp in [i for i in list_file if '.pha' in i]:
                #     # Transform counts data
                #     dic_data = fun_lightcurve(dic_data, data_path, file_tmp)
                # # Add poshist variables
                # file_tmp = [i for i in list_file if 'poshist' in i][0]
                # dic_data = fun_poshist(dic_data, file_tmp)
                # # Create final dataset
                # df_data = pd.DataFrame(dic_data)
                # # df_data['y'] = 1*((dic_data['met']>=row['tStart'])&(dic_data['met']<=row['tStop']))
                # print('Saving file: ', row['id'][0:6])
                # df_data.to_csv(data_path + row['id'][0:6] + '.csv', index=False)
                # # Delete folder tmp of daily data
                # if os.path.exists(data_path + 'tmp'):
                #     shutil.rmtree(data_path + 'tmp')

        except Exception as e:
            print(e)
            print('Error for file: ' + row['id'][0:6])
