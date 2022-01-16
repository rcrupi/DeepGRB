# import utils
import os
import shutil
# GBM data tools
from gbm.finder import ContinuousFtp
from gbm.finder import TriggerFtp
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

from gbm.finder import BurstCatalog
burstcat = BurstCatalog()
df_burst = pd.DataFrame(burstcat.get_table())

# Overwrite csv dataframe
bool_overwrite = False
data_type = 'tte' # 'ctime'

# Run 4 times the download to be sure that a day is downloaded
for round in [0]:
    # List csv files
    list_csv = [i for i in os.listdir(data_path + data_type)]
    # Cycle for each Burst day
    for row in df_burst['trigger_name']:
        try:
            # Check if dataset csv is already computed
            if np.sum([row in i for i in list_csv]) < 14 and not bool_overwrite:
                # Define what day download
                print('Initialise connection FTP for trigger: ', row)
                trig_find = TriggerFtp(row[2:])
                if data_type == 'ctime':
                    trig_find.get_ctime(data_path + 'ctime')
                elif data_type == 'tte':
                    trig_find.get_tte(data_path + 'tte')
        except Exception as e:
            print(e)
            print('Error for file: ' + row[2:])
