import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
# Preprocess data downloaded from FTP
from gbm.data import TTE
from gbm.binning.unbinned import bin_by_time
from gbm.plot import Lightcurve
from gbm.finder import BurstCatalog
from gbm.finder import ContinuousFtp
from gbm.time import Met
from gbm.background import BackgroundFitter
from gbm.background.binned import Polynomial
from gbm.plot import Lightcurve

tte_path = '/beegfs/rcrupi/zzz_other/per_giovanni/'

df_catalog = pd.DataFrame()
for (start_month, end_month) in [
    ("03-2019", "07-2019"),
    ("01-2014", "03-2014"),
    ("11-2010", "02-2011"),
]:
    df_catalog = df_catalog.append(
        pd.read_csv(tte_path + 'events_table_loc_' + start_month[3:] + '.csv')
    )

df_catalog = df_catalog.reset_index(drop=True).dropna(axis=0, subset=['catalog_triggers'])

# For loop into the events catalog
for idx, row in df_catalog.iterrows():
    print(idx, row)
    # Initialise the Met time of the event
    starttime_met = Met(0).from_iso(row['start_times_offset'].replace(' ', 'T')).met
    cont_finder = ContinuousFtp(met=starttime_met)
    # Download the TTE refering per each triggered detectors
    lst_det_triggered = []
    for det_tmp in ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb']:
        list_tte = os.listdir(tte_path)
        file_to_load = [i for i in list_tte if 'glg_tte_' + det_tmp + '_' +
                        row['start_times'][2:13].replace(' ', '_').replace('-', '') in i]
        # If the TTE is already present pass to the next detector
        if len(file_to_load) == 1:
            continue
        # Check if detector is triggered in the catalog
        if det_tmp in row['trig_dets']:
            # Download the TTE file
            cont_finder.get_tte(tte_path, dets=[det_tmp], full_day=False)
            lst_det_triggered.append(det_tmp)

    # Load the TTE and a dictionary: {'n1': TTE_n1, 'n2': ...}
    dct_tte = {}
    for det_tmp in lst_det_triggered:
        try:
            # Search in the file downloded the file
            list_tte = os.listdir(tte_path)
            file_to_load = [i for i in list_tte if 'glg_tte_' + det_tmp + '_' +
                            row['start_times'][2:13].replace(' ', '_').replace('-', '') in i]
            if len(file_to_load) == 1:
                file_to_load = file_to_load[0]
            # If not present continue to the next detector
            elif len(file_to_load) == 0:
                print('det not found: ', det_tmp)
                continue
            # read a tte file
            dct_tte[det_tmp] = TTE.open(tte_path + file_to_load)
            print(dct_tte[det_tmp])
            # bin in time (0.256s)
            flt_bin_time = 0.256
            # Load the TTE integrating in energy
            phaii = dct_tte[det_tmp].to_phaii(bin_by_time, flt_bin_time, time_range=(starttime_met-200, row['end_met']+200),
                                              energy_range=(50, 300)) #, time_ref=starttime_met)

            lcplot = Lightcurve(data=phaii.to_lightcurve())
            plt.show()

            bkgd_times = [(starttime_met - 20.0, starttime_met - 5.0), (row['end_met'] + 5, row['end_met'] + 20)]
            backfitter = BackgroundFitter.from_phaii(phaii, Polynomial, time_ranges=bkgd_times)
            backfitter.fit(order=1)


            pdc = phaii.data.counts
            lightcurve = pdc.sum(axis=1)
            plt.plot(lightcurve)
            plt.show()

        except:
            print('Can t find file of detector', det_tmp)

pass
