import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Preprocess data downloaded from FTP
from gbm.data import TTE
from gbm.binning.unbinned import bin_by_time
from gbm.finder import ContinuousFtp
from gbm.time import Met
from gbm.background import BackgroundFitter
from gbm.background.binned import Polynomial
from gbm.plot import Lightcurve

tte_path = '/beegfs/rcrupi/zzz_other/per_giovanni/'

# 2010, 2014, 2019
df_catalog = pd.read_csv(tte_path + 'events_table_loc_2014' + '.csv')

# For loop into the events catalog
for idx, row in df_catalog.iterrows():
    print(idx, row)
    # Initialise the Met time of the event
    starttime_met = Met(0).from_iso(row['start_times_offset'].replace(' ', 'T')).met
    cont_finder = ContinuousFtp(met=starttime_met)
    # Download the TTE event per each triggered detectors
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
    # Initialise residual array
    # bin in time (ex. 0.256s)
    lst_std = []
    for flt_bin_time in [0.01, 0.02, 0.05, 0.1, 0.256, 0.5]:
        res = 0
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
                # filter erange
                erange = (8.0, 900.0)
                # filter time
                time_before_after = max(row['duration']/2, 16)
                # Load the TTE integrating in energy
                phaii = dct_tte[det_tmp].to_phaii(bin_by_time, flt_bin_time,
                                                  time_range=(starttime_met - time_before_after,
                                                              row['end_met'] + time_before_after),
                                                  time_ref=starttime_met)
                # Lightcurve
                lc_data = phaii.to_lightcurve(energy_range=erange)
                # lcplot = Lightcurve(data=phaii.to_lightcurve(energy_range=erange))
                # plt.show()
                # Background fit
                bkgd_times = [(starttime_met - time_before_after, starttime_met),
                              (row['end_met'], row['end_met'] + time_before_after)]
                backfitter = BackgroundFitter.from_phaii(phaii, Polynomial, time_ranges=bkgd_times)
                backfitter.fit(order=1)
                # Apply fit
                bkgd = backfitter.interpolate_bins(phaii.data.tstart, phaii.data.tstop)
                lc_bkgd = bkgd.integrate_energy(*erange)
                # lcplot = Lightcurve(data=lc_data, background=lc_bkgd)
                # Compute residual
                res += lc_data.rates - lc_bkgd.rates
            except:
                print('Error in file of detector', det_tmp)
        # TODO apply microvariability
        lst_std.append(np.std(res))
    plt.plot(lst_std, 'x-')

pass
