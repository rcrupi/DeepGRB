import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize, LogNorm
sns.set_theme()
import numpy as np
import pandas as pd
from gbm.data import TTE
from gbm.binning.unbinned import bin_by_time
from gbm.plot import Lightcurve
from gbm.finder import BurstCatalog
import pickle

tte_path = '/beegfs/rcrupi/zzz_other/tte_pkl/' # tte
list_tte = os.listdir(tte_path)
bool_pkl = False
if bool_pkl:
    burstcat = BurstCatalog()
    df_burst = pd.DataFrame(burstcat.get_table())

    def det_triggered(str_mask):
        list_det = np.array(['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7',
                             'n8', 'n9', 'na', 'nb'])
        try:
            idx_det = np.where(np.array([int(i) for i in list(str_mask)]) == 1)
            return list(list_det[idx_det])
        except:
            print("Error, not found detectors triggered.")
            return list(list_det)

    ds_train = []
    max_len_time = 8000
    for tte_tmp in list_tte:
        # Check if detector has event signal counts
        str_det = df_burst.loc[df_burst['trigger_name'] == tte_tmp.split('_')[3], 'bcat_detector_mask'].values[0]
        list_det = det_triggered(str_mask=str_det)
        if tte_tmp.split('_')[2] not in list_det:
            # print(tte_tmp + ' not used.')
            continue
        # read a tte file
        tte = TTE.open(tte_path+tte_tmp)
        print(tte)
        # bin in time
        phaii = tte.to_phaii(bin_by_time, 0.256, time_ref=0.0)
        type(phaii)
        # # plot the lightcurve
        # lcplot = Lightcurve(data=phaii.to_lightcurve())
        # plt.show()
        pdc = phaii.data.counts
        if pdc.shape[0] < max_len_time:
            diff_len = max_len_time - pdc.shape[0]
            pdc = np.pad(pdc, [(0, diff_len), (0, 0)], mode='constant', constant_values=0)
        else:
            print("An event is cut for the sake of dimensions. dim original: " + str(pdc.shape[0]))
            pdc = pdc[0:max_len_time, :]
        ds_train.append(pdc)
        # # Draw a heatmap with the numeric values in each cell
        # index = (phaii.data.time_centroids>=-15) & (phaii.data.time_centroids<15)
        # pd_ctime = pd.DataFrame(phaii.data.counts[index]).T
        # pd_ctime.index = np.around(phaii.data.energy_centroids)
        # pd_ctime.columns = np.around(phaii.data.time_centroids[index], 2)
        # f, ax = plt.subplots(figsize=(12, 8))
        # sns.heatmap(pd_ctime.loc[:, :], annot=False, fmt="d", linewidths=.5, ax=ax, norm=LogNorm()) # Normalize, LogNorm

    # mat_len = []
    # for i in ds_train:
    #     mat_len.append(i.shape)
    # mat_len = np.array(mat_len)
    # # max=8000, q75%=4000
    # plt.boxplot(mat_len[:, 0])

    ds_train = np.array(ds_train)

    for i in range(0, ds_train.shape[0]):
        with open(tte_path + 'ds_train' + str(i) + '.pickle', 'wb') as f:
            pickle.dump(ds_train[i, :, :], f)
    # with open(tte_path+'ds_train.pickle', 'wb') as f:
    #     pickle.dump(ds_train, f)
else:
    ds_train = []
    for i in os.listdir(tte_path):
        with open(tte_path+i, 'rb') as f:
            ds_train.append(pickle.load(f))
    ds_train = np.array(ds_train)

pass




