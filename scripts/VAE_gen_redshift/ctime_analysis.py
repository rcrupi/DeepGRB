# import utils
import os
import shutil
# GBM data tools
from gbm.data import Ctime, Cspec
from gbm.binning.binned import rebin_by_time
# Standard packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize, LogNorm
sns.set_theme()
from connections.utils.config import PATH_TO_SAVE
data_path = PATH_TO_SAVE + "ctime/"

list_csv = [i for i in os.listdir(data_path) if '.pha' in i]

for file_tmp in list_csv:
    c_tmp = Ctime.open(data_path + '/' + file_tmp)
    break



# Plot lightcurve
rebinned_cspec = c_tmp.rebin_time(rebin_by_time, 0.256)
# integrate over the four range keV
lightcurve = rebinned_cspec.to_lightcurve(energy_range=(30, 500))
index = (lightcurve.centroids>-20) & (lightcurve.centroids<20)
plt.plot(lightcurve.centroids[index], lightcurve.rates[index], '-x')

# Define pandas of ctime
index = (c_tmp.data.time_centroids>=0) & (c_tmp.data.time_centroids<15)
plt.plot(c_tmp.data.time_centroids, c_tmp.data.counts[:,0], '-x')
pd_ctime = pd.DataFrame(c_tmp.data.counts[index]).T
pd_ctime.index = np.around(c_tmp.data.energy_centroids)
pd_ctime.columns = np.around(c_tmp.data.time_centroids[index], 2)

# list_spec_sum = 0*list_spec[0].data.counts
# for i in range(len(list_spec)):
#   list_spec_sum += list_spec[i].data.counts
# index_spec = (list_spec[0].data.tstart >= 577492200) & (list_spec[0].data.tstop <= 577492600)
# print(list_spec[0].data.energy_centroids[0:50])
# cspec_counts = pd.DataFrame(list_spec_sum[index_spec]).T.iloc[list(range(0,50))]
# cspec_counts.index = np.around(list_spec[0].data.energy_centroids[cspec_counts.index])

# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(18, 12))
sns.heatmap(pd_ctime.loc[:, :], annot=False, fmt="d", linewidths=.5, ax=ax, norm=Normalize()) # Normalize, LogNorm