# import modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.localization import localization

# Fermi SkyPlot
from gbm.data import HealPix
from gbm.data import PosHist
from gbm.plot import SkyPlot
from gbm.finder import ContinuousFtp

# Dataset xxx_101225.csv xxx_2014.csv xxx_19_01-06.csv
df_frg = pd.read_csv('/beegfs/rcrupi/pred/'+'frg_19_01-06.csv')
df_bkg = pd.read_csv('/beegfs/rcrupi/pred/'+'bkg_19_01-06.csv')
# 101111, 140102, 140112, 190404, 190420
df_event = pd.read_csv('/beegfs/rcrupi/bkg/'+'190420.csv')
# 311194700, 410351700, 411228000, 576076200, 577492400
met_event = 577492400

df_bkg['met'] = df_frg['met'].values
df_frg_bkg = pd.merge(df_event, df_bkg, how='left', on=['met'], suffixes=('_frg', '_bkg'))

col_ra = np.sort([i for i in df_frg_bkg.columns if '_ra' in i and len(i) == 5 and 'n' in i])
col_dec = np.sort([i for i in df_frg_bkg.columns if '_dec' in i and len(i) == 6 and 'n' in i])
# select energy range to 1
col_count_frg = np.sort([i for i in df_frg_bkg.columns if '_frg' in i and 'n' in i and '_r1_' in i])
col_count_bkg = np.sort([i for i in df_frg_bkg.columns if '_bkg' in i and 'n' in i and '_r1_' in i])
# Define columns for residual counts
col_det = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb']
# Calculate the residual
df_frg_bkg[col_det] = df_frg_bkg[col_count_frg].values - df_frg_bkg[col_count_bkg].values

df_frg_bkg_event = df_frg_bkg.loc[(df_frg_bkg['met'] > met_event - 1000) & (df_frg_bkg['met'] < met_event + 1000), col_det]
df_frg_bkg_event['n6'].plot()
max_fin = 0
for i in col_det:
  max_tmp = df_frg_bkg_event.loc[:, i].max()
  if max_tmp>max_fin:
    max_fin=max_tmp
    ind_max=df_frg_bkg_event.loc[:, i].idxmax()
print('The index of the peak is: ', ind_max)

col_filter = range(0, 12)
list_ra = df_frg_bkg.loc[ind_max, np.array(col_ra)[col_filter]].values/180*np.pi
list_dec = df_frg_bkg.loc[ind_max, np.array(col_dec)[col_filter]].values/180*np.pi
counts = np.maximum(df_frg_bkg.loc[ind_max, np.array(col_det)[col_filter]].values, 0)
counts_frg = df_frg_bkg.loc[ind_max, np.array(col_count_frg)[col_filter]].values
counts_bkg = df_frg_bkg.loc[ind_max, np.array(col_count_bkg)[col_filter]].values

loc = localization(list_ra, list_dec, counts_frg, counts_bkg)
res = loc.fit()
print(res)
rnd_res = loc.fit_conf_int(500)
mean, cov = loc.plot()

# initialize the continuous data finder with a time (Fermi MET, UTC, or GPS)
cont_finder = ContinuousFtp(met=met_event)
cont_finder.get_poshist('tmp')
# open a poshist file
poshist = PosHist.open("tmp/" + os.listdir("../tmp")[0])
os.remove("tmp/" + os.listdir("../tmp")[0])
# initialize plot
skyplot = SkyPlot()
# plot the orientation of the detectors and Earth blockage at our time of interest
skyplot.add_poshist(poshist, trigtime=met_event)
gauss_map = HealPix.from_gaussian(np.round(res['ra']), np.round(res['dec']), 10)
skyplot.add_healpix(gauss_map)
plt.show()

pass
