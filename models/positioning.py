# import modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.loc.localization_class import localization
from connections.utils.config import PATH_TO_SAVE, FOLD_PRED, FOLD_BKG
from gbm.time import Met

# Fermi SkyPlot
from gbm.data import HealPix
from gbm.data import PosHist
from gbm.plot import SkyPlot
from gbm.finder import ContinuousFtp

start_month = "01-2019"
end_month = "06-2019"

# Dataset xxx_101225.csv xxx_2014.csv xxx_19_01-06.csv
df_frg = pd.read_csv(PATH_TO_SAVE + FOLD_PRED + "/" + 'frg_' + start_month + '_' + end_month + '.csv')
df_bkg = pd.read_csv(PATH_TO_SAVE + FOLD_PRED + "/" + 'bkg_' + start_month + '_' + end_month + '.csv')
# 101111, 140102, 140112, 190404, 190420
# TODO select interval of triggers in trigs_table.csv
df_event = pd.read_csv(PATH_TO_SAVE + FOLD_BKG + '/' + '190301.csv')
# 311194700, 410351700, 411228000, 576076200, 577492400
timestamp_event = '2019-03-01 13:04:31'
obj_met = Met(0)
met_event = obj_met.from_iso(timestamp_event.replace(" ", "T")).met

df_bkg['met'] = df_frg['met'].values
df_frg_bkg = pd.merge(df_event, df_bkg, how='left', on=['met'], suffixes=('_frg', '_bkg'))

col_ra = np.sort([i for i in df_frg_bkg.columns if '_ra' in i and len(i) == 5 and 'n' in i])
col_dec = np.sort([i for i in df_frg_bkg.columns if '_dec' in i and len(i) == 6 and 'n' in i])
# select energy range to 1
col_count_frg = np.sort([i for i in df_frg_bkg.columns if '_frg' in i and 'n' in i and '_r0_' in i])
col_count_bkg = np.sort([i for i in df_frg_bkg.columns if '_bkg' in i and 'n' in i and '_r0_' in i])
# Define columns for residual counts
col_det = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb']
# Calculate the residual
df_frg_bkg[col_det] = df_frg_bkg[col_count_frg].values - df_frg_bkg[col_count_bkg].values

df_frg_bkg_event = df_frg_bkg.loc[(df_frg_bkg['met'] > met_event - 500) & (df_frg_bkg['met'] < met_event +100), col_det]
df_frg_bkg_event['n0'].plot()
max_fin = 0
for i in col_det:
  max_tmp = df_frg_bkg_event.loc[:, i].max()
  if max_tmp>max_fin:
    max_fin=max_tmp
    ind_max=df_frg_bkg_event.loc[:, i].idxmax()
print('The index of the peak is: ', ind_max)

# Initialise inputs for localization
list_ra = pd.DataFrame()
list_dec = pd.DataFrame()
counts = pd.DataFrame()
counts_frg = pd.DataFrame()
counts_bkg = pd.DataFrame()

col_filter = range(0, 12)
# ind_max, index_final_filter
list_ra = list_ra.append(
  df_frg_bkg.loc[ind_max, np.array(col_ra)[col_filter]] / 180 * np.pi)  # TODO update index start/end
list_dec = list_dec.append(df_frg_bkg.loc[ind_max, np.array(col_dec)[col_filter]] / 180 * np.pi)
counts = counts.append(np.maximum(df_frg_bkg.loc[ind_max, np.array(col_det)[col_filter]], 0))
counts_frg = counts_frg.append(df_frg_bkg.loc[ind_max, np.array(col_count_frg)[col_filter]])
counts_bkg = counts_bkg.append(df_frg_bkg.loc[ind_max, np.array(col_count_bkg)[col_filter]])

loc = localization(list_ra.values, list_dec.values, counts_frg.values, counts_bkg.values)
res = loc.fit()
print(res)
rnd_res = loc.fit_conf_int(250)
mean, cov = loc.plot()

# initialize the continuous data finder with a time (Fermi MET, UTC, or GPS)
cont_finder = ContinuousFtp(met=met_event)
cont_finder.get_poshist('/home/rcrupi/PycharmProjects/fermi_ml/tmp')
# open a poshist file
poshist = PosHist.open("/home/rcrupi/PycharmProjects/fermi_ml/tmp/"
                       + os.listdir("/home/rcrupi/PycharmProjects/fermi_ml/tmp/")[0])
os.remove("/home/rcrupi/PycharmProjects/fermi_ml/tmp/"
                       + os.listdir("/home/rcrupi/PycharmProjects/fermi_ml/tmp/")[0])
# initialize plot
skyplot = SkyPlot()
# plot the orientation of the detectors and Earth blockage at our time of interest
skyplot.add_poshist(poshist, trigtime=met_event)
gauss_map = HealPix.from_gaussian(np.round(res['ra']), np.round(res['dec']), 10)
skyplot.add_healpix(gauss_map)
plt.show()
print('finish')
pass
