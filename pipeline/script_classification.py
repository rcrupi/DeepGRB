import numpy as np
import pandas as pd
from connections.utils.config import FOLD_RES

pd.set_option("display.width", 1000)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)

df_catalog = pd.DataFrame()
for (start_month, end_month) in [
    ("03-2019", "07-2019"),
    ("01-2014", "03-2014"),
    ("11-2010", "02-2011"),
]:
    df_catalog = df_catalog.append(
        pd.read_csv(FOLD_RES + 'frg_' + start_month + '_' + end_month + '/events_table_loc.csv')
    )

df_catalog = df_catalog.reset_index(drop=True).dropna(axis=0, subset=['catalog_triggers'])

# Target variable
# GRB: Gamma-Ray burst
# SFL: Solar flare
# LOC: Local particles
# TGF: Terrestrial Gamma-Ray Flash
# TRA: Generic transient
# SGR: Soft gamma repeater
# GAL: Galactic binary
# DIS: Distance particle event
# UNC: Uncertain classification
df_catalog = df_catalog[~df_catalog['catalog_triggers'].str.slice(0, 3).isin(['TRA', 'TGF'])].copy()

y = df_catalog['catalog_triggers'].str.slice(0, 3)
y = y.replace({'GRB': 0, 'SFL': 1, 'LOC': 2})
print(y.value_counts())

X = df_catalog[['trig_dets', 'sigma_r0', 'sigma_r1', 'sigma_r2', 'duration',
                # 'qtl_cut_r0', 'qtl_cut_r1', 'qtl_cut_r2',
                'ra', 'dec',
                # 'ra_montecarlo', 'dec_montecarlo', 'ra_std', 'dec_std',
                'ra_earth', 'dec_earth',
                'earth_vis',
                # 'sun_vis',
                'ra_sun', 'dec_sun',
                # 'l_galactic',
                'b_galactic',
                # 'lat_fermi', 'lon_fermi',
                # 'alt_fermi'
                ]].copy()

print(df_catalog[df_catalog['ra'].isna()])
X = X.fillna(0)

# Feature engineering
X['num_det_rng'] = None
X['num_r0'] = None
X['num_r1'] = None
X['num_r2'] = None

X['num_det_rng'] = X['trig_dets'].apply(lambda x: len(x.split(' ')) + 1)
for det_tmp in ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb']:
    X['num_' + det_tmp] = None
    X['num_' + det_tmp] = X['trig_dets'].apply(lambda x: 1 if det_tmp in x else 0)
X['num_r0'] = X['trig_dets'].apply(lambda x: 1 if 'r0' in x else 0)
X['num_r1'] = X['trig_dets'].apply(lambda x: 1 if 'r1' in x else 0)
X['num_r2'] = X['trig_dets'].apply(lambda x: 1 if 'r2' in x else 0)
del X['trig_dets']

X['diff_sun'] = np.minimum(abs(X['ra'] - X['ra_sun']), 360 - abs(X['ra'] - X['ra_sun'])) +\
                np.minimum(abs(X['dec'] - X['dec_sun']), 180 - abs(X['dec'] - X['dec_sun']))
X['diff_earth'] = np.minimum(abs(X['ra'] - X['ra_earth']), 360 - abs(X['ra'] - X['ra_earth'])) +\
                np.minimum(abs(X['dec'] - X['dec_earth']), 180 - abs(X['dec'] - X['dec_earth']))
del X['ra_sun'], X['dec_sun'], X['ra_earth'], X['dec_earth'], X['ra'], X['dec']

# Train a DT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0, max_depth=3, class_weight='balanced')
clf.fit(X_train, y_train)
y_pred_tot = clf.predict(X)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y, y_pred_tot))
print(confusion_matrix(y_train, y_pred_train))
print(confusion_matrix(y_test, y_pred_test))

from sklearn import tree
import matplotlib.pyplot as plt
plt.figure(figsize=(16, 12))
tree.plot_tree(clf, filled=True, feature_names=X_train.columns)
plt.show()

# # # Manual Decision Tree
print("Statistics for distance from Sun")
print(X.loc[y == 0, 'diff_sun'].mean(), X.loc[y == 0, 'diff_sun'].std(), X.loc[y == 0, 'diff_sun'].max())
print(X.loc[y == 1, 'diff_sun'].mean(), X.loc[y == 1, 'diff_sun'].std(), X.loc[y == 1, 'diff_sun'].max())
print(X.loc[y == 2, 'diff_sun'].mean(), X.loc[y == 2, 'diff_sun'].std(), X.loc[y == 2, 'diff_sun'].max())
# Look event 89 of 2014. It is a TGF
print(X.loc[y == 1].iloc[np.where(X.loc[y == 1, 'sigma_r1'] > X.loc[y == 1, 'sigma_r0'])])
print(df_catalog.loc[(y == 1) & (df_catalog['sigma_r0'] == 0)])
print("Statistics for distance from Galactic plane")
print(X.loc[y == 0, 'b_galactic'].mean(), X.loc[y == 0, 'b_galactic'].std(), X.loc[y == 0, 'b_galactic'].max())
print(X.loc[y == 1, 'b_galactic'].mean(), X.loc[y == 1, 'b_galactic'].std(), X.loc[y == 1, 'b_galactic'].max())
print(X.loc[y == 2, 'b_galactic'].mean(), X.loc[y == 2, 'b_galactic'].std(), X.loc[y == 2, 'b_galactic'].max())
print("Statistics for distance from Earth")
print(X.loc[y == 0, 'diff_earth'].mean(), X.loc[y == 0, 'diff_earth'].std(), X.loc[y == 0, 'diff_earth'].max())
print(X.loc[y == 1, 'diff_earth'].mean(), X.loc[y == 1, 'diff_earth'].std(), X.loc[y == 1, 'diff_earth'].max())
print(X.loc[y == 2, 'diff_earth'].mean(), X.loc[y == 2, 'diff_earth'].std(), X.loc[y == 2, 'diff_earth'].max())
print(df_catalog.loc[df_catalog.earth_vis == False])
print(df_catalog.loc[X['diff_earth'] < 80])
print("Statistics for num. det triggered")
print(X.loc[y == 0, 'num_det_rng'].mean(), X.loc[y == 0, 'num_det_rng'].std(), X.loc[y == 0, 'num_det_rng'].max())
print(X.loc[y == 1, 'num_det_rng'].mean(), X.loc[y == 1, 'num_det_rng'].std(), X.loc[y == 1, 'num_det_rng'].max())
print(X.loc[y == 2, 'num_det_rng'].mean(), X.loc[y == 2, 'num_det_rng'].std(), X.loc[y == 2, 'num_det_rng'].max())
print("Statistics for lat and lon")
print(df_catalog.loc[y == 2, ['lat_fermi', 'lon_fermi']])

y_pred_sf = (X['diff_sun'] < 50) & (X['sigma_r0'] > 0)
y_pred_tgf = (1-df_catalog['earth_vis']) | (X['diff_earth'] < 80)
y_pred_gf = (abs(df_catalog['b_galactic']) < 10) & (df_catalog['earth_vis'])
y_pred_lp = ((((20 < df_catalog['lat_fermi']) & (df_catalog['lat_fermi'] < 30)) &
              ((240 < df_catalog['lon_fermi']) & (df_catalog['lon_fermi'] < 280))) |\
            (((-10 < df_catalog['lat_fermi']) & (df_catalog['lat_fermi'] < 5)) &
             ((240 < df_catalog['lon_fermi']) & (df_catalog['lon_fermi'] < 300)))) &\
            (X['num_det_rng'] >= 18)
y_pred_grb = (df_catalog['earth_vis']) & (X['diff_sun'] >= 50) & (X['sigma_r1'] > 0) & \
             (abs(df_catalog['b_galactic']) >= 10)
y_pred_unc = 1 - (y_pred_sf | y_pred_tgf | y_pred_gf | y_pred_lp | y_pred_grb)

print(
    y_pred_sf.sum(),
    y_pred_tgf.sum(),
    y_pred_gf.sum(),
    y_pred_lp.sum(),
    y_pred_grb.sum(),
    y_pred_unc.sum()
      )

from gbm.data import PosHist
from gbm.plot import EarthPlot
from gbm.finder import ContinuousFtp
import os
earthplot = EarthPlot()
cont_finder = ContinuousFtp(met=df_catalog.loc[132, 'start_met'])
poshist_name = cont_finder.ls_poshist()[0]
cont_finder.get_poshist('./tmp_pos')
# open a poshist file
poshist = PosHist.open('./tmp_pos/' + poshist_name)
os.remove('./tmp_pos/' + poshist_name)
earthplot.add_poshist(poshist, trigtime=df_catalog.loc[132, 'met_localisation'],
                      time_range=(df_catalog.loc[132, 'start_met'], df_catalog.loc[132, 'end_met']))

pass
