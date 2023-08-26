# standard utils
import numpy as np
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
# ML utils
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import tree
from imodels import SkopeRulesClassifier # , RuleFitClassifier,  HSTreeClassifierCV
# local utils
from connections.utils.config import FOLD_RES

pd.set_option("display.width", 1000)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)

# Load catalog from paper
df_class = pd.read_csv(Path(os.getcwd()).parent / "data/DeepGRB_catalog.csv", index_col=0)
# Load raw catalogs
df_catalog = pd.DataFrame()
for (start_month, end_month) in [
    ("03-2019", "07-2019"),
    ("01-2014", "03-2014"),
    ("11-2010", "02-2011"),
]:
    df_catalog = df_catalog.append(
        pd.read_csv(FOLD_RES + 'frg_' + start_month + '_' + end_month + '/events_table_loc.csv')
    )
# drop index and define datetime
df_catalog = df_catalog.reset_index(drop=True)
df_catalog['datetime'] = df_catalog['start_times'].str.slice(0, 19)
# merge the two catalogs
df_catalog = pd.merge(df_catalog, df_class[['datetime', 'catalog_triggers']], how="left", on=["datetime"])
df_catalog = df_catalog.dropna(subset=['catalog_triggers_y'])
df_catalog['catalog_triggers'] = df_catalog['catalog_triggers_y']
del df_catalog['catalog_triggers_x'], df_catalog['catalog_triggers_y']
# define the catalog for the unknown events
idx_unknown = df_catalog['catalog_triggers'].str.contains("UNKNOWN")
df_catalog_unk = df_catalog[df_catalog['catalog_triggers'].str.contains("UNKNOWN")].copy()
print(df_catalog_unk['catalog_triggers'].unique())
del df_catalog_unk


# # # Target variable in the paper
# GRB: Gamma-Ray burst
# SF: Solar flare
# UNC(LP): Local particles
# TGF: Terrestrial Gamma-Ray Flash
# GF: Galactic flare
# UNC: Uncertain classification
# # # Statistics of know events
# Target variable GBM. GRB: Gamma-Ray burst, SFL: Solar flare, LOC: Local particles, TGF: Terrestrial Gamma-Ray Flash,
# TRA: Generic transient, SGR: Soft gamma repeater, GAL: Galactic binary, DIS: Distance particle event,
# UNC: Uncertain classification.

# # # Analysis of events
ev_type_list = ['GRB', 'SF', 'UNC(LP)', 'TGF', 'GF', 'UNC']
for ev_type in ev_type_list:
    df_catalog[ev_type] = False
    df_catalog.loc[idx_unknown, ev_type] = df_catalog.loc[idx_unknown, 'catalog_triggers'].apply(
        lambda x: ev_type in x.replace("UNKNOWN: ", "").split("/")).values
    dct_ev = {"GRB": "GRB", "SFL": "SF", "LOC": "UNC(LP)", "TRA": "UNC", "TGF": "TGF", "UNC": "UNC",
              "SGR": "UNC", "GAL": "GF", "DIS": "UNC"}
    df_catalog.loc[~idx_unknown, ev_type] = df_catalog.loc[~idx_unknown, 'catalog_triggers'].apply(
        lambda x: ev_type == dct_ev[x[0:3]]).values

print(df_catalog[ev_type_list].sum())

# Prepare the dataset
def prepare_X(df_catalog):
    X = df_catalog[['trig_dets', 'sigma_r0', 'sigma_r1', 'sigma_r2', 'duration',
                    'qtl_cut_r0', 'qtl_cut_r1', 'qtl_cut_r2',
                    'ra', 'dec',
                    'ra_montecarlo', 'dec_montecarlo', 'ra_std', 'dec_std',
                    'ra_earth', 'dec_earth',
                    'earth_vis',
                    'sun_vis',
                    'ra_sun', 'dec_sun',
                    'l_galactic',
                    'b_galactic',
                    'lat_fermi', 'lon_fermi',
                    'alt_fermi',
                    'l'
                    ]].copy()

    print(df_catalog[df_catalog['ra'].isna()])
    X = X.fillna(0)

    # Feature engineering
    X['num_det_rng'] = None
    X['num_r0'] = None
    X['num_r1'] = None
    X['num_r2'] = None

    X['num_det_rng'] = X['trig_dets'].apply(lambda x: len(x.split(' ')) + 1)
    X['num_det'] = 0
    for det_tmp in ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb']:
        X['num_' + det_tmp] = None
        X['num_' + det_tmp] = X['trig_dets'].apply(lambda x: 1 if det_tmp in x else 0)
        X['num_det'] += X['num_' + det_tmp]

    X['num_r0'] = X['trig_dets'].apply(lambda x: 1 if 'r0' in x else 0)
    X['num_r1'] = X['trig_dets'].apply(lambda x: 1 if 'r1' in x else 0)
    X['num_r2'] = X['trig_dets'].apply(lambda x: 1 if 'r2' in x else 0)
    del X['trig_dets']

    X['mean_det_sol_face'] = X[['num_' + i for i in ['n0', 'n1', 'n2', 'n3', 'n4', 'n5']]].mean(axis=1)
    X['mean_det_not_sol_face'] = X[['num_' + i for i in ['n6', 'n7', 'n8', 'n9', 'na', 'nb']]].mean(axis=1)

    X.loc[df_catalog.loc[:, 'UNC(LP)'] == 1, ['num_' + i for i in ['n6', 'n7', 'n8', 'n9', 'na', 'nb']]].mean(
        axis=1)

    X['diff_sun'] = np.minimum(abs(X['ra'] - X['ra_sun']), 360 - abs(X['ra'] - X['ra_sun'])) +\
                    np.minimum(abs(X['dec'] - X['dec_sun']), 180 - abs(X['dec'] - X['dec_sun']))
    X['diff_earth'] = np.minimum(abs(X['ra'] - X['ra_earth']), 360 - abs(X['ra'] - X['ra_earth'])) +\
                    np.minimum(abs(X['dec'] - X['dec_earth']), 180 - abs(X['dec'] - X['dec_earth']))
    del X['ra_sun'], X['dec_sun'], X['ra_earth'], X['dec_earth'], X['ra'], X['dec']

    X['HR10'] = np.minimum(X['sigma_r1'] / X['sigma_r0'], 10)
    X['HR21'] = np.minimum(X['sigma_r2'] / X['sigma_r1'], 10)

    X['sigma_r0_ratio'] = X['sigma_r0'] / X['duration']
    X['sigma_r1_ratio'] = X['sigma_r1'] / X['duration']
    X['sigma_r2_ratio'] = X['sigma_r2'] / X['duration']

    X['num_anti_coincidence'] = (X['num_n1'] & X['num_n8']) + (X['num_n2'] & X['num_n7']) + \
                                 (X['num_n3'] & X['num_nb']) + (X['num_n4'] & X['num_na']) + \
                                (X['num_n5'] & X['num_nb'])

    X['lon_fermi_shift'] = X['lon_fermi'].apply(lambda x: x if x <= 180 else -(360-x))
    lst_point_saa = [(30, -30), (15, -22.5), (0, -15), (-15, -7.5), (-30, 0), (-45, 3), (-60, 0), (-80, -3),
                     (-90, -7.5), (-95, -15),
                     # (-90, -22.5), (-85, -30)
                     (-115, -22.5), (-135, -30),
                     # (-100, 30), (100, - 30)
                     ]
    # lst_point_saa_lon = [i[0] for i in lst_point_saa]
    # lst_point_saa_lat = [i[1] for i in lst_point_saa]
    # X['dist_saa'] = X[['lon_fermi_shift', 'lat_fermi']].apply(
    #      lambda x: min([abs(x[0]-i[0])+abs(x[1]-i[1]) for i in lst_point_saa]), axis=1)
    X['dist_saa_lon'] = X[['lon_fermi_shift', 'lat_fermi']].apply(
        lambda x: min([abs(x[0] - i[0]) for i in lst_point_saa]), axis=1)
    X['dist_saa_lat'] = X[['lon_fermi_shift', 'lat_fermi']].apply(
        lambda x: min([abs(x[1] - i[1]) for i in lst_point_saa]), axis=1)

    X['dist_polo_nord_lon'] = X[['lon_fermi_shift', 'lat_fermi']].apply(
        lambda x: abs(x[0] - (-100)), axis=1)
    X['dist_polo_nord_lat'] = X[['lon_fermi_shift', 'lat_fermi']].apply(
        lambda x: abs(x[1] - 30), axis=1)
    X['dist_polo_sud_lon'] = X[['lon_fermi_shift', 'lat_fermi']].apply(
        lambda x: abs(x[0] - 100), axis=1)
    X['dist_polo_sud_lat'] = X[['lon_fermi_shift', 'lat_fermi']].apply(
        lambda x: abs(x[1] - (-30)), axis=1)

    # plt.figure()
    # plt.scatter(X['lon_fermi_shift'], X['lat_fermi'], c=(X['dist_saa'] < 15))
    # plt.figure()
    # plt.scatter(X['lon_fermi_shift'], X['lat_fermi'], c=(df_catalog['UNC(LP)']))

    return X

X = prepare_X(df_catalog)

# Train a DT
for ev_type in ['GRB', 'SF', 'UNC(LP)']:  # ev_type_list 'GRB', 'SF', 'UNC(LP)'
    print("Type of event analysed: ", ev_type)
    y = df_catalog[ev_type]

    # lst_select_col = ['dist_saa', 'l', 'num_det_rng', 'mean_det_sol_face', 'diff_sun', 'mean_det_not_sol_face']
    lst_select_col = X.columns

    X_train, X_test, y_train, y_test = train_test_split(X[lst_select_col], y, test_size=0.15, random_state=42, stratify=y)
    def wrap_fit(clf, X, X_train, X_test, y, y_train, y_test):
        clf.fit(X_train, y_train)
        y_pred_tot = clf.predict(X)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        print(confusion_matrix(y, y_pred_tot))
        print(confusion_matrix(y_train, y_pred_train))
        print(confusion_matrix(y_test, y_pred_test))
        print(classification_report(y_test, y_pred_test))
        return clf

    # Decision Tree
    clf = DecisionTreeClassifier(random_state=0, max_depth=3, class_weight='balanced', criterion="gini",
                                 min_impurity_decrease=0.01, min_samples_leaf=2, splitter="best") #0.01, 3
    clf = wrap_fit(clf, X[lst_select_col], X_train, X_test, y, y_train, y_test)
    plt.figure(figsize=(16, 12))
    tree.plot_tree(clf, filled=True, feature_names=X_train.columns, class_names=[f'NON {ev_type}', f'{ev_type}'])
    plt.title(ev_type)
    plt.show()
    # imodels
    clf = SkopeRulesClassifier()
    # clf = HSTreeClassifierCV(DecisionTreeClassifier(), reg_param=1, shrinkage_scheme_='node_based')
    clf = wrap_fit(clf, X, X_train, X_test, y, y_train, y_test)
    print(clf.rules_[0:3])

# # Plot Fermi position of earth when local particles occur
# plt.figure()
# plt.scatter(X.loc[df_catalog['UNC(LP)'], 'lon_fermi'].apply(lambda x: x if x < 180 else x-360),
#             X.loc[df_catalog['UNC(LP)'], 'lat_fermi'], c=X.loc[df_catalog['UNC(LP)'], 'l'])
# plt.colorbar()
# plt.figure()
# plt.boxplot(X.loc[df_catalog['UNC(LP)'], ['num_det', 'num_det_rng']])

# Manual classification logic
def classification_logic(df_catalog):
    y_pred = {}
    X = prepare_X(df_catalog)
    # y_pred['SF'] = (X['diff_sun'] < 41) & (X['sigma_r0'] > 0) & (df_catalog['sun_vis']) & \
    #                (1 - ((X['mean_det_not_sol_face'] >= 0.5) & (X['l'] >= 1.5)))
    y_pred['SF'] = (X['HR10'] <= 0.392) & (X['diff_sun'] < 63.49)
    # (X['sigma_r0'] > 27)
    y_pred['TGF'] = (1-df_catalog['earth_vis']) | (X['diff_earth'] < 80)
    y_pred['GF'] = (abs(df_catalog['b_galactic']) < 10) & (df_catalog['earth_vis'])
    # y_pred['UNC(LP)'] = \
    #     (((((15 < df_catalog['lat_fermi']) & (df_catalog['lat_fermi'] < 30)) &
    #        ((220 < df_catalog['lon_fermi']) & (df_catalog['lon_fermi'] < 275))) |
    #       (((-30 < df_catalog['lat_fermi']) & (df_catalog['lat_fermi'] < 8)) &
    #        ((230 < df_catalog['lon_fermi']) & (df_catalog['lon_fermi'] < 360))) |
    #       ((df_catalog['lat_fermi'] < -10) & (df_catalog['lon_fermi'] > 0))) & \
    #      (1 - y_pred['SF']) & (X['sigma_r0'] + X['sigma_r1'] + X['sigma_r2'] >= 11) & (
    #                  (X['ra_std'] >= 100) | X['mean_det_not_sol_face'] >= X['mean_det_sol_face'])) & \
    #     (X['num_det'] >= 9)
    y_pred['UNC(LP)'] = ((((X['dist_saa_lon'] <= 15) & (X['dist_saa_lat'] <= 3.6)) |
                         ((X['dist_polo_nord_lon'] <= 19) & (X['dist_polo_nord_lat'] <= 7.6)) |
                         ((X['dist_polo_sud_lon'] <= 19) & (X['dist_polo_sud_lat'] <= 7.6)) |
                         ((X['l'] >= 1.55))
                         ) & (((X['num_det'] >= 9) | (X['sigma_r0'] >= 100))) &
                         ((X['diff_sun'] > 11) | (np.maximum(X['ra_std'], X['dec_std']) > 100)))

    # rule from Decision Tree
    #     (X['num_det'] <= 8.5)*(df_catalog['l']<=1.551)*(df_catalog['dec_std']>=1179.5)+\
    #     (X['num_det'] <= 8.5)*(df_catalog['l']>1.551)*(df_catalog['ra']>110.5)+\
    #     (X['num_det'] > 8.5)*(df_catalog['ra_std']<=100)*(df_catalog['alt_fermi']<=528117.969)+\
    #     (X['num_det'] > 8.5)*(df_catalog['ra_std']>100)*(df_catalog['l']>1.11)

    # y_pred['GRB'] = (df_catalog['earth_vis']) & (X['diff_sun'] >= 25) & (X['sigma_r1'] > 0) & \
    #                 (abs(df_catalog['b_galactic']) >= 0) & (X['num_det'] <= 6) & (X['sigma_r0'] <= 20)

    y_pred['GRB'] = ((X['HR10'] > 0.64433) &
                     ((X['sigma_r0'] <= 17.389) | (X['qtl_cut_r1'] > 0.325) | ((X['qtl_cut_r1'] <= 0.325) &
                                                                               (X['num_anti_coincidence'] <= 1)))
                     & (~y_pred['UNC(LP)']))

            #(X['diff_sun'] > 10.57662) & (X['HR10'] > 0.64433) & (X['HR21'] <= 0.37867) & (X['sigma_r0'] <= 50.74399) & (X['sigma_r2'] <= 12.71737))

    y_pred['UNC'] = 1 - (y_pred['SF'] | y_pred['TGF'] | y_pred['GF'] | y_pred['UNC(LP)'] | y_pred['GRB'])
    return y_pred

y_pred = classification_logic(df_catalog)

for ev_type in ['SF', 'UNC(LP)', 'GRB']: #,  'SF', 'UNC(LP)'  # 'TGF', 'GF',  'UNC'
    print("Event: ", ev_type)
    print(confusion_matrix(df_catalog[ev_type], y_pred[ev_type]))
    print(classification_report(df_catalog[ev_type], y_pred[ev_type]))
    # To remove
    print("True in catalog, False with new rule")
    lst_col_show = ['trig_ids', 'start_times', 'sigma_r0', 'sigma_r1', 'sigma_r2', 'duration', 'ra', 'dec', 'ra_earth',
                    'dec_earth', 'earth_vis', 'sun_vis', 'ra_sun', 'dec_sun', 'l_galactic', 'b_galactic',
                    'lat_fermi', 'lon_fermi', 'alt_fermi', 'l', 'catalog_triggers', 'GRB', 'SF', 'UNC(LP)', 'TGF', 'GF',
                    'UNC', 'trig_dets', 'ra_montecarlo', 'dec_montecarlo', 'ra_std', 'dec_std']
    lst_X_col_show = ['num_det_rng', 'num_r0', 'num_r1', 'num_r2', 'num_det', 'num_n0', 'num_n1', 'num_n2', 'num_n3',
                      'num_n4', 'num_n5', 'num_n6', 'num_n7', 'num_n8', 'num_n9', 'num_na', 'num_nb',
                      'mean_det_sol_face', 'mean_det_not_sol_face', 'diff_sun', 'diff_earth']

    print(df_catalog.loc[(df_catalog[ev_type] == 1) & (y_pred[ev_type] == 0), lst_col_show])
    print(X.loc[(df_catalog[ev_type] == 1) & (y_pred[ev_type] == 0), lst_X_col_show])
    # To add
    print("False in catalog, True with new rule")
    print(df_catalog.loc[(df_catalog[ev_type] == 0) & (y_pred[ev_type] == 1), lst_col_show])
    print(X.loc[(df_catalog[ev_type] == 0) & (y_pred[ev_type] == 1), lst_X_col_show])
    pass

pass

