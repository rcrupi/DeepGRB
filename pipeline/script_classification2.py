# standard utils
import numpy as np
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
# ML utils
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import SequentialFeatureSelector
from imodels import SkopeRulesClassifier, RuleFitClassifier,  HSTreeClassifierCV
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from anchor import anchor_tabular
# local utils
from connections.utils.config import FOLD_RES

pd.set_option("display.width", 1000)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)

# # # Load catalog from paper
df_class = pd.read_csv(Path(os.getcwd()).parent / "data/DeepGRB_catalog.csv", index_col=0)
# # # Load raw catalogs
df_catalog = pd.DataFrame()
for (start_month, end_month) in [
    ("03-2019", "07-2019"),
    ("01-2014", "03-2014"),
    ("11-2010", "02-2011"),
]:
    df_catalog = df_catalog.append(
        pd.read_csv(FOLD_RES + 'frg_' + start_month + '_' + end_month + '/events_table_loc_wavelet_norm_ext_bkg2.csv')
    )
# drop index and define datetime
df_catalog = df_catalog.reset_index(drop=True)
df_catalog['datetime'] = df_catalog['start_times'].str.slice(0, 19)
# merge the two catalogs
df_catalog = pd.merge(df_catalog, df_class[['datetime', 'catalog_triggers']], how="left", on=["datetime"])
df_catalog['catalog_triggers_y'] = df_catalog['catalog_triggers_y'].fillna('UNKNOWN: FP')  # False Positive events
# df_catalog = df_catalog.dropna(subset=['catalog_triggers_y'])
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
ev_type_list = ['GRB', 'SF', 'UNC(LP)', 'TGF', 'GF', 'UNC', 'FP']
for ev_type in ev_type_list:
    df_catalog[ev_type] = False
    df_catalog.loc[idx_unknown, ev_type] = df_catalog.loc[idx_unknown, 'catalog_triggers'].apply(
        lambda x: ev_type in x.replace("UNKNOWN: ", "").split("/")).values
    dct_ev = {"GRB": "GRB", "SFL": "SF", "LOC": "UNC(LP)", "TRA": "UNC", "TGF": "TGF", "UNC": "UNC",
              "SGR": "UNC", "GAL": "GF", "DIS": "UNC"}
    df_catalog.loc[~idx_unknown, ev_type] = df_catalog.loc[~idx_unknown, 'catalog_triggers'].apply(
        lambda x: ev_type == dct_ev[x[0:3]]).values

print('Statistics total:')
print(df_catalog[ev_type_list].sum())
print(df_catalog[['GRB', 'SF', 'UNC(LP)', 'TGF', 'GF', 'UNC', 'FP']].mean())
print('Statistics GBM:')
print(df_catalog.loc[~df_catalog.catalog_triggers.str.contains('UNKNOWN'), ev_type_list].sum())
print(df_catalog.loc[~df_catalog.catalog_triggers.str.contains('UNKNOWN'),
['GRB', 'SF', 'UNC(LP)', 'TGF', 'GF', 'UNC', 'FP']].mean())
print('Statistics unknown:')
print(df_catalog.loc[df_catalog.catalog_triggers.str.contains('UNKNOWN'), ev_type_list].sum())
print(df_catalog.loc[df_catalog.catalog_triggers.str.contains('UNKNOWN'),
['GRB', 'SF', 'UNC(LP)', 'TGF', 'GF', 'UNC', 'FP']].mean())

# # # Prepare the dataset
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
                    'l'] + [i for i in df_catalog.columns if 'fe_' in i]].copy()

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
    # del X['ra_sun'], X['dec_sun'], X['ra_earth'], X['dec_earth'], X['ra'], X['dec']

    X['HR10'] = np.minimum(X['sigma_r1'] / X['sigma_r0'], 10)
    X['HR21'] = np.minimum(X['sigma_r2'] / X['sigma_r1'], 10)

    X['sigma_tot'] = X['sigma_r0'] + X['sigma_r1'] + X['sigma_r2']

    X['sigma_r0_ratio'] = X['sigma_r0'] / X['duration']
    X['sigma_r1_ratio'] = X['sigma_r1'] / X['duration']
    X['sigma_r2_ratio'] = X['sigma_r2'] / X['duration']
    X['sigma_tot_ratio'] = X['sigma_tot'] / X['duration']

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
    X['dist_saa'] = X[['lon_fermi_shift', 'lat_fermi']].apply(
         lambda x: min([abs(x[0]-i[0])+abs(x[1]-i[1]) for i in lst_point_saa]), axis=1)
    X['dist_saa_lon'] = X['lon_fermi_shift'].apply(
        lambda x: min([abs(x - i[0]) for i in lst_point_saa]))
    X['dist_saa_lat'] = X['lat_fermi'].apply(
        lambda x: min([abs(x - i[1]) for i in lst_point_saa]))

    X['dist_polo_nord_lon'] = X['lon_fermi_shift'].apply(
        lambda x: abs(x - (-100)))
    X['dist_polo_nord_lat'] = X['lat_fermi'].apply(
        lambda x: abs(x - 30))
    X['dist_polo_sud_lon'] = X['lon_fermi_shift'].apply(
        lambda x: abs(x - 100))
    X['dist_polo_sud_lat'] = X['lat_fermi'].apply(
        lambda x: abs(x - (-30)))

    # plt.figure()
    # plt.scatter(X['lon_fermi_shift'], X['lat_fermi'], c=(X['dist_saa'] < 15))
    # plt.figure()
    # plt.scatter(X['lon_fermi_shift'], X['lat_fermi'], c=(df_catalog['UNC(LP)']))

    for col_stat in ['fe_kur', 'fe_skw', 'fe_max', 'fe_min', 'fe_mea', 'fe_med', 'fe_std']:
        X[col_stat + '_ratio_med'] = X[col_stat] / X['fe_med']
        X[col_stat + '_ratio_std'] = X[col_stat] / X['fe_std']
        X[col_stat + '_ratio_bkg_int'] = X[col_stat] / (abs(X['fe_bkg_max']) - abs(X['fe_bkg_min']))
        X[col_stat + '_ratio_int'] = X[col_stat] / (abs(X['fe_max']) - abs(X['fe_min']))
        # X[col_stat + '_norm_med_bkg'] = X[col_stat] / X['fe_bkg_med']
        # X[col_stat + '_norm_all'] = X[col_stat] / np.maximum(abs(X['fe_bkg_max']), abs(X['fe_bkg_min']))


    X['fe_bkg_step'] = np.maximum(abs(X['fe_bkg_step_min']), abs(X['fe_bkg_step_max']))
    for col_stat in ['fe_bkg_np', 'fe_bkg_pp', 'fe_bkg_kur', 'fe_bkg_skw', 'fe_bkg_max', 'fe_bkg_min', 'fe_bkg_step',
                     'fe_bkg_med', 'fe_bkg_mea', 'fe_bkg_std', 'fe_bkg_step_max', 'fe_bkg_step_min', 'fe_bkg_step_med']:
        # X[col_stat + '_norm1'] = X[col_stat] / abs(X['fe_bkg_med'])
        # X[col_stat + '_norm2'] = X[col_stat] / abs(X['fe_max'])
        # X[col_stat + '_norm3'] = X[col_stat] / X['fe_std']
        X[col_stat + '_ratio_bkg_int'] = X[col_stat] / (abs(X['fe_bkg_max']) - abs(X['fe_bkg_min']))
        X[col_stat + '_ratio_bkg_max'] = X[col_stat] / np.maximum(abs(X['fe_bkg_max']), abs(X['fe_bkg_min']))
        X[col_stat + '_ratio_max_val'] = X[col_stat] / np.maximum(abs(X['fe_max']), abs(X['fe_min']))
        X[col_stat + '_ratio_std'] = X[col_stat] / X['fe_std']
        X[col_stat + '_ratio_bkg_std'] = X[col_stat] / X['fe_std']
        # X[col_stat + '_norm_all_lc'] = X[col_stat] / (abs(X['fe_max']) - abs(X['fe_min']))
        # X[col_stat + '_norm_std'] = X[col_stat] / X['fe_bkg_std']
        # X[col_stat + '_norm_std_lc'] = X[col_stat] / X['fe_std']

    return X

X = prepare_X(df_catalog)

# # # Train a DT
# lst_select_col = ['fe_wam1', 'fe_wen1', 'fe_wstd1', 'fe_wam2', 'fe_wen2', 'fe_wstd2', 'fe_wam3', 'fe_wen3',
#                 'fe_wstd3', 'fe_wam4', 'fe_wen4', 'fe_wstd4', 'fe_wam5', 'fe_wen5', 'fe_wstd5', 'fe_wam6',
#                 'fe_wen6', 'fe_wstd6', 'fe_wam7', 'fe_wen7', 'fe_wstd7', 'fe_wam8', 'fe_wen8', 'fe_wstd8',
#                 'fe_wam9', 'fe_wen9', 'fe_wstd9', 'fe_wet',
#                 'fe_np', 'fe_pp', 'fe_kur', 'fe_skw', 'fe_max', 'fe_min', 'fe_med', 'fe_mea', 'fe_std',
#                   'dist_polo_sud_lat', 'fe_kur_norm', 'fe_skw_norm', 'fe_max_norm', 'fe_min_norm', 'fe_mea_norm',
#                   'fe_std_norm'
#                   ]
# lst_select_col = [i for i in X.columns if 'fe_w' in i] + ['HR10', 'HR21']
lst_select_col = X.columns
# lst_select_col = ['HR10', 'fe_wet', 'dist_saa']

y_tmp = 3*df_catalog['SF'].copy()
y_tmp.loc[y_tmp != 3] = 2*df_catalog.loc[y_tmp != 3, 'UNC(LP)']
y_tmp.loc[(y_tmp != 3) & (y_tmp != 2)] = 1*df_catalog.loc[(y_tmp != 3) & (y_tmp != 2), 'GRB']
y_tmp.loc[(y_tmp != 3) & (y_tmp != 2) & (y_tmp != 1)] = 0
y_tmp.loc[df_catalog['FP'] == True] = 4


X = X.astype('float32')

if X[lst_select_col].isna().sum().sum() > 0 or np.isinf(X).sum().sum() > 0:
    print("Warning. NaN found in feature table. Filled with -1.")
    print(X[lst_select_col].isna().sum())
    X = X.fillna(-1)
    X = X.replace(np.inf, -1)
    X = X.replace(-np.inf, -1)

X_train, X_test, y_tmp_train, y_tmp_test = train_test_split(X[lst_select_col], y_tmp, test_size=0.2, random_state=42,
                                         stratify=y_tmp)

rf_mc = RandomForestClassifier(n_estimators=200, max_depth=4, class_weight='balanced', random_state=0,
                                 min_impurity_decrease=0.01)
rf_mc.fit(X_train[y_tmp_train!=0], y_tmp_train[y_tmp_train!=0])
y_pred_mc_test = rf_mc.predict(X_test)
y_pred_mc_train = rf_mc.predict(X_train)
# print(confusion_matrix(y_tmp_train, y_pred_mc_train, normalize=None))
# print(balanced_accuracy_score(y_tmp_train, y_pred_mc_train))
print(confusion_matrix(y_tmp_test[y_tmp_test!=0], y_pred_mc_test[y_tmp_test!=0], normalize=None))
print(balanced_accuracy_score(y_tmp_test[y_tmp_test!=0], y_pred_mc_test[y_tmp_test!=0]))

dct_pred_rf = {'FP': [], 'GRB': [], 'SF': [], 'UNC(LP)': []}
lst_select_col_old = lst_select_col
bln_feature_selection = True
for ev_type in ['FP', 'GRB', 'SF', 'UNC(LP)']:  # ev_type_list 'GRB', 'SF', 'UNC(LP)'
    lst_select_col = lst_select_col_old
    print("Type of event analysed: ", ev_type)
    y = df_catalog[ev_type].copy()
    y_train = y[X_train.index].copy()
    y_test = y[X_test.index].copy()

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

    # Random Forest with feature selection
    # clf = DecisionTreeClassifier(random_state=0, max_depth=None, class_weight='balanced', criterion="gini",
    #                              min_impurity_decrease=0.0, min_samples_leaf=2, splitter="best")
    # sfs = SequentialFeatureSelector(clf, n_features_to_select=20, n_jobs=-1, cv=2, direction="forward")
    # sfs.fit(X_train, y_train)
    # lst_select_col = list(X_train.columns[sfs.get_support()])
    if bln_feature_selection:
        qt = QuantileTransformer(n_quantiles=20, random_state=0)
        X_train_normed = pd.DataFrame(qt.fit_transform(X_train), columns=lst_select_col, index=X_train.index)
        lsvc = LinearSVC(C=20, penalty="l1", dual=False, class_weight='balanced', max_iter=100000, random_state=0).fit(
            X_train_normed, y_train)
        model_l1 = SelectFromModel(lsvc, prefit=True)
        lst_select_col = list(X_train.columns[model_l1.get_support()])
        print("Num. features selected: ", len(lst_select_col))

    # Random Forest feature importance
    clf = RandomForestClassifier(n_estimators=200, max_depth=4, class_weight='balanced', random_state=0,
                                 min_impurity_decrease=0.01)
    clf = wrap_fit(clf, X[lst_select_col], X_train[lst_select_col], X_test[lst_select_col], y, y_train, y_test)
    dct_pred_rf[ev_type] = clf.predict_proba(X[lst_select_col])[:, 1]
    print("Feature Importance Random Forest.")
    best_10_col = pd.Series(dict(zip(lst_select_col, clf.feature_importances_))).sort_values(ascending=False).head(10)
    print(best_10_col)

    # Anchor explainer
    if ev_type:
        explainer = anchor_tabular.AnchorTabularExplainer([0, 1], lst_select_col, X_train[lst_select_col].values)
        for idx in [ # 17, 337, 37, 72,
                    301, 302,  # False positive GRB for RF
                    293, 160, 236  # False negative GRB for RF
                    ]:  # 160, 337
            np.random.seed(1)
            print('Prediction: ', clf.predict(X.loc[idx, lst_select_col].values.reshape(1, -1))[0],
                  'Proba: ', clf.predict_proba(X.loc[idx, lst_select_col].values.reshape(1, -1))[:, 1][0],
                  ', idx: ', idx)
            # def wrap_predict_rf(X):
            #     X = pd.DataFrame(X, columns=lst_select_col)
            #     return clf.predict(X)
            # exp = explainer.explain_instance(X.loc[idx, lst_select_col].values.reshape(1, -1), wrap_predict_rf, threshold=0.95)
            # print('Anchor: %s' % (' AND '.join(exp.names())))
            # print('Precision: %.2f' % exp.precision())
            # print('Coverage: %.2f' % exp.coverage())

    # Decision Tree
    lst_select_col_dt = lst_select_col  # list(best_10_col.index), lst_select_col
    clf = DecisionTreeClassifier(random_state=0, max_depth=3, class_weight='balanced', criterion="gini",
                                 min_impurity_decrease=0.01, min_samples_leaf=2, splitter="best")
    clf = wrap_fit(clf, X[lst_select_col_dt], X_train[lst_select_col_dt], X_test[lst_select_col_dt], y, y_train, y_test)
    plt.figure(figsize=(16, 12))
    tree.plot_tree(clf, filled=True, feature_names=lst_select_col_dt, class_names=[f'NON {ev_type}', f'{ev_type}'])
    plt.title(ev_type)
    plt.show()
    # # imodels
    # clf = RuleFitClassifier(n_estimators=200, tree_size=4, max_rules=30, random_state=0) #SkopeRulesClassifier()
    # # clf = HSTreeClassifierCV(DecisionTreeClassifier(), reg_param=1, shrinkage_scheme_='node_based')
    # from sklearn.preprocessing import QuantileTransformer
    # qt = QuantileTransformer(n_quantiles=20, random_state=0)
    # X_train_normed = pd.DataFrame(qt.fit_transform(X_train[lst_select_col]), columns=lst_select_col, index=X_train.index)
    # X_test_normed = pd.DataFrame(qt.transform(X_test[lst_select_col]), columns=lst_select_col, index=X_test.index)
    # X_tot_norm = pd.DataFrame(qt.transform(X[lst_select_col]), columns=lst_select_col, index=X.index)
    # clf = wrap_fit(clf, X_tot_norm, X_train_normed, X_test_normed, y, y_train, y_test)
    # print(clf.rules_[0:4])
    del y, y_train, y_test
    print('-----------------------------------------------------------------------------------------------------------')

pred1vsall = pd.DataFrame(dct_pred_rf).idxmax(axis=1)
pred1vsall = pred1vsall.replace({'SF': 3, 'UNC(LP)': 2, 'GRB': 1, 'FP': 4})
# print(confusion_matrix(y_tmp_test, pred1vsall, normalize=None))
# print(balanced_accuracy_score(y_tmp_test, pred1vsall))
# print(confusion_matrix(y_tmp_test[y_tmp_test != 0], pred1vsall[y_tmp_test.values != 0], normalize=None))
# print(balanced_accuracy_score(y_tmp_test[y_tmp_test != 0], pred1vsall[y_tmp_test.values != 0]))
print(confusion_matrix(y_tmp_test[y_tmp_test != 0], pred1vsall[X_test.index][y_tmp_test.values != 0], normalize=None))
print(balanced_accuracy_score(y_tmp_test[y_tmp_test != 0], pred1vsall[X_test.index][y_tmp_test.values != 0]))
print(confusion_matrix(y_tmp[y_tmp != 0], pred1vsall[y_tmp.values != 0], normalize=None))
print(balanced_accuracy_score(y_tmp[y_tmp != 0], pred1vsall[y_tmp.values != 0]))

# # False positive
# aaa = y_tmp_test[y_tmp_test != 0]
# bbb = pred1vsall[X_test.index][y_tmp_test.values != 0]
# print(y_tmp_test[y_tmp_test.values != 0].loc[(aaa==1).values & (bbb!=1).values])
# aaa = y_tmp[y_tmp != 0]
# bbb = pred1vsall[y_tmp.values != 0]
# print(y_tmp[y_tmp.values != 0].loc[(aaa==1).values & (bbb!=1).values])

# # Plot Fermi position of earth when local particles occur
# plt.figure()
# plt.scatter(X.loc[df_catalog['UNC(LP)'], 'lon_fermi'].apply(lambda x: x if x < 180 else x-360),
#             X.loc[df_catalog['UNC(LP)'], 'lat_fermi'], c=X.loc[df_catalog['UNC(LP)'], 'l'])
# plt.colorbar()
# plt.figure()
# plt.boxplot(X.loc[df_catalog['UNC(LP)'], ['num_det', 'num_det_rng']])


# X['FP'] = df_catalog['FP']
# print(X[['FP', 'fe_bkg_step']].groupby('FP').agg(['mean', 'median', 'std', 'mad']))
# print(X[['FP', 'fe_bkg_step_min']].groupby('FP').agg(['mean', 'median', 'std', 'mad']))
# print(X[['FP', 'fe_bkg_step_min_norm_all']].groupby('FP').agg(['mean', 'median', 'std', 'mad']))
# print(X[['FP', 'fe_bkg_step_min_norm_all_max']].groupby('FP').agg(['mean', 'median', 'std', 'mad']))
# print(X[['FP', 'fe_bkg_step_min_norm_std']].groupby('FP').agg(['mean', 'median', 'std', 'mad']))
# print(X[['FP', 'fe_bkg_std']].groupby('FP').agg(['mean', 'median', 'std', 'mad']))
# df_catalog[(X['fe_bkg_step_min'] > 0)&(X['FP'] == True)]

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
    y_pred['UNC(LP)'] = ((((X['dist_saa_lon'] <= 9) & (X['dist_saa_lat'] <= 3.6)) |
                         ((X['dist_polo_nord_lon'] <= 19) & (X['dist_polo_nord_lat'] <= 7.6)) |
                         ((X['dist_polo_sud_lon'] <= 19) & (X['dist_polo_sud_lat'] <= 7.6)) #|
                         # ((X['l'] >= 1.55))
                         ) & ((X['num_det'] >= 9) | (X['fe_skw'] <= 0.345)) &  #  (X['fe_std_norm'] < 0.4) (X['fe_wstd1'] <= 10.365) (X['fe_wet'] < 2))
                         ((X['diff_sun'] > 35) | (np.maximum(X['ra_std'], X['dec_std']) > 100))
                         )

    # rule from Decision Tree
    #     (X['num_det'] <= 8.5)*(df_catalog['l']<=1.551)*(df_catalog['dec_std']>=1179.5)+\
    #     (X['num_det'] <= 8.5)*(df_catalog['l']>1.551)*(df_catalog['ra']>110.5)+\
    #     (X['num_det'] > 8.5)*(df_catalog['ra_std']<=100)*(df_catalog['alt_fermi']<=528117.969)+\
    #     (X['num_det'] > 8.5)*(df_catalog['ra_std']>100)*(df_catalog['l']>1.11)

    y_pred['FP'] = (X['fe_bkg_step_min_ratio_bkg_max'] <= -0.028) | (X['fe_bkg_step_min_ratio_bkg_max'] > 0.03) | (
                X['fe_std'] < 4.8)

    # y_pred['GRB'] = (df_catalog['earth_vis']) & (X['diff_sun'] >= 25) & (X['sigma_r1'] > 0) & \
    #                 (abs(df_catalog['b_galactic']) >= 0) & (X['num_det'] <= 6) & (X['sigma_r0'] <= 20)

    # y_pred['GRB'] = ((X['HR10'] > 0.64433) &
    #                  ((X['sigma_r0'] <= 17.389) | (X['qtl_cut_r1'] > 0.325) | ((X['qtl_cut_r1'] <= 0.325) &
    #                                                                            (X['num_anti_coincidence'] <= 1)))
    #                  & (~y_pred['UNC(LP)']))
    # y_pred['GRB'] = ((X['HR10'] > 0.64433) & (X['fe_wen4'] > 7.889) & (X['dist_saa'] > 9.687))#  & (~y_pred['UNC(LP)']) & (~ y_pred['SF'])
    # y_pred['GRB'] = ((X['HR10'] > 0.735) & (X['dist_saa'] > 10.225) & (X['fe_wet'] > 1.982) & (X['fe_skw_ratio_std'] > 0.03))  #(X['fe_med'] <= 11.221) (X['fe_skw_ratio_std'] > 0.045)
    # y_pred['GRB'] = ((X['HR10'] > 0.384) & (X['HR21'] <= 0.382) & (X['fe_wet'] > 1.982) & (X['fe_skw_ratio_std'] >= 0.045))  #(X['fe_med'] <= 11.221) (X['fe_skw_ratio_std'] > 0.045)
    y_pred['GRB'] = ((X['HR10'] > 0.449) & (X['HR21'] <= 0.375) & (X['fe_wet'] > 2.054))
    # y_pred['GRB'] = ((X['fe_skw_norm'] > 0.045) & (X['fe_min_norm'] <= -0.015) & (X['HR10'] > 0.404))
    # y_pred['GRB'] = ((X['HR10'] > 0.404) & (X['fe_wet'] > 2.06)) # fe_wet > 2.12, fe_std_norm > 0.753, fe_mea <= 52.291 (X['fe_min_norm'] < -0.015)
    # y_pred['GRB'] = ((X['diff_earth'] <= 0.17014) & (X['fe_mea'] <= 0.48646) | (X['HR10'] > 0.35691)) & ((X['b_galactic'] <= 0.81454) & (X['fe_max_norm'] > 0.11102) & (X['fe_med_norm'] > 0.62616) & (X['lon_fermi_shift'] <= 0.83409)) &\
    #                 ((X['fe_max_norm'] <= 0.8501) & (X['fe_med_norm'] <= 0.88899) & (X['fe_skw_norm'] <= 0.50918)) & \
    #                 ((X['HR10'] > 0.36701) & (X['dist_saa_lat'] <= 0.30618) & (X['fe_med_norm'] > 0.60515))
    # X['fe_wstd2'] <= 9.194 (X['fe_wet'] > 2.001) (X['fe_skw_norm'] >= 0.069) (X['HR10'] > 0.64433)
    # (X['fe_wen1'] > 7.889) & (X['HR21'] <= 0.492))
    # y_pred['GRB'] = ((X['fe_mea'] <= 48.119) & (X['fe_wen1'] > 7.889) & (X['fe_wet'] > 2.001))



    y_pred['UNC'] = 1 - (y_pred['SF'] | y_pred['TGF'] | y_pred['GF'] | y_pred['UNC(LP)'] | y_pred['GRB'])
    return y_pred

y_pred = classification_logic(df_catalog)

for ev_type in ['SF', 'UNC(LP)', 'FP', 'GRB']: #,  'SF', 'UNC(LP)'  # 'TGF', 'GF',  'UNC'
    print("Event: ", ev_type)
    print("Total data")
    print(confusion_matrix(df_catalog[ev_type], y_pred[ev_type]))
    print(classification_report(df_catalog[ev_type], y_pred[ev_type]))
    print("Test data")
    print(confusion_matrix(df_catalog.loc[X_test.index, ev_type], y_pred[ev_type].loc[X_test.index]))
    print(classification_report(df_catalog.loc[X_test.index, ev_type], y_pred[ev_type].loc[X_test.index]))
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
    print(X.loc[(df_catalog[ev_type] == 1) & (y_pred[ev_type] == 0), lst_X_col_show + []]) #  [i for i in X.columns if 'fe_' in i]]
    # To add
    print("False in catalog, True with new rule")
    print(df_catalog.loc[(df_catalog[ev_type] == 0) & (y_pred[ev_type] == 1), lst_col_show])
    print(X.loc[(df_catalog[ev_type] == 0) & (y_pred[ev_type] == 1), lst_X_col_show + []]) # [i for i in X.columns if 'fe_' in i]]
    print('-----------------------------------------------------------------------------------------------------------')
    pass

pass

