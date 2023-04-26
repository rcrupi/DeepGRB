import numpy as np
import os
import pandas as pd
from connections.utils.config import FOLD_RES

pd.set_option("display.width", 1000)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)

df_class = pd.read_csv("/".join(os.getcwd().split('/')[:-1]) + "/data/DeepGRB_catalog.csv", index_col=0)

df_catalog = pd.DataFrame()
for (start_month, end_month) in [
    ("03-2019", "07-2019"),
    ("01-2014", "03-2014"),
    ("11-2010", "02-2011"),
]:
    df_catalog = df_catalog.append(
        pd.read_csv(FOLD_RES + 'frg_' + start_month + '_' + end_month + '/events_table_loc.csv')
    )

df_catalog = df_catalog.reset_index(drop=True)
df_catalog['datetime'] = df_catalog['start_times'].str.slice(0, 19)

df_catalog = pd.merge(df_catalog, df_class[['datetime', 'catalog_triggers']], how="left", on=["datetime"])
df_catalog = df_catalog.dropna(subset=['catalog_triggers_y'])
df_catalog['catalog_triggers'] = df_catalog['catalog_triggers_y']
del df_catalog['catalog_triggers_x'], df_catalog['catalog_triggers_y']

df_catalog = df_catalog[df_catalog['catalog_triggers'].str.contains("UNKNOWN")]

print(df_catalog['catalog_triggers'].unique())

# Target variable
# GRB: Gamma-Ray burst
# SF: Solar flare
# UNC(LP): Local particles
# TGF: Terrestrial Gamma-Ray Flash
# GF: Galactic flare
# UNC: Uncertain classification
# # # Analysis of UNKNOWN events

ev_type_list = ['GRB', 'SF', 'UNC(LP)', 'TGF', 'GF', 'UNC']
for ev_type in ev_type_list:
    df_catalog[ev_type] = df_catalog['catalog_triggers'].apply(lambda x: ev_type in x.replace("UNKNOWN: ", "").split("/"))

print(df_catalog[ev_type_list].sum())

# Prepare the dataset
X = df_catalog[['trig_dets', 'sigma_r0', 'sigma_r1', 'sigma_r2', 'duration',
                # 'qtl_cut_r0', 'qtl_cut_r1', 'qtl_cut_r2',
                'ra', 'dec',
                # 'ra_montecarlo', 'dec_montecarlo', 'ra_std', 'dec_std',
                'ra_earth', 'dec_earth',
                'earth_vis',
                'sun_vis',
                'ra_sun', 'dec_sun',
                # 'l_galactic',
                'b_galactic',
                'lat_fermi', 'lon_fermi',
                # 'alt_fermi',
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

X['diff_sun'] = np.minimum(abs(X['ra'] - X['ra_sun']), 360 - abs(X['ra'] - X['ra_sun'])) +\
                np.minimum(abs(X['dec'] - X['dec_sun']), 180 - abs(X['dec'] - X['dec_sun']))
X['diff_earth'] = np.minimum(abs(X['ra'] - X['ra_earth']), 360 - abs(X['ra'] - X['ra_earth'])) +\
                np.minimum(abs(X['dec'] - X['dec_earth']), 180 - abs(X['dec'] - X['dec_earth']))
del X['ra_sun'], X['dec_sun'], X['ra_earth'], X['dec_earth'], X['ra'], X['dec']


# Train a DT
for ev_type in ev_type_list[2:3]:
    print("Type of event analysed: ", ev_type)
    y = df_catalog[ev_type]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

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
    plt.title(ev_type)
    plt.show()

# Plot Fermi position of earth when local particles occur
plt.scatter(X.loc[df_catalog['UNC(LP)'], 'lon_fermi'].apply(lambda x: x if x < 180 else x-360),
            X.loc[df_catalog['UNC(LP)'], 'lat_fermi'])
plt.boxplot(X.loc[df_catalog['UNC(LP)'], 'num_det_rng'])

# Manual classification logic
y_pred = {}
y_pred['SF'] = (X['diff_sun'] < 50) & (X['sigma_r0'] > 0) & (df_catalog['sun_vis'])
y_pred['TGF'] = (1-df_catalog['earth_vis']) | (X['diff_earth'] < 80)
y_pred['GF'] = (abs(df_catalog['b_galactic']) < 10) & (df_catalog['earth_vis'])
y_pred['UNC(LP)'] = ((((15 < df_catalog['lat_fermi']) & (df_catalog['lat_fermi'] < 30)) &
              ((220 < df_catalog['lon_fermi']) & (df_catalog['lon_fermi'] < 275))) |
             (((-30 < df_catalog['lat_fermi']) & (df_catalog['lat_fermi'] < 8)) &
             ((230 < df_catalog['lon_fermi']) & (df_catalog['lon_fermi'] < 360))) |
            ((df_catalog['lat_fermi'] < -10) & (df_catalog['lon_fermi'] > 0))) &\
                    ((X['num_det'] >= 9) | (X['num_det_rng'] >= 12))
y_pred['GRB'] = (df_catalog['earth_vis']) & (X['diff_sun'] >= 50) & (X['sigma_r1'] > 0) & \
             (abs(df_catalog['b_galactic']) >= 10)
y_pred['UNC'] = 1 - (y_pred['SF'] | y_pred['TGF'] | y_pred['GF'] | y_pred['UNC(LP)'] | y_pred['GRB'])

for ev_type in ev_type_list:
    print("Event: ", ev_type)
    print(confusion_matrix(df_catalog[ev_type], y_pred[ev_type]))
    print(df_catalog[(df_catalog[ev_type] == 1) & (y_pred[ev_type] == 0)])
    print(X[(df_catalog[ev_type] == 1) & (y_pred[ev_type] == 0)])
    input()

pass
