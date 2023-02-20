import pandas as pd
import numpy as np
from pipeline.manual_label import p_manual_2011, p_manual_2014, p_manual_2019
from connections.utils.config import FOLD_RES
path_1 = FOLD_RES
p_2014 = "frg_01-2014_03-2014/"
p_2019 = "frg_03-2019_07-2019/"
p_2011 = "frg_11-2010_02-2011/"

df_tmp = pd.DataFrame()
e_v = pd.read_csv(path_1 + p_2011 + "events_table.csv")
e_v['catalog_triggers'] = p_manual_2011
df_tmp = df_tmp.append(e_v)
e_v = pd.read_csv(path_1 + p_2014 + "events_table.csv")
e_v['catalog_triggers'] = p_manual_2014
df_tmp = df_tmp.append(e_v)
e_v = pd.read_csv(path_1 + p_2019 + "events_table.csv")
e_v['catalog_triggers'] = p_manual_2019
df_tmp = df_tmp.append(e_v)

df_tmp = df_tmp.reset_index(drop=True)
df_tmp.drop(columns=['start_index', 'end_index', 'end_times'], inplace=True)
df_tmp['period'] = (df_tmp['start_times'].str.slice(0, 4)).apply(lambda x: '2011' if x == '2010' else x)
df_tmp['trig_ids'] = df_tmp['period'] + '_' + df_tmp['trig_ids'].astype(str)
df_tmp['datetime'] = df_tmp['start_times'].str.slice(0, 19)
df_tmp['det trigs'] = df_tmp['trig_dets'].apply(lambda x: ' '.join(np.sort(list(set([i.split('_')[0] for i in x.split(' ')])))))
df_tmp['start_met'] = df_tmp['start_met'].astype('int')
df_tmp['end_met'] = df_tmp['end_met'].astype('int')
df_tmp['sigma_r0'] = df_tmp['sigma_r0'].round(2)
df_tmp['sigma_r1'] = df_tmp['sigma_r1'].round(2)
df_tmp['sigma_r2'] = df_tmp['sigma_r2'].round(2)
df_tmp['sigma_max'] = df_tmp[['sigma_r0', 'sigma_r1', 'sigma_r2']].max(axis=1)

df_tmp2 = df_tmp[['trig_ids', 'datetime', 'start_met', 'end_met', 'det trigs', 'catalog_triggers',
                 'sigma_r0', 'sigma_r1', 'sigma_r2', 'sigma_max']]

df_tmp2 = df_tmp2.loc[(df_tmp2['catalog_triggers'] != 'f') & (df_tmp2['catalog_triggers'] != 'f (ssa)')]
df_tmp2['catalog_triggers'] = df_tmp2['catalog_triggers'].replace('u', 'UNKNOWN')
df_tmp2 = df_tmp2.reset_index(drop=True)

# All
print('Stat. with UNKNOWN', df_tmp2.loc[df_tmp2['catalog_triggers'] == 'UNKNOWN',
                                        ['sigma_r0', 'sigma_r1', 'sigma_r2', 'sigma_max']].describe())
print('Stat. with event in catalog', df_tmp2.loc[df_tmp2['catalog_triggers'] != 'UNKNOWN',
                                                 ['sigma_r0', 'sigma_r1', 'sigma_r2', 'sigma_max']].describe())
# 2011
print('Stat. with UNKNOWN 2011', df_tmp2.loc[(df_tmp2['catalog_triggers'] == 'UNKNOWN')&(
        (df_tmp2['datetime'].str.slice(0, 4)=='2010')|(df_tmp2['datetime'].str.slice(0, 4)=='2011')),
                                        ['sigma_r0', 'sigma_r1', 'sigma_r2', 'sigma_max']].describe())
print('Stat. with event in catalog', df_tmp2.loc[(df_tmp2['catalog_triggers'] != 'UNKNOWN')&(
        (df_tmp2['datetime'].str.slice(0, 4)=='2010')|(df_tmp2['datetime'].str.slice(0, 4)=='2011')),
                                                 ['sigma_r0', 'sigma_r1', 'sigma_r2', 'sigma_max']].describe())
# 2014
print('Stat. with UNKNOWN 2014', df_tmp2.loc[(df_tmp2['catalog_triggers'] == 'UNKNOWN')&(df_tmp2['datetime'].str.slice(0, 4)=='2014'),
                                        ['sigma_r0', 'sigma_r1', 'sigma_r2', 'sigma_max']].describe())
print('Stat. with event in catalog', df_tmp2.loc[(df_tmp2['catalog_triggers'] != 'UNKNOWN')&(df_tmp2['datetime'].str.slice(0, 4)=='2014'),
                                                 ['sigma_r0', 'sigma_r1', 'sigma_r2', 'sigma_max']].describe())
# 2019
print('Stat. with UNKNOWN 2019', df_tmp2.loc[(df_tmp2['catalog_triggers'] == 'UNKNOWN')&(df_tmp2['datetime'].str.slice(0, 4)=='2019'),
                                        ['sigma_r0', 'sigma_r1', 'sigma_r2', 'sigma_max']].describe())
print('Stat. with event in catalog', df_tmp2.loc[(df_tmp2['catalog_triggers'] != 'UNKNOWN')&(df_tmp2['datetime'].str.slice(0, 4)=='2019'),
                                                 ['sigma_r0', 'sigma_r1', 'sigma_r2', 'sigma_max']].describe())

del df_tmp2['sigma_max']

print(df_tmp2.to_latex())
