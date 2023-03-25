import pandas as pd
import numpy as np
from pipeline.manual_label import p_manual_2011, p_manual_2014, p_manual_2019, selected_trig_eve, event_2010, \
    event_2014, event_2019, the_events
from connections.utils.config import FOLD_RES

pd.options.display.max_rows = 100
pd.options.display.max_columns = 100
pd.options.display.width = 1000

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
df_tmp['period'] = (df_tmp['start_times'].str.slice(0, 4)).apply(lambda x: '2010' if x == '2011' else x)
df_tmp['trig_ids'] = df_tmp['period'] + '_' + df_tmp['trig_ids'].astype(str)
df_tmp['datetime'] = df_tmp['start_times'].str.slice(0, 19)
df_tmp['det trigs'] = df_tmp['trig_dets'].apply(lambda x: ' '.join(np.sort(list(set([i.split('_')[0] for i in x.split(' ')])))))
df_tmp['start_met'] = df_tmp['start_met'].astype('int')
df_tmp['end_met'] = df_tmp['end_met'].astype('int')
df_tmp['sigma_r0'] = df_tmp['sigma_r0'].round(2)
df_tmp['sigma_r1'] = df_tmp['sigma_r1'].round(2)
df_tmp['sigma_r2'] = df_tmp['sigma_r2'].round(2)
df_tmp['duration'] = df_tmp['duration'].round(2)
df_tmp['sigma_max'] = df_tmp[['sigma_r0', 'sigma_r1', 'sigma_r2']].max(axis=1)

df_tmp2 = df_tmp[['trig_ids', 'datetime', 'duration', 'det trigs', 'catalog_triggers',
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

df_tmp2['S_type'] = 'None'
df_tmp2.loc[(((df_tmp2[['sigma_r0', 'sigma_r1', 'sigma_r2']]>0).sum(axis=1)>1)&\
             ((df_tmp2['det trigs'].str.len())>2)), 'S_type'] = 'R'
df_tmp2.loc[(((df_tmp2[['sigma_r0', 'sigma_r1', 'sigma_r2']]>0).sum(axis=1)==1)&\
                   ((df_tmp2['det trigs'].str.len())>2)), 'S_type'] = 'S'
df_tmp2.loc[ (((df_tmp2['det trigs'].str.len())==2)), 'S_type'] = 'P'

for col_sigma in ['sigma_r0', 'sigma_r1', 'sigma_r2']:
    if df_tmp2.loc[df_tmp2[col_sigma] < 0, :].shape[0] > 0:
        print(df_tmp2.loc[df_tmp2[col_sigma] < 0, :])
        df_tmp2.loc[df_tmp2[col_sigma] < 0, col_sigma] = 0
    # Set a symbol to avoid enormous standard score
    #df_tmp2[col_sigma] = df_tmp2[col_sigma].astype('str')
    df_tmp2.loc[df_tmp2[col_sigma] > 10, col_sigma] = '$>10$'

df_tmp_sorted = df_tmp.sort_values(by=['sigma_max'])
print(df_tmp2[df_tmp2['datetime'].isin(selected_trig_eve)])

# Join tentative assign class transient
ev_class_2010 = pd.DataFrame({'trig_ids': ['2010_'+str(i) for i in event_2010.keys()], 'class': event_2010.values()})
ev_class_2014 = pd.DataFrame({'trig_ids': ['2014_'+str(i) for i in event_2014.keys()], 'class': event_2014.values()})
ev_class_2019 = pd.DataFrame({'trig_ids': ['2019_'+str(i) for i in event_2019.keys()], 'class': event_2019.values()})
ev_class = ev_class_2010.append(ev_class_2014, ignore_index=True).append(ev_class_2019, ignore_index=True)
df_tmp2_class = pd.merge(df_tmp2, ev_class, how='left', on=['trig_ids'])

# Unknown class - add tentative event class
idx_tmp = np.where(df_tmp2_class.loc[df_tmp2_class['catalog_triggers'] == 'UNKNOWN', 'class'].isna())
if len(idx_tmp):
    print(df_tmp2_class.loc[df_tmp2_class['catalog_triggers'] == 'UNKNOWN', :].iloc[idx_tmp])
idx_tmp = np.where(df_tmp2_class.loc[df_tmp2_class['catalog_triggers'] != 'UNKNOWN', 'class'].notna())
if len(idx_tmp):
    print(df_tmp2_class.loc[df_tmp2_class['catalog_triggers'] != 'UNKNOWN', :].iloc[idx_tmp])

df_tmp2_class['class'] = df_tmp2_class['class'].fillna('')

idx_unknown = df_tmp2_class['catalog_triggers'] == 'UNKNOWN'
df_tmp2_class.loc[idx_unknown, 'catalog_triggers'] = 'UNKNOWN: ' + df_tmp2_class['class']
del df_tmp2_class['class']

# add * to the seven events
for trig_ids_tmp in the_events:
    df_tmp2_class.loc[df_tmp2_class['trig_ids'] == trig_ids_tmp, 'trig_ids'] = trig_ids_tmp + '*'

# print latex tables
print(df_tmp2_class[idx_unknown].to_latex())
print(df_tmp2_class[(idx_unknown!=True)].to_latex())

pass