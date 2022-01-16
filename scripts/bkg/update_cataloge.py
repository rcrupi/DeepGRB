# import modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.localization import localization
from sqlalchemy import create_engine
# Fermi SkyPlot
from gbm.data import HealPix
from gbm.data import PosHist
from gbm.plot import SkyPlot
from gbm.finder import ContinuousFtp
# Project variables
from connections.utils.config import PATH_TO_SAVE

PATH_PRED = PATH_TO_SAVE + "pred/"
PATH_BKG = PATH_TO_SAVE + "bkg/"
if 'loc' not in os.listdir(PATH_PRED):
    os.mkdir(PATH_PRED + 'loc')

# Load catalogue of triggered events
if 'CATdatabase.db' not in os.listdir(PATH_PRED):
    df_trig = pd.read_csv(PATH_PRED + 'trigs_table.csv')
    df_trig['ra'] = None
    df_trig['dec'] = None
    df_trig['ra_montecarlo'] = None
    df_trig['dec_montecarlo'] = None
    df_trig['ra_std'] = None
    df_trig['dec_std'] = None
    engine = create_engine('sqlite:////' + PATH_PRED + 'CATdatabase.db')
    df_trig.to_sql('DEEP_TRI', index=False, con=engine)
# Not in catalogue Fermi

engine = create_engine('sqlite:////' + PATH_PRED + 'CATdatabase.db')
deep_tri = pd.read_sql_table('DEEP_TRI', con=engine)

deep_tri = deep_tri.loc[deep_tri[['ra', 'dec', 'ra_std', 'dec_std']].isna().any(axis=1)]
# Dataset xxx_101225.csv xxx_2014.csv xxx_19_01-06.csv
df_frg = pd.read_csv(PATH_PRED + 'frg_19_01-06.csv')
df_bkg = pd.read_csv(PATH_PRED + 'bkg_19_01-06.csv')
df_bkg['met'] = df_frg['met'].values
col_det = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb']

for _, row in deep_tri.iterrows():
    day_name = row.start_times[2:10].replace('-', '')
    # 101111, 140102, 140112, 190404, 190420
    df_event = pd.read_csv(PATH_BKG + day_name + '.csv')
    # 311194700, 410351700, 411228000, 576076200, 577492400
    met_event = row.start_met
    met_event_end = row.end_met
    # Energy range detected
    print('Trigger n°: ', row.trig_ids)
    print('Detectors triggered: '+row.trig_dets)
    energy_list = list((i[-1] for i in ['_r0', '_r1', '_r2'] if i in row.trig_dets))
    # Select timestamps of the background event
    df_frg_bkg = pd.merge(df_event, df_bkg, how='left', on=['met'], suffixes=('_frg', '_bkg'))
    # Column of detectors detection
    col_ra = np.sort([i for i in df_frg_bkg.columns if '_ra' in i and len(i) == 5 and 'n' in i])
    col_dec = np.sort([i for i in df_frg_bkg.columns if '_dec' in i and len(i) == 6 and 'n' in i])
    # Initialise inputs for localization
    list_ra = pd.DataFrame()
    list_dec = pd.DataFrame()
    counts = pd.DataFrame()
    counts_frg = pd.DataFrame()
    counts_bkg = pd.DataFrame()
    # select energy range to 0, 1, 2
    e_max = 0
    e_selected = None
    for e_tmp in energy_list:
        col_count_frg = np.sort([i for i in df_frg_bkg.columns if '_frg' in i and 'n' in i and '_r' + e_tmp + '_' in i])
        col_count_bkg = np.sort([i for i in df_frg_bkg.columns if '_bkg' in i and 'n' in i and '_r' + e_tmp + '_' in i])
        # Calculate the residuals
        df_frg_bkg[col_det] = df_frg_bkg[col_count_frg].values - df_frg_bkg[col_count_bkg].values
        # Filter the event with starttime (-4 seconds) and endtime of FoCus
        df_frg_bkg_event = df_frg_bkg.loc[(df_frg_bkg['met'] >= met_event - 4) & (df_frg_bkg['met'] <= met_event_end), col_det]
        # df_frg_bkg_event[col_det].plot()
        if e_max < df_frg_bkg_event.max().max():
            e_max = df_frg_bkg_event.max().max()
            e_selected = e_tmp
    # Noe choose the best energy range
    col_count_frg = np.sort([i for i in df_frg_bkg.columns if '_frg' in i and 'n' in i and '_r' + e_selected + '_' in i])
    col_count_bkg = np.sort([i for i in df_frg_bkg.columns if '_bkg' in i and 'n' in i and '_r' + e_selected + '_' in i])
    # Calculate the residuals
    df_frg_bkg[col_det] = df_frg_bkg[col_count_frg].values - df_frg_bkg[col_count_bkg].values
    # Filter the event with starttime (-4 seconds) and endtime of FoCus
    df_frg_bkg_event = df_frg_bkg.loc[
        (df_frg_bkg['met'] >= met_event - 4) & (df_frg_bkg['met'] <= met_event_end), col_det]
    # Another filter in time near the maximum values of residuals for the most triggered detector
    max_fin = 0
    ind_max = None
    high_detector = None
    for i in col_det:
      max_tmp = df_frg_bkg_event.loc[:, i].max()
      if max_tmp > max_fin:
        max_fin = max_tmp
        ind_max = df_frg_bkg_event.loc[:, i].idxmax()
        high_detector = i
    print('The index of the peak is: ', ind_max)
    index_not_event = df_frg_bkg_event.loc[(df_frg_bkg_event.loc[:, high_detector] < 0), :].index
    if len(index_not_event[index_not_event <= ind_max]) == 0:
        index_start_peak = df_frg_bkg_event.index[0]
    else:
        index_start_peak = index_not_event[index_not_event <= ind_max][-1]
    if len(index_not_event[index_not_event >= ind_max]) == 0:
        index_end_peak = df_frg_bkg_event.index[-1]
    else:
        index_end_peak = index_not_event[index_not_event >= ind_max][0]
    df_frg_bkg_event = df_frg_bkg_event.loc[index_start_peak:index_end_peak, :].dropna(axis=0)
    index_final_filter = df_frg_bkg_event.index

    col_filter = range(0, 12)
    # ind_max, index_final_filter
    list_ra = list_ra.append(df_frg_bkg.loc[ind_max, np.array(col_ra)[col_filter]]/180*np.pi) # TODO update index start/end
    list_dec = list_dec.append(df_frg_bkg.loc[ind_max, np.array(col_dec)[col_filter]]/180*np.pi)
    counts = counts.append(np.maximum(df_frg_bkg.loc[ind_max, np.array(col_det)[col_filter]], 0))
    counts_frg = counts_frg.append(df_frg_bkg.loc[ind_max, np.array(col_count_frg)[col_filter]])
    counts_bkg = counts_bkg.append(df_frg_bkg.loc[ind_max, np.array(col_count_bkg)[col_filter]])

    loc = localization(list_ra.values, list_dec.values, counts_frg.values, counts_bkg.values)
    res = loc.fit()
    print(res)
    rnd_res = loc.fit_conf_int(250)
    mean, cov = loc.plot()

    # Update calalogue DB
    sql = "UPDATE DEEP_TRI SET " + \
          " ra = " + str(res['ra']) + "," + \
          " dec = " + str(res['dec']) + "," + \
          " ra_montecarlo = " + str(mean[0]) + "," + \
          " dec_montecarlo = " + str(mean[1]) + "," + \
          " ra_std = " + str(np.sqrt(cov[0][0])) + "," + \
          " dec_std = " + str(np.sqrt(cov[1][1])) + \
          " WHERE trig_ids = " + str(row.trig_ids)
    with engine.begin() as conn:
        conn.execute(sql)

    # initialize the continuous data finder with a time (Fermi MET, UTC, or GPS)
    for j in range(0, 3):
        try:
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
            plt.savefig(PATH_PRED + 'loc/' + 'out_' + str(row.trig_ids) +'_loc.png')
            plt.close()
            break
        except Exception as e:
            print(e)
            [os.remove("tmp/" + file) for file in os.listdir("../tmp")]
            pass

pass
