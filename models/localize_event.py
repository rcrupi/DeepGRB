# import modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.loc.localization_class import localization
from connections.utils.config import PATH_TO_SAVE, FOLD_PRED, FOLD_BKG, FOLD_RES, FOLD_POSHIST
from gbm.time import Met
from pathlib import Path

# Fermi SkyPlot
from gbm.data import HealPix, PosHist
from gbm.plot import SkyPlot, EarthPlot
from gbm.finder import ContinuousFtp


def localize(start_month, end_month, pre_delay=8, bln_only_trig_det=False, bln_folder=True, trig_id=None):
    """
    Update event catalog with a draft localization estimate and uncertainty (based on montecarlo simulation).
    :param start_month: start of the period
    :param end_month: end of the period
    :param pre_delay: int, seconds before the starttime of the event
    :param bln_only_trig_det: bool, if True only the triggered detectors are used. False otherwise.
    :param bln_folder: bool, if True the poshist files are save in FOLD_POSHIST.
                        If False, are saved and deleted in a temporary folder.
    :param trig_id: int, specify the trigger event. Catalog won't be saved.
    :return: None
    """
    # Read Dataset foreground and background
    df_frg = pd.read_csv(PATH_TO_SAVE + FOLD_PRED + "/" + 'frg_' + start_month + '_' + end_month + '.csv')
    df_bkg = pd.read_csv(PATH_TO_SAVE + FOLD_PRED + "/" + 'bkg_' + start_month + '_' + end_month + '.csv')

    # Select interval of triggers in events_table.csv
    folder_name = 'frg_' + start_month + '_' + end_month + "/"
    folder_result = FOLD_RES + folder_name
    ev_tab = pd.read_csv(folder_result + 'events_table.csv')
    if trig_id is not None:
        ev_tab = ev_tab.loc[ev_tab['trig_ids'] == trig_id]
    # Add new columns
    ev_tab['ra'] = None
    ev_tab['dec'] = None
    ev_tab['ra_montecarlo'] = None
    ev_tab['dec_montecarlo'] = None
    ev_tab['ra_std'] = None
    ev_tab['dec_std'] = None
    # Create folder for localization plots
    plot_loc_folder = Path(folder_result + "plots/loc/")
    plot_loc_folder.mkdir(parents=True, exist_ok=True)
    # Create folder for poshist files
    poshist_folder = Path(PATH_TO_SAVE) / FOLD_POSHIST
    poshist_folder.mkdir(parents=True, exist_ok=True)
    # Parameters
    n_sample_montecarlo = 250
    std_sky_plot_gaussian = 10

    # Iterate per each event in catalog
    for index, row in ev_tab.iterrows():
        print(row['start_met'], 'trig id: ', row['trig_ids'])

        try:
            # Define times of the event
            timestamp_event = row['start_times']
            met_event = row['start_met']
            met_event_end = row['end_met']
            # Detectors and rages triggered
            trig_dets = [i.split('_')[0] for i in row['trig_dets'].split(' ')]
            rng_dets = [i.split('_')[1] for i in row['trig_dets'].split(' ')]
            rng_dets_cnt = [sum(np.array(rng_dets) == i) for i in ['r0', 'r1', 'r2']]
            rng_max = np.argmax(rng_dets_cnt)
            # Day of the trigger
            day_event = ''.join(timestamp_event[2:10].split('-'))
            df_event = pd.read_csv(PATH_TO_SAVE + FOLD_BKG + '/' + day_event + '.csv')
            # Merge bkg and frg
            df_bkg['met'] = df_frg['met'].values
            df_frg_bkg = pd.merge(df_event, df_bkg, how='left', on=['met'], suffixes=('_frg', '_bkg'))
            # Select directions of detectors
            col_ra = np.sort([i for i in df_frg_bkg.columns if '_ra' in i and len(i) == 5 and 'n' in i])
            col_dec = np.sort([i for i in df_frg_bkg.columns if '_dec' in i and len(i) == 6 and 'n' in i])
            # select energy range with the max number of detectors
            col_count_frg = np.sort([i for i in df_frg_bkg.columns if '_frg' in i and 'n' in i and '_r' +
                                     str(rng_max) + '_' in i])
            col_count_bkg = np.sort([i for i in df_frg_bkg.columns if '_bkg' in i and 'n' in i and '_r' +
                                     str(rng_max) + '_' in i])
            # Define columns for residual counts
            col_det = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb']
            # Calculate the residual
            df_frg_bkg[col_det] = df_frg_bkg[col_count_frg].values - df_frg_bkg[col_count_bkg].values
            # Select the period of the event
            df_frg_bkg_event = df_frg_bkg.loc[(df_frg_bkg['met'] > met_event - pre_delay) &
                                              (df_frg_bkg['met'] < met_event_end),
                                              col_det]
            # Find the peak of energy of the event. From that index localization is computed
            max_fin = 0
            ind_max = None
            # high_detector = None
            for i in col_det:
                max_tmp = df_frg_bkg_event.loc[:, i].max()
                if max_tmp > max_fin:
                    max_fin = max_tmp
                    ind_max = df_frg_bkg_event.loc[:, i].idxmax()
                    # high_detector = i
            print('The index of the peak is: ', ind_max)

            # # Select the interval of the period around the peak that has positive residuals
            # index_not_event = df_frg_bkg_event.loc[(df_frg_bkg_event.loc[:, high_detector] < 0), :].index
            # if len(index_not_event[index_not_event <= ind_max]) == 0:
            #     index_start_peak = df_frg_bkg_event.index[0]
            # else:
            #     index_start_peak = index_not_event[index_not_event <= ind_max][-1]
            # if len(index_not_event[index_not_event >= ind_max]) == 0:
            #     index_end_peak = df_frg_bkg_event.index[-1]
            # else:
            #     index_end_peak = index_not_event[index_not_event >= ind_max][0]
            # df_frg_bkg_event = df_frg_bkg_event.loc[index_start_peak:index_end_peak, :].dropna(axis=0)

            # Initialise inputs for localization
            list_ra = pd.DataFrame()
            list_dec = pd.DataFrame()
            counts_frg = pd.DataFrame()
            counts_bkg = pd.DataFrame()
            # Choose only triggered detectors or all detectors for localization
            if bln_only_trig_det:
                col_filter = [i for i in range(0, 12) if col_det[i] in trig_dets]
            else:
                col_filter = range(0, 12)
            # Define list ra and dec and their relative counts in ind_max. TODO update index start/end
            list_ra = list_ra.append(
              df_frg_bkg.loc[ind_max, np.array(col_ra)[col_filter]] / 180 * np.pi)
            list_dec = list_dec.append(df_frg_bkg.loc[ind_max, np.array(col_dec)[col_filter]] / 180 * np.pi)
            counts_frg = counts_frg.append(df_frg_bkg.loc[ind_max, np.array(col_count_frg)[col_filter]])
            counts_bkg = counts_bkg.append(df_frg_bkg.loc[ind_max, np.array(col_count_bkg)[col_filter]])
            # Run localization algorithm
            loc = localization(list_ra.values, list_dec.values, counts_frg.values, counts_bkg.values)
            res = loc.fit()
            print(res)
            _ = loc.fit_conf_int(n_sample_montecarlo)
            mean, cov = loc.plot()
            # Download and load poshist files
            if bln_folder:
                cont_finder = ContinuousFtp(met=int(met_event))
                poshist_name = cont_finder.ls_poshist()[0]
                if poshist_name not in os.listdir(PATH_TO_SAVE + FOLD_POSHIST):
                    try:
                        cont_finder = ContinuousFtp(met=int(met_event))
                        cont_finder.get_poshist(PATH_TO_SAVE + FOLD_POSHIST)
                    except:
                        print('Error in downloading poshist', row['trigs_id'])
                        continue
                # Open a poshist file
                poshist = PosHist.open(PATH_TO_SAVE + FOLD_POSHIST + "/" + poshist_name)
            else:
                # initialize the continuous data finder with a time (Fermi MET, UTC, or GPS)
                cont_finder = ContinuousFtp(met=met_event)
                poshist_name = cont_finder.ls_poshist()[0]
                cont_finder.get_poshist('./tmp_pos')
                # open a poshist file
                poshist = PosHist.open('./tmp_pos/' + poshist_name)
                os.remove('./tmp_pos/' + poshist_name)

            # initialize plot
            skyplot = SkyPlot()
            # Plot the orientation of the detectors and Earth blockage at our time of interest
            # Use the begin of the event (met_event) to diplay the detectors on the map
            skyplot.add_poshist(poshist, trigtime=met_event)
            gauss_map = HealPix.from_gaussian(np.round(res['ra']), np.round(res['dec']), std_sky_plot_gaussian)
            skyplot.add_healpix(gauss_map)
            plt.title(str(row['start_times'])[:-7] + "\n" + str(np.unique(trig_dets)))
            plt.xlabel(str(row['catalog_triggers']))
            plt.show()
            # Save localization sky plot
            if trig_id is None:
                plt.savefig(folder_result + "plots/loc/out" + str(row['trig_ids']) + '_loc.png')
                plt.close('all')
            # Update catalog for event trig_ids
            ev_tab.loc[ev_tab['trig_ids'] == row['trig_ids'], 'ra'] = np.round(res['ra'], 0)
            ev_tab.loc[ev_tab['trig_ids'] == row['trig_ids'], 'dec'] = np.round(res['dec'], 0)
            ev_tab.loc[ev_tab['trig_ids'] == row['trig_ids'], 'ra_montecarlo'] = np.round(mean[0], 0)
            ev_tab.loc[ev_tab['trig_ids'] == row['trig_ids'], 'dec_montecarlo'] = np.round(mean[1], 0)
            ev_tab.loc[ev_tab['trig_ids'] == row['trig_ids'], 'ra_std'] = np.round(cov[0][0], 0)
            ev_tab.loc[ev_tab['trig_ids'] == row['trig_ids'], 'dec_std'] = np.round(cov[1][1], 0)

        except Exception as e:
            print(e)
            print("Error. Problem with event id: ", row['trig_ids'])
    # Save catalog csv
    if trig_id is None:
        ev_tab.to_csv(folder_result + 'events_table_loc.csv', index=False)
    return None


def plot_gbm_loc(starttime, endtime=None, type_plot='earth'):
    """
    Plot the localization of the gbm satellite in a interval
    :param starttime: int or str, start time in met or str of the interval. E.g. 581718900 or "2019-06-06 14:56:00".
    :param endtime: int or str, start time in met or str of the interval. E.g. 581719400 or "2019-06-06 14:56:00".
    :return: None
    """
    if endtime is None:
        endtime = starttime
    if isinstance(starttime, str) and isinstance(endtime, str):
        obj_met = Met(0)
        met_start = obj_met.from_iso(starttime.replace(" ", "T")).met
        met_end = obj_met.from_iso(endtime.replace(" ", "T")).met
    elif (isinstance(starttime, float) and isinstance(endtime, float))\
            or (isinstance(starttime, int) and isinstance(endtime, int)):
        met_start = starttime
        met_end = endtime
    else:
        print("Error type of starttime and endtime.")
        return None
    met_event = int((met_start + met_end) / 2)
    cont_finder = ContinuousFtp(met=met_event)
    poshist_name = cont_finder.ls_poshist()[0]
    cont_finder.get_poshist('./tmp_pos')
    # open a poshist file
    poshist = PosHist.open('./tmp_pos/' + poshist_name)
    os.remove('./tmp_pos/' + poshist_name)
    if type_plot == 'sky':
        skyplot = SkyPlot()
        skyplot.add_poshist(poshist, trigtime=met_event)
    if type_plot == 'earth':
        earthplot = EarthPlot()
        earthplot.add_poshist(poshist, trigtime=met_event, time_range=(met_start, met_end))
    plt.show()


def compute_flux(df_frg, df_bkg, ev_tab, bln_max=True):
    """
    Plot the count rates sum of the event in the catalog versus the duration.
    :param df_frg: Dataframe of foreground
    :param df_bkg: Dataframe of background
    :param ev_tab: Table of events
    :param bln_max: Boolean, if True is it summed the count rates of the most triggered detector.
                    If False all triggered detectors are summed.
    :return: None
    """
    # Add proxy variables for Fluence, number of detector triggered and event duration
    ev_tab['fluence'] = None
    ev_tab['n_det'] = None
    ev_tab['duration'] = ev_tab.end_met - ev_tab.start_met
    # Iterate per each event in catalog
    for index, row in ev_tab.iterrows():
        print(row['start_met'], 'trig id: ', row['trig_ids'])
        trig_dets = list(set([i.split('_')[0] for i in row['trig_dets'].split(' ')]))
        ev_tab.loc[index, 'n_det'] = len(trig_dets)
        lst_det_rng = []
        for i in trig_dets:
            lst_det_rng.append(i+'_r0')
            lst_det_rng .append(i+'_r1')
            lst_det_rng.append(i+'_r2')
        time_index = (df_frg.met >= row['start_met']) & (df_frg.met <= row['end_met'])
        if bln_max:
            idx_max = (np.maximum(df_frg.loc[time_index, lst_det_rng] - df_bkg.loc[time_index, lst_det_rng], 0)).sum().idxmax()
            det_max = idx_max.split('_')[0]
            lst_det_rng = [det_max+'_r0', det_max+'_r1', det_max+'_r2']
        # Sum the count rates accordingly to the bln_max logic
        ev_tab.loc[index, 'fluence'] = (np.maximum(df_frg.loc[time_index, lst_det_rng] -
                                                       df_bkg.loc[time_index, lst_det_rng], 0).sum().sum())
    # Compute Flux
    ev_tab['flux'] = ev_tab['fluence'] / ev_tab['duration']
    # Plot the scatter plot for events in catalog and not
    plt.scatter(ev_tab.loc[ev_tab['catalog_triggers'].isna(), 'duration'],
                ev_tab.loc[ev_tab['catalog_triggers'].isna(), 'flux'], label='Not in catalog')
    plt.scatter(ev_tab.loc[ev_tab['catalog_triggers'].notna(), 'duration'],
                ev_tab.loc[ev_tab['catalog_triggers'].notna(), 'flux'], label='In catalog')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Duration [s]')
    plt.legend()
    plt.ylabel('Proxy Flux [count rate / $s^2$]')
