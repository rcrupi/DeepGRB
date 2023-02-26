import pandas as pd
import numpy as np
from scipy import stats
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from connections.utils.config import PATH_TO_SAVE, FOLD_TRIG, FOLD_RES, FOLD_PRED, GBM_BURST_DB, GBM_TRIG_DB
from utils.keys import get_keys, filter_keys
from ast import literal_eval
from bisect import bisect_left, bisect_right
from itertools import groupby
from operator import itemgetter
from math import ceil
import sqlite3
from pathlib import Path
from typing import List
import matplotlib

matplotlib.use('agg')

pd.options.display.max_rows = 100
pd.options.display.max_columns = 100
pd.options.display.width = 1000

BINLENGTH = 4.096
MIN_DET_NUMBER = 1
MAX_DET_NUMBER = 13


class MissingDataError(Exception):
    """An error when missing data."""


def init(start_month, end_month):
    global fermi
    global nn
    global trigger_catalog
    global focus
    global trigs
    global sigma_residual
    global offset

    fermi = pd.read_csv(PATH_TO_SAVE + FOLD_PRED + "/" + 'frg_' + start_month + '_' + end_month + '.csv')
    nn = pd.read_csv(PATH_TO_SAVE + FOLD_PRED + "/" + 'bkg_' + start_month + '_' + end_month + '.csv')
    focus = pd.read_csv(PATH_TO_SAVE + FOLD_TRIG + "/" + 'trig_' + start_month + '_' + end_month + '.csv')
    offset = pd.read_csv(PATH_TO_SAVE + FOLD_TRIG + "/" + 'offset_' + start_month + '_' + end_month + '.csv')

    mad = stats.median_abs_deviation(fermi[nn.columns] - nn, axis=0, scale="normal", nan_policy="omit")
    sigma_residual = dict(zip(nn.columns, mad))

    trigger_catalog = crop_catalog(fermi.met.values[0], fermi.met.values[-1])
    trigs = _trigs_make()


def analyze(start_month, end_month, threshold, type_time='t90', type_counts='flux', bln_plot=True):
    print("Analysis started.")
    # prepare directories and stuff
    init(start_month, end_month)
    folder_name = 'frg_' + start_month + '_' + end_month + "/"
    results_folder = Path(FOLD_RES) / folder_name
    results_folder.mkdir(parents=True, exist_ok=True)
    plots_folder = Path(FOLD_RES) / folder_name / "plots/"
    plots_folder.mkdir(parents=True, exist_ok=True)
    triggers_plots_folder = Path(FOLD_RES) / folder_name / "plots/triggers/"
    triggers_plots_folder.mkdir(parents=True, exist_ok=True)
    bursts_plots_folder = Path(FOLD_RES) / folder_name / "plots/bursts/"
    bursts_plots_folder.mkdir(parents=True, exist_ok=True)
    events_data_folder = Path(FOLD_RES) / folder_name / "data/events/"
    events_data_folder.mkdir(parents=True, exist_ok=True)
    bursts_data_folder = Path(FOLD_RES) / folder_name / "data/bursts/"
    bursts_data_folder.mkdir(parents=True, exist_ok=True)

    # fetch triggers from trigger table
    triggers_limits = fetch_triggers(focus, threshold, MIN_DET_NUMBER, MAX_DET_NUMBER)
    # triggers_limits_2 = fetch_triggers(focus, threshold/3, MIN_DET_NUMBER, MAX_DET_NUMBER)
    #
    # triggers_limits_extended = []
    # for t in triggers_limits:
    #     for t2 in triggers_limits_2:
    #         if t[0] >= t2[0] and t[1] <= t2[1]:
    #             if (t2[0], t[1]) not in triggers_limits_extended:
    #                 triggers_limits_extended.append((t2[0], t[1]))
    #             break
    # lst_true = np.array([False]*triggers_limits_extended[-1][1])
    # for t in triggers_limits_extended:
    #     lst_true[t[0]:t[1]] = True
    # bln_tmp = False
    # triggers_limits = []
    # counter = 0
    # for i in lst_true:
    #     if i and not bln_tmp:
    #         t0 = counter
    #         bln_tmp = True
    #     if not i and bln_tmp:
    #         t1 = counter
    #         bln_tmp = False
    #         triggers_limits.append((t0, t1))
    #     counter += 1
    # if i:
    #     triggers_limits.append((t0, counter))

    triggers = [Segment(*t) for t in triggers_limits]
    triggers_table = tableize(triggers, threshold, sigma_residual)
    triggers_table.to_csv(results_folder / 'triggers_table.csv', index=False)
    print("found {} trigger segments".format(len(triggers)))
    print('made triggers table.')

    events_limits = merge(triggers_limits, length = int(600/BINLENGTH))
    events = [Segment(*t) for t in events_limits]
    events_table = tableize(events, threshold, sigma_residual)
    events_table.to_csv(results_folder / 'events_table.csv', index=False)
    stat_table = statistics_table(events_table)
    stat_table.to_csv(results_folder / 'stat_table.csv', index=True)
    print("found {} events".format(len(events)))
    print('made events table.')

    # check against catalogs
    detected, undetected, missing = check_against_gbmcatalogs(threshold, type_time, type_counts)
    print('detected {} events in GBM trig catalog;\nundetected: {};\nmissing: {}'
          .format(len(detected), len(undetected), len(missing)))

    with open(results_folder / 'summary.txt', "w") as f:
        f.write("found {} trigger segments.\n".format(len(triggers)))
        f.write("from which {} events were resolved.\n".format(len(events)))
        f.write(
            "detected {} events in GBM trig catalog;\nundetected: {};\nmissing: {}"
            .format(len(detected), len(undetected), len(missing))
        )

    if not bln_plot:
        return True

    # stat and events
    # events_table = triggers_table(segments, threshold)
    # events_table.to_csv(results_folder / 'segments_table.csv', index=False)
    # events_table_red, events = reduce_table(events_table.copy(), segments, t_filt=600)
    # events_table_red.to_csv(results_folder / 'events_table.csv', index=False)
    # stat_table = statistics_table(events_table)
    # stat_table.to_csv(results_folder / 'stat_table.csv', index=True)
    # print("found {} events".format(len(events)))
    # print('made events table.')

    # plots and data
    save_greenred_plot(detected, undetected, missing, plots_folder, type_time, type_counts)
    save_gbmbursts_plots(threshold, bursts_plots_folder)
    save_triggers_plots(events, threshold, triggers_plots_folder)
    export_events_data(events, events_data_folder)
    export_bursts_data(bursts_data_folder)
    # print('made plots.')
    print('Analysis complete.')
    return True


def merge(data, length):
    """
    inputs a list of 2-tuples and an int.
    outputs a list of 2-tuples.

    example:
    data = [(1,4), (5,9), (10, 11), (12, 13), (20, 24), (25, 26)]
    len = 10

    returns:
    [(1,9), (10, 13), (20, 26)]

    (1,4) and (5,9) are merged in (1,9) since 9 - 1 < 10.
    """
    out = []
    i = 0
    while i < len(data):
        j = 0
        while (
            (i + j < len(data))
            and (data[i + j][1] - data[i][0] < length)
        ):
            j += 1

        if j == 0:
            out.append((data[i][0], data[i][1]))
            i += 1
        else:
            out.append((data[i][0], data[i + j - 1][1]))
            i = i + j
    return out


def fetch_triggers(table, threshold, min_dets_num, max_dets_num):
    """
    implements trigger condition.
    inputs: a table of trigger values and the parameters for the trigger condition.
    outputs: a list of 2-tuples.
    each of the output tuples represents the start and the end index of a trigger.
    trigger condition is verified if table values are greater
    than threshold over at least one range, simultaneously for a number of
    detectors greater-equal than min_dets_num and less than max_dets_num.
    """
    out = {}
    # iter over detectors and check where significance values overcome threshold
    # over at least one energy range. merge into one single column per detector.
    for i in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b']:
        table_ni = table[get_keys(ns=[i], rs=['1'])]
        out[i] = table_ni[table_ni > threshold].any(axis=1)
    merged_ranges_df = pd.DataFrame(out)

    # check where a minimum number of detectors were above threshold
    dets_over_trig = merged_ranges_df[merged_ranges_df == True].count(axis=1)
    data = dets_over_trig[dets_over_trig >= min_dets_num]

    trig_segs = []
    # compile output. do not include a trigger if it spans
    # at least max_dets_num detectors.
    for k, g in groupby(enumerate(data.index), lambda ix: ix[0] - ix[1]):
        tup = tuple(map(itemgetter(1), g))
        start, end = tup[0], tup[-1] + 1
        if (dets_over_trig[start:end + 1] < max_dets_num).all():
            trig_segs.append((start, end))
    return trig_segs


def statistics_table(trig_table):
    """
    Make a table with statistics regarding the percentage of detectors and ranges are triggered per event.
    :param trig_table: the table of events. Need 'trigs_det' column.
    :return: A table of percentage for each detector and range per: all events, solar flares and GRB.
    """
    # Define range columns of the output dataset.
    dct_perc = {'r0': {}, 'r1': {}, 'r2': {}, 'perc_rng': {}}
    # Loop for type of event
    for event_stat in ['all', 'GRB', 'SFLARE']:
        if event_stat == 'all':
            trig_table_tmp = trig_table
            n = trig_table_tmp.shape[0]
        else:
            trig_table_tmp = trig_table.loc[trig_table.catalog_triggers.str.contains(event_stat), :]
            n = trig_table_tmp.shape[0]
            if n == 0:
                continue
        # Loop for energy range
        for rng in ['r0', 'r1', 'r2']:
            # Percentage of events that range rng triggered
            dct_perc[rng]['perc_det'+'_'+event_stat] = np.round(
                sum([1. for i in trig_table_tmp.loc[:, 'trig_dets'] if '_' + rng in str(i)]) / n, 3)
            for det in ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb']:
                # Percentage of events that detector det was triggered
                dct_perc['perc_rng'][det+'_'+event_stat] = np.round(
                    sum([1. for i in trig_table_tmp.loc[:, 'trig_dets'] if det + '_' in str(i)]) / n, 3)
                # Percentage of events that detector det and range rng were triggered
                dct_perc[rng][det+'_'+event_stat] = np.round(
                    sum([1. for i in trig_table_tmp.loc[:, 'trig_dets'] if det+'_'+rng in str(i)]) / n, 3)
    dtf_out = pd.DataFrame(dct_perc)
    return dtf_out


def reduce_table(events_table, events, t_filt=300, bln_gbm=True):
    """
    Method that reduces trigger_table into an event_table. The list of events are reduced accordingly to this.
    :param events_table: the triggers table.
    :param events: the list of events (Segment objects).
    :param t_filt: From the first trigger events, t_filt is the time waited that join the triggers into an event.
    :param bln_gbm: If True GBM catalog are considered.
    :return: Table of events, list of events.
    """
    # Define the event table
    events_table_red = pd.DataFrame(columns=events_table.columns)
    for i in range(1, events_table.shape[0]):
        # If the next trigger is greater then t_filt add it as an event (except GBM catalog says it's the same)
        if events_table.loc[i, 'start_met'] - events_table.loc[i-1, 'start_met'] >= t_filt:
            # join only if GBM catalog says it's belonging to the same event
            if events_table.loc[i-1, 'catalog_triggers'] != '' and bln_gbm:
                if events_table.loc[i-1, 'catalog_triggers'] == events_table.loc[i, 'catalog_triggers']:
                    # Update in the current trigger event the previous start times and join the detectors triggered
                    events_table.loc[i, 'start_met'] = events_table.loc[i - 1, 'start_met']
                    events_table.loc[i, 'start_index'] = events_table.loc[i - 1, 'start_index']
                    events_table.loc[i, 'start_times'] = events_table.loc[i - 1, 'start_times']
                    events_table.loc[i, 'trig_ids'] = events_table.loc[i - 1, 'trig_ids']
                    events_table.loc[i, 'trig_dets'] = ' '.join(np.sort(list(
                        set(events_table.loc[i - 1, 'trig_dets'].split(' ')).union(
                            set(events_table.loc[i, 'trig_dets'].split(' '))))
                    ))
                else:
                    # Add trigger as event
                    events_table_red = events_table_red.append(events_table.loc[i - 1], ignore_index=True)
            else:
                # Add trigger as event
                events_table_red = events_table_red.append(events_table.loc[i - 1], ignore_index=True)
        # If the next trigger is greater then t_filt join only if GBM catalg says it's belonging to the same event
        else:
            if events_table.loc[i-1, 'catalog_triggers'] != '' and bln_gbm:
                # If the name of the GBM event coincide update the the current trigger
                if events_table.loc[i-1, 'catalog_triggers'] == events_table.loc[i, 'catalog_triggers']:
                    pass
                else:
                    # The name of the GBM trigger catalog don't coincide, add trigger as event and go to the next one
                    events_table_red = events_table_red.append(events_table.loc[i - 1], ignore_index=True)
                    continue
            # Update in the current trigger event the previous start times and join the detectors triggered
            events_table.loc[i, 'start_met'] = events_table.loc[i - 1, 'start_met']
            events_table.loc[i, 'start_index'] = events_table.loc[i - 1, 'start_index']
            events_table.loc[i, 'start_times'] = events_table.loc[i - 1, 'start_times']
            events_table.loc[i, 'trig_ids'] = events_table.loc[i - 1, 'trig_ids']
            events_table.loc[i, 'trig_dets'] = ' '.join(np.sort(list(
                set(events_table.loc[i - 1, 'trig_dets'].split(' ')).union(
                    set(events_table.loc[i, 'trig_dets'].split(' '))))
            ))
            if not bln_gbm:
                # Joining events could overlap events present in the GBM catalog
                events_table.iloc[i].catalog_triggers = ''.join([events_table.iloc[i].catalog_triggers,
                                                                 events_table.iloc[i-1].catalog_triggers])
    # Add the remaining trigger as event
    events_table_red = events_table_red.append(events_table.iloc[events_table.shape[0] - 1], ignore_index=True)
    events_table_red = events_table_red.reset_index()
    events_table_red = events_table_red.rename(columns={'index': 'event_ids'})

    # Update and delete segment events if the start time is not present
    lst_idx_ev_del = np.where(~pd.Series([e.start for e in events]).isin(events_table_red.start_index))[0]
    for idx_del in np.sort(lst_idx_ev_del)[::-1]:
        del events[idx_del]
    # Update the end time of the events
    for ev in events:
        ev.end = events_table_red.loc[events_table_red.start_index == ev.start, 'end_index'].values[0]

    return events_table_red, events


def tableize(events, threshold, sigma_r=None, sigma_type='SC_poisson', det_num=2):
    """
    Build the catalog table.
    :param events: events object.
    :param threshold: the theshold value express in sigma.
    :param sigma_r: dictionary that map the sigma of residuals per each detector_range.
    :param sigma_type: str, 'focus' uses the significance of focus, 'SC_poisson' uses the Standard Score with the
                        hypothesis that the process is poisson (mu=std_dev^2), 'SC_residual' uses the Standard Score but
                         with std_dev the one corresponding to det_rng in sigma_residual.
    :param det_num: int, if None are considered only the detectors triggered. If an integer k is specified takes the
                          highest k residuals detectors.
    :return: Catalog table with information about start/end time, significance and duration.
    """
    def stringify(lst):
        return ' '.join(str(e) for e in lst)

    trig_ids = [i for i in range(len(events))]
    start_ids = [s.start for s in events]
    start_mets = [s.fermi['met'][s.start] for s in events]
    start_times = [s.fermi['timestamp'][s.start] for s in events]
    start_times_offset = [s.fermi_offset['timestamp'][s.start_offset] for s in events]
    end_ids = [s.end for s in events]
    end_mets = [s.fermi['met'][s.end] for s in events]
    durations = [s.fermi['met'][s.end] - s.fermi['met'][s.start] for s in events]
    end_times = [s.fermi['timestamp'][s.end] for s in events]
    trig_dets = [stringify(list(s.focus[s.focus > threshold].any()[s.focus[s.focus > threshold].any() == True].keys()))
                 for s in events]
    quantile_cut = {'r0': [], 'r1': [], 'r2': []}
    # Calculate the significance per each event
    lst_sigma = {'r0': [], 'r1': [], 'r2': []}
    lst_det = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb']
    for i in range(0, len(events)):
        for rng in ['r0', 'r1', 'r2']:
            if rng in trig_dets[i]:
                if det_num is None:
                    lst_det_trig = [d_t + '_' + rng for d_t in lst_det if d_t in trig_dets[i]]
                else:
                    lst_det_trig = np.argsort((events[i].fermi_offset[[d_t + '_' + rng for d_t in lst_det]] -
                                               events[i].nn_offset[[d_t + '_' + rng for d_t in lst_det]]).sum(axis=0).
                                              sort_values(ascending=False))[0:det_num].index
                if sigma_type == 'focus':
                    sigma_tmp = (events[i].focus[lst_det_trig].sum(axis=1) / np.sqrt(len(lst_det_trig))).max().round(2)
                elif sigma_type == 'SC_poisson':
                    # ev_fermi_offset = events[i].fermi_offset.reset_index()
                    # ev_nn_offset = events[i].nn_offset.reset_index()
                    ev_fermi_offset = events[i].fermi_offset.reset_index().loc[:, lst_det_trig].sum(axis=1)
                    ev_nn_offset = events[i].nn_offset.reset_index().loc[:, lst_det_trig].sum(axis=1)
                    sigma_tmp_max = 0
                    qtl_max = 0
                    for qtl in np.arange(0, 21)/20:
                        index = np.where(ev_fermi_offset-ev_nn_offset >= np.quantile(ev_fermi_offset-ev_nn_offset, qtl,
                                                                                     interpolation='linear'))[0]
                        sigma_tmp = (ev_fermi_offset.loc[index] - ev_nn_offset.loc[index]).sum() / \
                                    np.sqrt(ev_nn_offset.loc[index].sum()).round(2)
                        if sigma_tmp > sigma_tmp_max:
                            sigma_tmp_max = sigma_tmp
                            qtl_max = qtl
                    sigma_tmp = sigma_tmp_max
                    quantile_cut[rng].append(qtl_max)
                elif sigma_type == 'SC_residual' and sigma_r is not None:
                    len_lst_det_trig = len(lst_det_trig)
                    len_event = events[i].fermi_offset.shape[0]
                    lst_sigma_tmp = [(events[i].fermi_offset[det_rng_tmp] - events[i].nn_offset[det_rng_tmp]).sum(axis=0) /
                                     (sigma_r[det_rng_tmp]*np.sqrt(len_event)) for det_rng_tmp in lst_det_trig]
                    sigma_tmp = np.sum(lst_sigma_tmp)/np.sqrt(len_lst_det_trig)
                else:
                    print('Warning: bad specification in sigma_type. Focus significance is used.')
                    sigma_tmp = (events[i].focus[lst_det_trig].sum(axis=1) / np.sqrt(len(lst_det_trig))).max().round(2)
                lst_sigma[rng].append(sigma_tmp)
            else:
                lst_sigma[rng].append(0)
                quantile_cut[rng].append(0)

    if len(quantile_cut['r0']) != len(events) or len(quantile_cut['r1']) != len(events) or len(quantile_cut['r2']) != len(events):
        quantile_cut = {'r0': [0]*len(events), 'r1': [0]*len(events), 'r2': [0]*len(events)}

    catalog_trigs = [stringify(s.get_catalog_triggers()) for s in events]
    trig_dic = {'trig_ids': trig_ids,
                'start_index': start_ids,
                'start_met': start_mets,
                'start_times': start_times,
                'start_times_offset': start_times_offset,
                'end_index': end_ids,
                'end_met': end_mets,
                'end_times': end_times,
                'catalog_triggers': catalog_trigs,
                'trig_dets': trig_dets,
                'sigma_r0': lst_sigma['r0'],
                'sigma_r1': lst_sigma['r1'],
                'sigma_r2': lst_sigma['r2'],
                'duration': durations,
                'qtl_cut_r0': quantile_cut['r0'],
                'qtl_cut_r1': quantile_cut['r1'],
                'qtl_cut_r2': quantile_cut['r2'],
                }
    out = pd.DataFrame(trig_dic)
    return out


def save_triggers_plots(events, threshold, folder):
    # TODO: parallelize this
    for n, event in enumerate(events):
        mask = event.focus[event.focus > threshold].any()
        triggered_dets = list(mask[mask == True].keys())
        untriggered_dets = list(set(get_keys()) ^ set(triggered_dets))

        ranges = list(set([int(t[-1]) for t in triggered_dets]))

        fig, ax = event.plot(triggered_dets, figsize=(7, 6), enlarge=100)
        for i in ranges:
            ax[i].axvspan(fermi.met.iloc[event.start_offset - 1], fermi.met.iloc[event.end], color='r', alpha=0.1)
        fig.suptitle('{} ({},{}) '.format(event.fermi.timestamp.iloc[0][:-7], event.start, event.end), x=0.34)

        try:
            fig.savefig(folder/'out{}.png'.format(n))
        except:
            print('Error saving image.')
        plt.close()

        fig, ax = event.plot(untriggered_dets, figsize=(7, 6), enlarge=100)
        fig.suptitle('{} ({},{}) '.format(event.fermi.timestamp.iloc[0][:-7], event.start, event.end), x=0.34)
        plt.savefig(folder/'out{}_untriggered.png'.format(n))
        plt.close()
    return True


def save_gbmbursts_plots(threshold, folder):
    for i, row in trigger_catalog.iterrows():
        if row['name'][:3] == 'GRB':
            try:
                t = GBMtrigger(row['name'])
            except Exception as e:
                print(e)
                print("check_against_gbmcatalogs: Possible trig or burst catalog not updated. "
                      "Use from connections.fermi_data_tools import df_trigger_catalog.")
                continue

            string = ""
            try:
                if t.did_focus_trigger(threshold, MIN_DET_NUMBER, MAX_DET_NUMBER):
                    string = "detected"
                else:
                    string = "undetected"
            except MissingDataError as e:
                string = "missed"

            fig, ax = t.plot()
            for i in range(3):
                ax[i].set_ylabel('range {}'.format(str(i)))
                ax[i].axvline(
                    t.get_metadata()['tTrig'],
                    linestyle = 'dashed',
                    label = 'GBM trigger time'
                )
            try:
                fig.savefig(folder/'{}_{}.png'.format(row['name'], string))
            except:
                print('Error saving image.')
            plt.close()
    return


def save_greenred_plot(detected, undetected, missing, folder, type_time=None, type_count=None):
    if detected:
        detected_names, detected_t90s, detected_fluences = list(zip(*detected))
    if undetected:
        undetected_names, undetected_t90s, undetected_fluences = list(zip(*undetected))
    if missing:
        missing_names, missing_t90s, missing_fluences = list(zip(*missing))

    with sns.plotting_context("talk"):
        fig, ax = plt.subplots(figsize=(15, 10))
        if detected:
            ax.scatter(
                detected_t90s,
                detected_fluences,
                c='lightgreen',
                label='detected',
            )
        if undetected:
            for i, (name, t90, fluence) in enumerate(undetected):
                ax.annotate(
                    name[3:],
                    (t90, fluence),
                    xytext=(-40, 10),
                    textcoords='offset points',
                    fontsize=8,
                )
                plt.scatter(
                    t90,
                    fluence,
                    color='red',
                    label="undetected" if i == 0 else "",
                )
        if missing:
            ax.scatter(missing_t90s, missing_fluences, c='lightgrey', label='missing')
        ax.axvspan(0.01, 4, color='grey', alpha=0.05)
        ax.semilogy()
        ax.semilogx()
        if type_time == 't90':
            ax.set_xlabel('$T_{90} \: [s]$')
        elif type_time == 't50':
            ax.set_xlabel('$T_{50} \: [s]$')
        else:
            print("Warning, time axis (x) not specified")
        if type_count == 'fluence':
            ax.set_ylabel('Fluence')
        elif type_count == 'flux':
            ax.set_ylabel(r'Flux $[erg \: cm^{-2} \: s^{-1}]$')
        else:
            print("Warning, counts axis (y) not specified")
        ax.legend()
        fig.savefig(folder / "greenred.png")
    return fig


def export_events_data(segments, filepath):
    for t in segments:
        t.export(filepath)
    return


def export_bursts_data(filepath):
    for i, row in trigger_catalog.iterrows():
        if row['name'][:3] == 'GRB':
            t = GBMtrigger(row['name'])
            t.export(filepath, filename = row['name'])
    return


def check_against_gbmcatalogs(threshold, type_time='t90', type_counts='flux'):
    """
    :param threshold:
    :param type_time: t90 or t50
    :param type_counts: fluence or flux
    :return:
    """
    detected = []
    undetected = []
    missing = []

    def info(t):
        t90 = t.get_metadata()[type_time]
        fluence = t.get_metadata()[type_counts]
        return (t.name, t90, fluence)

    for i, row in trigger_catalog.iterrows():
        if row['name'][:3] == 'GRB':
            try:
                t = GBMtrigger(row['name'])
            except Exception as e:
                print(e)
                print("check_against_gbmcatalogs: Possible trig or burst catalog not updated. "
                      "Use from connections.fermi_data_tools import df_trigger_catalog.")
                continue
            try:
                if t.did_focus_trigger(threshold, MIN_DET_NUMBER, MAX_DET_NUMBER):
                    detected.append(info(t))
                else:
                    undetected.append(info(t))
            except MissingDataError as e:
                print("did_focus_trigger: Possible trig or burst catalog not updated "
                      + row['name'] +
                      ". Use from connections.fermi_data_tools import df_trigger_catalog.")
                missing.append(info(t))

    return detected, undetected, missing


def crop_catalog(start, end):
    path_trig_cat = str(GBM_TRIG_DB)
    trigger_catalog = pd.read_csv(path_trig_cat).sort_values('met_time')

    START_MET = fermi.met.values[0]
    END_MET = fermi.met.values[-1]

    mask = (trigger_catalog.met_time > START_MET) & (trigger_catalog.met_end_time < END_MET)
    return trigger_catalog[mask]


def _create_connection():
    """
    Create a database connection to the SQLite database
    specified by db_file.

    Args:
        db_file: database file
    Return:
        Connection object or None.
    Raise:
        n/a
    """
    conn = sqlite3.connect(str(GBM_BURST_DB))
    return conn


def _grb_retrieve_fromdb(grb_id, verbose=False):
    """
    Fetches GRB metadata from GBM sqllite db file.

    Args:
        grb_id: a string containing the GRB id, e.g. something like '080916009'.

    Returns:
        If the fetching ended well the function will return two args:
        1. a tuple containing the GRB metadata and 2. a 0. If the function was
        unable to find a bcat for the requested GRB it will return
        'GRB_directory' and a 1.

    Raises:
        n/a
    """
    try:
        conn = _create_connection()
        cur = conn.cursor()
        triglist_unfetch = cur.execute("SELECT id,"
                                       " T90, "
                                       "T90_err, "
                                       " T50, "
                                       "T50_err, "
                                       "tStart, "
                                       "tStop, "
                                       "tTrigger, "
                                       "trig_det, "
                                       "fluence, "
                                       "flux "
                                       # "fluence_err, "
                                       # "fluenceb, "
                                       # "fluenceb_err, "
                                       # "pflx_int, "
                                       # "pflx, "
                                       # "pflx_err, "
                                       # "pflxb, "
                                       # "pflxb_err, "
                                       # "lobckint, "
                                       # "hibckint "
                                       "FROM GBM_GRB WHERE id =?", (grb_id,)).fetchall()[0]
        return triglist_unfetch, 0
    except Exception as e:
        print(e)


def query_db_about(grb_id):
    '''
    grb metadata
    :param grb_id:
    :return:
    '''
    try:
        metadata = _grb_retrieve_fromdb(grb_id)[0]
        strings = ('id', 't90', 't90_err', 't50', 't50_err', 'tStart', 'tStop', 'tTrig', 'trigDet', 'fluence', 'flux'
                   # , 'fluence_err',
                   # 'fluenceb', 'fluenceb_err', 'pflx_int', 'pflx', 'pflx_err', 'pflxb', 'pflxb_err', 'lobckint', 'hibckint'
                   )
        out = {key: val for (key, val) in list(zip(strings, metadata))}
        return out
    except Exception as e:
        print(e)


def _trigs_make():
    LEN = len(fermi)
    START_MET = fermi.met.values[0]
    END_MET = fermi.met.values[-1]

    out_dic = {}
    out_dic['met'] = np.array(fermi.met)
    out_dic['timestamp'] = np.array(fermi.timestamp)
    out_dic['id'] = pd.Series(['none' for i in range(LEN)], dtype=str)

    for index, name, trig_start_time, trig_end_time, dets_string, kind in list(zip(*(
            trigger_catalog.index,
            trigger_catalog.name,
            trigger_catalog.met_time,
            trigger_catalog.met_end_time,
            trigger_catalog.detector_mask,
            trigger_catalog.trigger_type))):

        if trig_start_time >= START_MET and trig_end_time <= END_MET:
            mask = (out_dic['met'] > trig_start_time) & (out_dic['met'] < trig_end_time)
            out_dic['id'][mask] = name
    return pd.DataFrame(out_dic)


#def fetch_triggers(threshold, min_dets_num=2, max_dets_num=8):
#    '''
#    returns a list of the triggers objects
#    from focus fildata.
#    :param threshold:
#    :return:
#    '''
#    out = {}
#    for i in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b']:
#        focus_ni = focus[get_keys(ns=[i], rs=['0', '1', '2'])]
#        out[i] = focus_ni[focus_ni > threshold].any(axis=1)
#    merged_ranges_df = pd.DataFrame(out)
#    dets_over_trig = merged_ranges_df[merged_ranges_df == True].count(axis=1)
#    data = dets_over_trig[dets_over_trig >= min_dets_num]
#
#    trig_segs = []
#    for k, g in groupby(enumerate(data.index), lambda ix: ix[0] - ix[1]):
#        tup = tuple(map(itemgetter(1), g))
#        start, end = tup[0], tup[-1]
#        if (dets_over_trig[start:end + 1] <= max_dets_num).all():
#            trig_segs.append(Segment(tup[0], tup[-1]))
#    return trig_segs



class GenericDisplay:
    def __str__(self):
        return '<%s: %s>' % (self.__class__.__name__, self.gather_attrs())

    def gather_attrs(self):
        attrs = '\n'
        for key in self.__dict__:
            if (lambda obj: isinstance(obj, list))(self.__dict__[key]) and (len(self.__dict__[key]) > 5):
                attrs += '\t{} = [{}..]\n'.format(key, ", ".join(["{}".format(val) for val in self.__dict__[key][:5]]))
            elif (lambda obj: isinstance(obj, pd.DataFrame))(self.__dict__[key]):
                # attrs += '\t{} = \n{}'.format(key,(self.__dict__[key]).head(1))
                pass
            else:
                attrs += '\t{} = {}\n'.format(key, self.__dict__[key])
        return attrs


class Segment(GenericDisplay):
    def __init__(self, start, end):
        '''
        :param start: start index
        :param end: end index
        '''
        self.start = start
        self.end = end
        offset_ev = offset.loc[self.start].fillna(0).min()
        self.start_offset = int(max(start + offset_ev + 1, 0))
        self.end_offset = end - 1
        self.fermi = fermi.loc[self.start:self.end][:]
        self.nn = nn.loc[self.start:self.end][:]
        self.focus = focus.loc[self.start:self.end][:]
        self.trigs = trigs.loc[self.start:self.end][:]
        self.fermi_offset = fermi.loc[self.start_offset:self.end_offset][:]
        self.nn_offset = nn.loc[self.start_offset:self.end_offset][:]
        self.focus_offset = focus.loc[self.start_offset:self.end_offset][:]
        self.trigs_offset = trigs.loc[self.start_offset:self.end_offset][:]

    def export(self, filepath, filename = 'timestamp'):
        if filename == 'timestamp':
            timestamp = self.fermi.timestamp.values[0].replace(' ', '_').replace(':', '-')[:19]
            label = timestamp
        else:
            label = filename

        self.fermi.to_csv(filepath / "{}_fermi.csv".format(label))
        self.focus.to_csv(filepath / "{}_focus.csv".format(label))
        self.nn.to_csv(filepath / "{}_nn.csv".format(label))
        self.fermi_offset.to_csv(filepath / "{}_fermi_offset.csv".format(label))
        self.focus_offset.to_csv(filepath / "{}_focus_offset.csv".format(label))
        self.nn_offset.to_csv(filepath / "{}_nn_offset.csv".format(label))


    def get_catalog_triggers(self):
        return set(self.trigs.id) - set(['none'])

    def enlarge(self, n):
        return Segment(self.start - n, self.end + n)

    def did_focus_trigger(self, threshold, min_dets_num, max_dets_num):
        '''
        check if you got focus trigger on dets list like ['n1_r0','na_r2']
        :param dets:
        :return:
        '''

        # if np.isnan(self.focus).all(axis=None):
        #     print("All trigger data for this event are nans!")
        #     raise MissingDataError('Missing Data.')
#
        triggers = fetch_triggers(self.focus, threshold, min_dets_num, max_dets_num)
        if triggers:
            return True
        return False

    def plot(self, det:  List[str], enlarge=0, figsize=None, legend=True, bln_ylim=True):
        '''
        matplotlib is messy and so is this thing. handle mindfully
        :param det:
        :param enlarge:
        :param figsize:
        :param legend:
        :param bln_ylim:
        :return:
        '''

        def get_indeces(seq):
            # order preserving
            seq = [s[1] for s in seq]
            checked = []
            for e in seq:
                if e not in checked:
                    checked.append(e)
            return checked

        det = sorted(det)  # sort detectors or get messed labels
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0., 0.8, 12))
        colors_dic = {d: color for d, color in list(zip([str(i) for i in range(10)] + ['a', 'b'], colors))}
        custom_lines = {i: Line2D([0], [0], color=colors_dic[i], lw=4) for i in
                        [str(i) for i in range(10)] + ['a', 'b']}

        # not using the segment but a copy of it
        p_seg = Segment(self.start - enlarge, self.end + enlarge)

        if not figsize:
            fig, ax = plt.subplots(3, 1, sharex=True, tight_layout=True)
        else:
            fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize, tight_layout=True)

        for d in det:
            range_label = int(d[-1])
            mets = p_seg.fermi.met.values
            fermi_data = p_seg.fermi[d]
            nn_data = p_seg.nn[d]

            ax[range_label].step(mets, fermi_data, color=colors_dic[d[1]], where='pre', label=d[:2])
            ax[range_label].plot(mets, nn_data, color=colors_dic[d[1]])

        for trig in p_seg.get_catalog_triggers():
            if trig != 'none':
                mask = (p_seg.trigs.id == trig)
                start, end = p_seg.fermi[mask].met.values[0], p_seg.fermi[mask].met.values[-1]

                for i in range(3):
                    ax[i].axvspan(start, end, color='black', alpha=0.1)
                    if i == 2:
                        ymin, ymax = ax[i].get_ylim()
                        ax[i].text(
                            start,
                            0, #ymin + (ymax - ymin) * 0.5 / 10,
                            trig,
                            fontsize=12)

        for i in range(3):
            if bln_ylim:
                ax[i].set_ylim(bottom=0, top=None)
            ax[i].set_ylabel('range {}'.format(str(i)))
        if legend and det:
            labels = ['n' + i for i in get_indeces(det)]
            lines = [custom_lines[i] for i in get_indeces(det)]
            if len(labels) > 0:
                fig.legend(lines, labels, framealpha=1., ncol=ceil(len(labels) / 4),
                           loc='upper right', bbox_to_anchor=(1.01, 1.005),
                           fancybox=True, shadow=True)
            else:
                print("Warning, no labels to plot in image legend.")
        try:
            fig.supylabel('count rate')
            fig.supxlabel('time [MET]')
        except Exception as e:
            print(e)
            print("Update matplotlib to 3.4 and Python to 3.7.")
        # fig.text(0.5, 0.04, 'ciao1', ha='center')
        # fig.text(0.04, 0.5, 'ciao2', va='center', rotation='vertical')
        # plt.xlabel('count rate')
        # plt.ylabel('time [MET]')
        return fig, ax


class GBMtrigger(Segment):
    def __init__(self, name):
        # get relevant trigger information
        catalog_row = trigger_catalog.loc[trigger_catalog.name == name].to_dict(orient='records')[0]
        self.name = catalog_row['name']
        self.met_time = catalog_row['met_time']
        self.met_end_time = catalog_row['met_end_time']
        self.trigger_type = catalog_row['trigger_type']

        # complete initializing associate seg
        start = bisect_right(fermi.met, self.met_time) - 1
        end = bisect_left(fermi.met, self.met_end_time)
        Segment.__init__(self, start, end)

    def triggered_detectors(self):
        def db_triggered_dets():
            '''
            fetch sql for trigs masks.
            this trigs are those used for spectroscopic analysis at burst db.
            :return:
            '''
            out = []
            for t in eval(self.get_metadata()['trigDet'].replace(" ", "")):
                if t[-2:] == '10':
                    out.append('na')
                elif t[-2:] == '11':
                    out.append('nb')
                else:
                    out.append('n' + t[-1:])
            return out

        catalog_row = trigger_catalog.loc[trigger_catalog.name == self.name].to_dict(orient='records')[0]
        dets_triggermask = [n + '_r' + str(i) for n in literal_eval(catalog_row['detector_mask']) for i in
                            ['0', '1', '2']]
        dets_db = get_keys(ns=[n[-1] for n in db_triggered_dets()])
        dets = sorted(list(set(dets_triggermask + dets_db)))
        return dets

    def get_seg(self):
        return Segment(self.start, self.end)

    def get_metadata(self):
        try:
            return query_db_about(self.name[3:])
        except:
            raise ValueError('Probably the thing you are asking for is not a GRB')

    def plot(self, *args, **kwargs):
        if args:
            return Segment.plot(self, *args, **kwargs)
        else:
            dets = self.triggered_detectors()
            return Segment.plot(self, dets, **kwargs)

    def did_focus_trigger(self, *args, **kwargs):
        '''
        overriding parent method with triggers.
        we check if we got focus triggers inside the trigger mask.
        no need to call for some dets to check!
        :param dets:
        :return:
        '''

        # we have a specialized method for GRBs
        if self.name[:3] == 'GRB':
            try:
                trigger_time = self.get_metadata()['tTrig']
                mask = (
                        (self.fermi.met < trigger_time + BINLENGTH)
                      & (self.fermi.met > trigger_time - BINLENGTH)
                )  # check if nans at grbs trig time
                if np.isnan(self.focus[mask]).any(axis=None) or self.focus[mask].empty:
                    raise MissingDataError('Missing Data.')
            except Exception as e:
                print(e)
                print("did_focus_trigger: Possible trig or burst catalog not updated "
                      + self.name +
                      ". Use from connections.fermi_data_tools import df_trigger_catalog.")
                raise MissingDataError('Missing Data.')

        if args:
            # non entra mai qua... TODO
            return Segment.did_focus_trigger(self, *args, **kwargs)
        else:
            dets = self.triggered_detectors()
            return Segment.did_focus_trigger(self, dets, **kwargs)
