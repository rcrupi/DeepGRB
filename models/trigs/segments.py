from math import ceil
from itertools import groupby
from operator import itemgetter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils.keys import get_keys

NULL_EVENT = ['0', 0] # TODO: modify in ['none'] after event column frg fix


class Segment:
    def __init__(self, start, end, fermi, nn, focus):
        """
        :param start: int, start index
        :param end: int, end index
        :param fermi: dataframe, observation
        :param nn: dataframe, background
        :param focus: dataframe, trigger
        """
        self._environment = (fermi, nn, focus)
        self.start = start
        self.end = end
        self.fermi = fermi.loc[self.start:self.end][:]
        self.nn = nn.loc[self.start:self.end][:]
        self.focus = focus.loc[self.start:self.end][:]
        self.trigs = self.fermi['event']

    def get_mets(self):
        """
        :return: tuple, start and end seg mets
        """
        return self.fermi['met'].iloc[0], self.fermi['met'].iloc[-1]

    def get_timestamps(self):
        """
        :return: tuple, start and end seg timestamps
        """
        return self.fermi.timestamp.iloc[0][:-7], self.fermi.timestamp.iloc[-1][:-7]

    def get_catalog_triggers(self):
        # put all unique trigger ids in a set and intersect with null
        return set(self.trigs.unique()) - set(NULL_EVENT)

    def export(self, save_path):
        """
        :param save_path: string, where save
        """
        self.focus.to_csv(save_path, index=False)
        self.fermi.to_csv(save_path, index=False)
        self.nn.to_csv(save_path, index=False)
        return True

    def plot(self, det, enlarge=0, figsize=None, legend=True):
        """
        matplotlib is messy and so is this thing. handle mindfully
        :param det:
        :param enlarge:
        :param figsize:
        :param legend:
        :return:
        """

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
        custom_lines = {i: Line2D([0], [0], color=colors_dic[i], lw=4)
                        for i in [str(i) for i in range(10)] + ['a', 'b']}

        # not using the segment but a copy of it
        p_seg = Segment(self.start - enlarge, self.end + enlarge, *self._environment)

        if not figsize:
            fig, ax = plt.subplots(3, 1, sharex=True, tight_layout=True)
        else:
            fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize, tight_layout=True)

        if isinstance(det, str):
            raise ValueError("detectors should be in a list like e.g. ['n0_r0'] or ['n2_r1','n3_r0']")

        elif isinstance(det, list):
            for d in det:
                range_label = int(d[-1])
                mets = p_seg.fermi.met.values
                fermi_data = p_seg.fermi[d]
                nn_data = p_seg.nn[d]

                ax[range_label].step(mets, fermi_data, color=colors_dic[d[1]], where='pre', label=d[:2])
                ax[range_label].plot(mets, nn_data, color=colors_dic[d[1]])

        for trig in p_seg.get_catalog_triggers():
            if trig not in NULL_EVENT:
                mask = (p_seg.trigs == trig)
                start, end = p_seg.fermi[mask].met.values[0] , p_seg.fermi[mask].met.values[-1]

                for i in range(3):
                    ax[i].axvspan(start, end, color = 'black', alpha = 0.1)
                    if i == 2:
                        ymin, ymax = ax[i].get_ylim()
                        ax[i].text(start,ymin + (ymax - ymin)*0.5/10,trig, fontsize = 12)

        for i in range(3):
            ax[i].set_ylabel('range {}'.format(str(i)))
        if legend:
            labels = ['n' + i for i in get_indeces(det)]
            lines = [custom_lines[i] for i in get_indeces(det)]
            fig.legend(lines, labels, framealpha=1., ncol=ceil(len(labels)/4),
                       loc='upper right', bbox_to_anchor=(1.01, 1.005),
                       fancybox=True, shadow=True)
        fig.supylabel('count rate')
        fig.supxlabel('time [MET]')
        start_tstamp, _ = self.get_timestamps()
        fig.suptitle('{} ({},{}) '.format(start_tstamp, self.start, self.end), x=0.3)
        return fig, ax


def get_dets(seg, threshold):
    """
    :param seg: a segment object
    :param threshold: float, standard deviations units
    :return: a tuple containing a list of dets which went above threshold
             and a list of dets which did not get above threshold.
    """
    triggered_dets = list(seg.focus[seg.focus > threshold].any()
                          [seg.focus[seg.focus > threshold].any()].keys())
    untriggered_dets = list(set(get_keys()) ^ set(triggered_dets))
    return triggered_dets, untriggered_dets


def tplot(seg, threshold, **kwargs):
    """
    :param seg: a segment object
    :param threshold: float, in standard deviation units
    :param kwargs: see segment.plot
    :return: a tuple of figure and axes
    """
    trig_det, _ = get_dets(seg, threshold)
    ranges = list(set([int(t[-1]) for t in trig_det]))
    if 'enlarge' not in kwargs.keys():
        fig, axes = seg.plot(trig_det, enlarge=100, **kwargs)
    else:
        fig, axes = seg.plot(trig_det, **kwargs)

    for i in ranges:
        start_met, end_met = seg.get_mets()
        axes[i].axvspan(start_met, end_met, color='r', alpha=0.1)
    return fig, axes


def fetch_triggers(fermi, nn, focus, threshold=5.5, min_dets_num=2, max_dets_num=8):
    """
    take a trigger dataframe and returns a collection of segments.
    this function specifies your trigger condition.

    :param fermi: dataframe
    :param nn: dataframe
    :param focus: dataframe
    :param threshold: float, standard devs
    :param min_dets_num: int
    :param max_dets_num: int
    :return: a list of segs
    """
    temp = {}
    for i in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b']:
        focus_ni = focus[get_keys(ns=[i], rs=['0', '1', '2'])]
        temp[i] = focus_ni[focus_ni > threshold].any(axis=1)
    merged_ranges_df = pd.DataFrame(temp)
    dets_over_trig = merged_ranges_df[merged_ranges_df].count(axis=1)
    data = dets_over_trig[dets_over_trig >= min_dets_num]

    out = []
    for k, g in groupby(enumerate(data.index), lambda ix: ix[0] - ix[1]):
        tup = tuple(map(itemgetter(1), g))
        start, end = tup[0], tup[-1]
        if (dets_over_trig[start:end + 1] <= max_dets_num).all():
            out.append(Segment(tup[0], tup[-1], fermi, nn, focus))
    return out


def compile_catalogue(trig_collection, threshold):
    """
    :param trig_collection: a list of segments
    :param threshold: float, standard deviation units
    :return: a dictionary
    """
    trig_ids = [i for i, _ in enumerate(trig_collection)]
    start_ids = [t.start for t in trig_collection]
    end_ids = [t.end for t in trig_collection]
    start_mets, end_mets = list(zip(*[t.get_mets() for t in trig_collection]))
    start_times, end_times = list(zip(*[t.get_timestamps() for t in trig_collection]))
    trig_dets, _ = list(zip(*[get_dets(t, threshold) for t in trig_collection]))

    out = {'trig_ids': trig_ids,
           'start_index': start_ids,
           'start_met': start_mets,
           'start_times': start_times,
           'end_index': end_ids,
           'end_met': end_mets,
           'end_times': end_times,
           'trig_dets': trig_dets}
    return out
