import os
import pandas as pd
import matplotlib.pyplot as plt
# Fermi SkyPlot
from gbm.data import PosHist
from gbm.plot import SkyPlot
from gbm.finder import ContinuousFtp
# bkg = pd.read_csv('/beegfs/rcrupi/pred/bkg_09-2009_12-2009.csv')
# frg = pd.read_csv('/beegfs/rcrupi/pred/frg_09-2009_12-2009.csv')
# bkg = pd.read_csv('/beegfs/rcrupi/pred/bkg_01-2015_03-2015.csv')
# frg = pd.read_csv('/beegfs/rcrupi/pred/frg_01-2015_03-2015.csv')
bkg = pd.read_csv('/beegfs/rcrupi/pred/bkg_01-2019_07-2019_current.csv')
frg = pd.read_csv('/beegfs/rcrupi/pred/frg_01-2019_07-2019_current.csv')

# bkg = pd.read_csv('/home/rcrupi/Desktop/rcrupi/pred/bkg_03-2019_07-2019_current.csv')
# frg = pd.read_csv('/home/rcrupi/Desktop/rcrupi/pred/frg_03-2019_07-2019_current.csv')
# bkg['met'] = frg['met']
# # Compute the MAE for the timestep outside the events
# ev_tab = pd.read_csv("/home/rcrupi/Desktop/rcrupi/results/frg_03-2019_07-2019/events_table.csv")
# si = ev_tab.start_index[4]-10
# ei = ev_tab.end_index[4]
# plt.plot(bkg.loc[(bkg.index > si)&(bkg.index < ei), 'n0_r1'], 'x')
# plt.plot(frg.loc[(frg.index > si)&(frg.index < ei), 'n0_r1'], 'x-')
#
# idx_ev = []
# for idx, row in ev_tab.iterrows():
#     si = row.start_index
#     ei = row.end_index
#     idx_ev = idx_ev + list(range(si, ei, 1))
#
# bkg_filt = bkg.loc[~bkg.index.isin(idx_ev), :]
# frg_filt = frg.loc[~frg.index.isin(idx_ev), :]
# bkg_filt = bkg_filt.loc[(bkg_filt.notna().any(axis=1)) & (frg_filt.notna().any(axis=1)), :]
# frg_filt = frg_filt.loc[(bkg_filt.notna().any(axis=1)) & (frg_filt.notna().any(axis=1)), :]
#
# from sklearn.metrics import mean_absolute_error, median_absolute_error
# mean_absolute_error(bkg_filt, frg_filt.loc[:, bkg_filt.columns])
# median_absolute_error(bkg_filt, frg_filt.loc[:, bkg_filt.columns])


# start_time = pd.to_datetime('2009-10-24 07:55:00')
# end_time = pd.to_datetime('2009-10-24 10:15:00')
# start_time = pd.to_datetime('2015-01-26 00:00:01')
# end_time = pd.to_datetime('2015-01-26 23:59:59')
# start_time = pd.to_datetime('2019-02-08 21:34:00')
# end_time = pd.to_datetime('2019-02-08 21:54:00')
start_time = pd.to_datetime('2019-02-15 06:35:58') # '2019-02-15 06:37:58'
end_time = pd.to_datetime('2019-02-15 07:01:00')

# Filter event time
bkg_e = bkg.loc[
    (pd.to_datetime(frg['timestamp']) >= start_time) &
(pd.to_datetime(frg['timestamp']) <= end_time), :]
frg_e = frg.loc[
    (pd.to_datetime(frg['timestamp']) >= start_time) &
(pd.to_datetime(frg['timestamp']) <= end_time), :]

for col in ['n4_r1', #'n6_r0', 'n8_r0',
            #'n0_r1', 'n6_r1', 'n8_r1',
            #'n0_r2', 'n6_r2', 'n8_r2'
            ]: # bkg.columns[0:36]:
    fig, axs = plt.subplots(3, 1, sharex=True)
    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)
    fig.suptitle(col + " " + str(start_time))

    # Plot each graph, and manually set the y tick values
    axs[0].plot(frg_e['met'], frg_e[col], 'kx')
    axs[0].plot(bkg_e['met'], bkg_e[col], 'r-')

    # axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
    # axs[0].set_ylim(-1, 1)
    axs[0].set_title('foreground and background')
    axs[0].set_xlabel('met')
    axs[0].set_ylabel('Count Rate')
    axs[0].set_ylim(320, None)

    axs[1].plot(bkg_e['met'], frg_e[col]-bkg_e[col], 'kx')
    axs[1].plot(bkg_e['met'].fillna(method='ffill'), bkg_e['met'].fillna(0)*0, 'k-')
    # axs[1].set_yticks(np.arange(0.1, 1.0, 0.2))
    # axs[1].set_ylim(0, 1)
    axs[1].set_xlabel('met')
    axs[1].set_ylabel('Residuals')
    axs[1].set_ylim(-20, None)

    axs[2].plot(bkg_e['met'], (frg_e[col] - bkg_e[col])/bkg_e[col]*100, 'kx')
    axs[2].plot(bkg_e['met'].fillna(method='ffill'), bkg_e['met'].fillna(0) * 0, 'k-')
    # axs[1].set_yticks(np.arange(0.1, 1.0, 0.2))
    # axs[1].set_ylim(0, 1)
    axs[2].set_xlabel('met')
    axs[2].set_ylabel('Residuals %')
    axs[2].set_ylim(-10, None)
plt.savefig("/home/rcrupi/Downloads/saa_passage.png")

met_event = int(bkg_e['met'].mean())
cont_finder = ContinuousFtp(met=met_event)
cont_finder.get_poshist('tmp')
# open a poshist file
str_path_poshist = "/home/rcrupi/PycharmProjects/fermi_ml/scripts/bkg/tmp/"
poshist = PosHist.open(str_path_poshist + os.listdir(str_path_poshist)[0])
print(os.listdir(str_path_poshist)[0])
os.remove(str_path_poshist + os.listdir(str_path_poshist)[0])
# initialize plot
skyplot = SkyPlot()
# plot the orientation of the detectors and Earth blockage at our time of interest
skyplot.add_poshist(poshist, trigtime=met_event)

from gbm.plot import EarthPlot

# initialize plot
earthplot = EarthPlot()

# let's show the orbital path for +/-1000 s around our t0
earthplot.add_poshist(poshist, trigtime=met_event, time_range=(met_event-500.0, met_event+500.0))

pass