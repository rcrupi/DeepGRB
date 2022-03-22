import os
import pandas as pd
import matplotlib.pyplot as plt
# Fermi SkyPlot
from gbm.data import PosHist
from gbm.plot import SkyPlot
from gbm.finder import ContinuousFtp
bkg = pd.read_csv('/beegfs/rcrupi/pred/bkg_09-2009_12-2009.csv')
frg = pd.read_csv('/beegfs/rcrupi/pred/frg_09-2009_12-2009.csv')
# bkg = pd.read_csv('/beegfs/rcrupi/pred/bkg_01-2015_03-2015.csv')
# frg = pd.read_csv('/beegfs/rcrupi/pred/frg_01-2015_03-2015.csv')
# bkg = pd.read_csv('/beegfs/rcrupi/pred/bkg_01-2019_06-2019.csv')
# frg = pd.read_csv('/beegfs/rcrupi/pred/frg_01-2019_06-2019.csv')

start_time = pd.to_datetime('2009-10-24 07:55:00')
end_time = pd.to_datetime('2009-10-24 10:15:00')
# start_time = pd.to_datetime('2015-01-26 00:00:01')
# end_time = pd.to_datetime('2015-01-26 23:59:59')
# start_time = pd.to_datetime('2019-05-21 00:01:00')
# end_time = pd.to_datetime('2019-05-21 23:59:00')
# Filter event time
bkg_e = bkg.loc[
    (pd.to_datetime(frg['timestamp']) >= start_time) &
(pd.to_datetime(frg['timestamp']) <= end_time), :]
frg_e = frg.loc[
    (pd.to_datetime(frg['timestamp']) >= start_time) &
(pd.to_datetime(frg['timestamp']) <= end_time), :]

for col in ['n0_r0', 'n6_r0', 'n8_r0',
            'n0_r1', 'n6_r1', 'n8_r1',
            'n0_r2', 'n6_r2', 'n8_r2']: # bkg.columns[0:36]:
    fig, axs = plt.subplots(2, 1, sharex=True)
    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)
    fig.suptitle(col + " " + str(start_time))

    # Plot each graph, and manually set the y tick values
    axs[0].plot(frg_e['met'], frg_e[col], 'k-.')
    axs[0].plot(bkg_e['met'], bkg_e[col], 'r-')

    # axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
    # axs[0].set_ylim(-1, 1)
    axs[0].set_title('frg and bkg')
    axs[0].set_xlabel('met')
    axs[0].set_ylabel('Count Rate')

    axs[1].plot(bkg_e['met'], frg_e[col]-bkg_e[col], 'k-.')
    axs[1].plot(bkg_e['met'].fillna(method='ffill'), bkg_e['met'].fillna(0)*0, 'k-')
    # axs[1].set_yticks(np.arange(0.1, 1.0, 0.2))
    # axs[1].set_ylim(0, 1)
    axs[1].set_xlabel('met')
    axs[1].set_ylabel('Residuals')

met_event = bkg_e['met'].mean()
cont_finder = ContinuousFtp(met=met_event)
cont_finder.get_poshist('tmp')
# open a poshist file
poshist = PosHist.open("/home/rcrupi/PycharmProjects/fermi_ml/tmp/"
                       + os.listdir("/home/rcrupi/PycharmProjects/fermi_ml/tmp/")[0])
os.remove("/home/rcrupi/PycharmProjects/fermi_ml/tmp/"
                       + os.listdir("/home/rcrupi/PycharmProjects/fermi_ml/tmp/")[0])
# initialize plot
skyplot = SkyPlot()
# plot the orientation of the detectors and Earth blockage at our time of interest
skyplot.add_poshist(poshist, trigtime=met_event)

