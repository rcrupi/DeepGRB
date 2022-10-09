import os
import matplotlib.pyplot as plt
from gbm.data import TTE
from gbm.binning.unbinned import bin_by_time
from gbm.plot import Lightcurve

# read a tte file
tte_n6 = TTE.open("/beegfs/rcrupi/zzz_other/per_giovanni/glg_tte_n6_190404_13z_v00.fit.gz")
tte_n9 = TTE.open("/beegfs/rcrupi/zzz_other/per_giovanni/glg_tte_n9_190404_13z_v00.fit.gz")
tte_na = TTE.open("/beegfs/rcrupi/zzz_other/per_giovanni/glg_tte_na_190404_13z_v00.fit.gz")
# Datetime ISO
timestamp_event = '2019-04-04 13:08:15' #  '2019-04-04 13:14:29'
from gbm.time import Met
obj_met = Met(0)
met_event = obj_met.from_iso(timestamp_event.replace(" ", "T")).met
# Select only the interesting period
met_event_int = (met_event-100, met_event+100) #  (576076000.001468, 576076485)
grb190404b_n6 = tte_n6.slice_time([met_event_int])
grb190404b_n9 = tte_n9.slice_time([met_event_int])
grb190404b_na = tte_na.slice_time([met_event_int])
# bin in time (4.096s)
flt_bin_time = 1
er_int = (10, 20)
phaii_n6 = grb190404b_n6.to_phaii(bin_by_time, flt_bin_time, energy_range=er_int)
phaii_n9 = grb190404b_n9.to_phaii(bin_by_time, flt_bin_time, energy_range=er_int)
phaii_na = grb190404b_na.to_phaii(bin_by_time, flt_bin_time, energy_range=er_int)
# Plot lightcurve
lc_plot_n6 = Lightcurve(data=phaii_n6.to_lightcurve())
lc_plot_n9 = Lightcurve(data=phaii_n9.to_lightcurve())
lc_plot_na = Lightcurve(data=phaii_na.to_lightcurve())

plt.step(phaii_na.to_lightcurve().centroids ,phaii_na.to_lightcurve().counts + phaii_n6.to_lightcurve().counts + phaii_n9.to_lightcurve().counts)

from gbm.plot import SkyPlot
from gbm.data import PosHist
skyplot = SkyPlot()
poshist = PosHist.open("/beegfs/rcrupi/zzz_other/per_giovanni/glg_poshist_all_190404_v01.fit")
# plot the orientation of the detectors and Earth blockage at our time of interest
skyplot.add_poshist(poshist, trigtime=met_event)
