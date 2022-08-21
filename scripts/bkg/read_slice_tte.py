import os
import matplotlib.pyplot as plt
from gbm.data import TTE
from gbm.binning.unbinned import bin_by_time
from gbm.plot import Lightcurve

# read a tte file
tte_n6 = TTE.open("/beegfs/rcrupi/zzz_other/per_giovanni/glg_tte_n6_190404_13z_v00.fit.gz")
tte_n9 = TTE.open("/beegfs/rcrupi/zzz_other/per_giovanni/glg_tte_n9_190404_13z_v00.fit.gz")
tte_na = TTE.open("/beegfs/rcrupi/zzz_other/per_giovanni/glg_tte_na_190404_13z_v00.fit.gz")
# Select only the interesting period
grb190404b_n6 = tte_n6.slice_time([(576075485.001468, 576076485)])
grb190404b_n9 = tte_n9.slice_time([(576075485.001468, 576076485)])
grb190404b_na = tte_na.slice_time([(576075485.001468, 576076485)])
# bin in time (4.096s)
flt_bin_time = 4.096
phaii_n6 = grb190404b_n6.to_phaii(bin_by_time, flt_bin_time, energy_range=(28, 50))
phaii_n9 = grb190404b_n9.to_phaii(bin_by_time, flt_bin_time, energy_range=(28, 50))
phaii_na = grb190404b_na.to_phaii(bin_by_time, flt_bin_time, energy_range=(28, 50))
# Plot lightcurve
lc_plot_n6 = Lightcurve(data=phaii_n6.to_lightcurve())
lc_plot_n9 = Lightcurve(data=phaii_n9.to_lightcurve())
lc_plot_na = Lightcurve(data=phaii_na.to_lightcurve())
