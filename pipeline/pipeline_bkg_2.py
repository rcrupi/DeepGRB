# import packages
# from connections.fermi_data_tools import df_burst_catalog, df_trigger_catalog
# from models.download_bkg import download_spec
# from models.preprocess import build_table
# from models.model_nn import ModelNN
from models.trigger import run_trigger
from models.trigs import focus
from models.analyze import analyze
from models.utils.GBMutils import add_trig_gbm_to_frg
# from models.localize_event import localize
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

## Define range of energy
#erange = {'n': [(28, 50), (50, 300), (300, 500)],
#          'b': [(756, 5025), (5025, 50000)]}
start_month = "03-2019" # 11-2010 # 03-2019
end_month = "07-2019" # 02-2011 # 07-2019

for (start_month, end_month) in [
    ("03-2019", "07-2019"),
    ("01-2014", "03-2014"),
    ("11-2010", "02-2011"),
]:
    print("Running on data {} {}".format(start_month, end_month))
## 1 Download CSPEC and Poshist
#df_days = download_spec(start_month, end_month)
#
## 2 Elaborate CSPEC and Poshist -> to csv
#build_table(df_days, erange, bool_parallel=True, n_jobs=20)
#
## 3 Train NN
# nn = ModelNN(start_month, end_month)
#nn.prepare(bool_del_trig=False)
#nn.train(bool_train=True, bool_hyper=False, loss_type='mean', units=2048, epochs=64, lr=0.0005, bs=2048, do=0.05)
#nn.predict(time_to_del=0)  # set to 150 by default
# nn.plot(time_r=range(10000, 200000),  orbit_bin=1, det_rng='n6_r1')
#nn.plot(time_r=(-1000, 1000), time_iso='2011-01-01T12:00:00', det_rng='n6_r1') # range(2209866, 2219866)
# nn.plot(time_r=(-750, 750), time_iso='2019-05-01T19:03:42', det_rng='n0_r1')
# nn.plot(time_r=(-750, 750), time_iso='2019-05-01T19:03:42', det_rng='n3_r1')
# nn.plot(time_r=(-750, 750), time_iso='2019-05-01T19:03:42', det_rng='n6_r1')
# nn.plot(time_r=(-750, 750), time_iso='2019-05-01T19:03:42', det_rng='n7_r1')
#nn.plot(time_r=(-1000, 1000), time_iso='2019-04-04T13:08:00', det_rng='n6_r0') # range(2209866, 2219866)
#nn.explain(time_r=range(0, 10))

# 4 Run trigger (bkg, frg)
# add_trig_gbm_to_frg(start_month, end_month)
    trigger_algorithm = focus.set(mu_min = 1.2, t_max=25)
    run_trigger(start_month, end_month, trigger_algorithm)
    # bln_update_tables = False
    # if bln_update_tables:
    #     df_burst_catalog()
    #     df_trigger_catalog()
    analyze(start_month, end_month, threshold=3.0, type_time='t90', type_counts='flux')

    # 5 Localise events
    # localize(start_month, end_month)
pass
