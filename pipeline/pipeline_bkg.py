# import packages
from models.download_bkg import download_spec
from models.preprocess import build_table
from models.model_nn import ModelNN
from models.trigger import run_trigger
from models.trigs import focus
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# Define range of energy
erange = {'n': [(28, 50), (50, 300), (300, 500)],
          'b': [(756, 5025), (5025, 50000)]}
start_month = "01-2019"
end_month = "06-2019"

# 1 Download CSPEC and Poshist
df_days = download_spec(start_month, end_month)

# 2 Elaborate CSPEC and Poshist -> to csv
build_table(df_days, erange, bool_parallel=True, n_jobs=20)

# 3 Train NN
nn = ModelNN(start_month, end_month)
nn.prepare(bool_del_trig=True)
nn.train(bool_train=True, bool_hyper=False, loss_type='median', units=2048, epochs=64, lr=0.0005, bs=2048)
nn.predict()
nn.plot(time_r=range(0, 1000), det_rng='n1_r1')
nn.explain(time_r=range(0, 10))

# 4 Run trigger (bkg, frg)
trigger_algorithm = focus.set(mu_min=1.05, t_max=50)
trigger_results = run_trigger(start_month, end_month, trigger_algorithm,
                              threshold=5., min_dets_num=2, max_dets_num=12)

# 5 Localise events
# localize()
# build_calalogue()
pass
