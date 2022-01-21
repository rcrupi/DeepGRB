# import packages
from models.download_bkg import download_spec
from models.preprocess import build_table
from models.model_nn import ModelNN
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# Define range of energy
erange = {'n': [(28, 50), (50, 300), (300, 500)],
          'b': [(756, 5025), (5025, 50000)]}
start_month = "01-2014"
end_month = "04-2014"

# 1 Download CSPEC and Poshist
df_days = download_spec(start_month, end_month)

# 2 Elaborate CSPEC and Poshist
build_table(df_days, erange, bool_parallel=True)

# 3 Train NN
nn = ModelNN(start_month, end_month)
nn.prepare(bool_del_trig=True)
nn.train(bool_train=False, loss_robust=False, units=400, epochs=128, lr=0.001, bs=2000)
# nn.predict()

# 4 Run trigger akg
# run_trigger()

# 5 Localise events
# localize()
# build_calalogue()
pass