# import packages
from models.download_bkg import download_spec
from models.preprocess import build_table
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# Define range of energy
erange = {'n': [(28, 50), (50, 300), (300, 500)],
          'b': [(756, 5025), (5025, 50000)]}

# 1 Download CSPEC and Poshist
df_days = download_spec("01-2014", "04-2014")

# 2 Elaborate CSPEC and Poshist
build_table(df_days, erange)

# 3 Train NN
# fit_nn()
# predict_bkg()

# 4 Run trigger akg
# run_trigger()

# 5 Localise events
# localize()
# build_calalogue()