import os

# PATH_TO_SAVE = "C:/Users/peppe/Dropbox/Progetti/NN_FOCuS/DeepGRB-master/data/"
PATH_TO_SAVE = "D:/Dropbox/Progetti/NN_FOCuS/DeepGRB-master/data/"
FOLD_CSPEC_POS = "cspec"
FOLD_BKG = "bkg"
FOLD_PRED = "pred"
FOLD_NN = "nn_model"
FOLD_TRIG = "trig"
FOLD_PLOT = "plots"
db_path = os.path.dirname(__file__)
DB_PATH = db_path[0:(db_path.find('fermi_ml')+9)] + 'data/'
PATH_GRB_TABLE = PATH_TO_SAVE + "grb_classification/df_grb.csv"

# consider getting rid of the sql database for this csv version
GBMTDB_PATH = "C:/Users/peppe/Dropbox/Progetti/NN_FOCuS/DeepGRB-master/data/"
