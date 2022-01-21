import os

PATH_TO_SAVE = "/beegfs/rcrupi/"
FOLD_CSPEC_POS = "cspec"
FOLD_BKG = "bkg"
FOLD_PRED = "pred"
FOLD_NN = "nn_model"
db_path = os.path.dirname(__file__)
DB_PATH = db_path[0:(db_path.find('fermi_ml')+9)] + 'data/'
PATH_GRB_TABLE = PATH_TO_SAVE + "grb_classification/df_grb.csv"