# import utils
from connections.utils.config import PATH_TO_SAVE, FOLD_PRED, FOLD_TRIG
from utils.keys import get_keys
# import standard packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_trigger(start_month, end_month, trigger):
    """
    manages trigger algorithms and stores results
    :param start_month: string, format mm-yyyy
    :param end_month: string, format mm-yyyy
    :param trigger: function
    :param threshold: float, in standard deviation units
    :param min_dets_num: int, min number of simultaneous trig dets
    :param max_dets_num: int, max number of simultaneous trig dets
    :return:
    """

    # Load dataset of foreground and background
    fermi_data = pd.read_csv(PATH_TO_SAVE + FOLD_PRED + "/" + 'frg_' + start_month + '_' + end_month + '.csv')
    nn_pred = pd.read_csv(PATH_TO_SAVE + FOLD_PRED + "/" + 'bkg_' + start_month + '_' + end_month + '.csv')

    # Get det/range keys
    keys = get_keys()

    # Launch trigger
    print("Running trigger algorithm..")
    focus_res = pd.DataFrame({key: trigger(fermi_data[key], nn_pred[key]) for key in keys})
    # TODO: [GD] csv name should report trigger parameters
    # Save data to csv
    focus_res.to_csv(PATH_TO_SAVE + FOLD_TRIG + '/trig_' + start_month + '_' + end_month + '.csv',
                     index=False, float_format='%.2f')
    print("Done.")
    return focus_res