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

    # Clean zeros in the datasets
    index_zeros = (fermi_data == 0).any(axis=1) | (nn_pred == 0).any(axis=1)
    if index_zeros.sum() > 0:
        print("Warning, 0 in foreground or background. They are set to None.")
        fermi_data.loc[index_zeros, nn_pred.columns] = None
        nn_pred.loc[index_zeros, nn_pred.columns] = None

    # Get det/range keys
    keys = get_keys()

    # Launch trigger
    print("Running trigger algorithm..")
    dct_res = {}
    dct_offset = {}
    for key in keys:
        print("Focus trigger... Elaborating key: ", key)
        out, out_offset = trigger(fermi_data[key], nn_pred[key])
        dct_res[key] = out
        dct_offset[key] = out_offset
    focus_res = pd.DataFrame(dct_res)
    focus_offset = pd.DataFrame(dct_offset)
    # TODO: [GD] csv name should report trigger parameters
    # Save data to csv
    focus_res.to_csv(PATH_TO_SAVE + FOLD_TRIG + '/trig_' + start_month + '_' + end_month + '.csv',
                     index=False, float_format='%.2f')
    focus_offset.to_csv(PATH_TO_SAVE + FOLD_TRIG + '/offset_' + start_month + '_' + end_month + '.csv',
                     index=False, float_format='%.2f')
    print("Done.")
    return focus_res