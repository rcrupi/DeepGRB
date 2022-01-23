# import utils
from connections.utils.config import PATH_TO_SAVE, FOLD_PRED
# import standard packages
import matplotlib.pyplot as plt
import pandas as pd
# import trigger algorithm
import ruptures as rpt
from models.trigs.cusum import detect_cusum
from models.trigs.paramtrig import set_gbm


def run_trigger(start_month, end_month, type_trig='paramtrig'):

    # Load dataset of foreground and background
    df_ori = pd.read_csv(PATH_TO_SAVE + FOLD_PRED + "/" + 'frg_' + start_month + '_' + end_month + '.csv')
    y_pred = pd.read_csv(PATH_TO_SAVE + FOLD_PRED + "/" + 'bkg_' + start_month + '_' + end_month + '.csv')
    col_range = [i for i in y_pred.columns if i != 'timestamp' and i != 'met']
    # Calculate the residuals
    df_res = (df_ori[col_range] - y_pred[col_range]).astype('float32')
    df_res = df_res.dropna(axis=0)
    # df_res = df_res[df_res > 0].fillna(0)
    # breakpoints
    # TODO select trigger start and end for breakpoints

    i = 10000
    df_tmp = df_res.iloc[i:(i+10000), [0, 1, 2]]

    if type_trig == 'paramtrig':
        trigger = set_gbm(threshold=5.5, bg_len=4, fg_len=1,
                          hs=[1, 2, 2],
                          gs=[0, 0, 1])
        res = {}
        for col in df_tmp.columns:
            res[col] = trigger(df_tmp[col])

    elif type_trig == 'cusum':
        # detection Cusum
        df_tmp = df_tmp.rolling(32).sum()
        df_tmp = df_tmp.fillna(df_tmp.mean())
        ta, tai, taf, amp = detect_cusum(df_tmp.iloc[:, 1], threshold=20, drift=1., ending=True, show=True)
    elif type_trig == 'window':
        # detection Window ruptures
        sigma = 5.3

        n = df_tmp.shape[0]
        # https://centre-borelli.github.io/ruptures-docs/user-guide/detection/window/
        algo = rpt.Window(model="l2", width=40, min_size=2, jump=5).fit(df_tmp.values)
        result = algo.predict(epsilon=3 * n * sigma**2)

        # display
        bkps = result  # TODO put the real change points of trigger events
        rpt.display(df_tmp.values, bkps, result)
        plt.show()
