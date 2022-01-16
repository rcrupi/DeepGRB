import pandas as pd
import os
from connections.utils.config import PATH_TO_SAVE
import matplotlib.pyplot as plt
db_path = os.path.dirname(__file__)
db_path = db_path[0:(db_path.find('fermi_ml')+9)] + 'data/'
data_path = PATH_TO_SAVE + "bkg/"
pred_path = PATH_TO_SAVE + "pred/"
df_ori = pd.read_csv(pred_path+'frg_210603.csv', index=False)
y_pred = pd.read_csv(pred_path+'bkg_210603.csv', index=False)

for i in range(3, 8):
    plt.figure()
    col_hist = 'n9_r2'
    month_hist = i
    df_ori_tmp = df_ori.loc[
        (df_ori[col_hist]<df_ori[col_hist].quantile(0.99))&
        (df_ori[col_hist]>df_ori[col_hist].quantile(0.001))&
        (df_ori['timestamp'].dt.month==month_hist)
    , col_hist]
    df_ori_tmp.hist(bins=40, alpha=0.5)
    y_pred_tmp = y_pred.loc[
        (y_pred[col_hist]<y_pred[col_hist].quantile(0.99))&
        (y_pred[col_hist]>y_pred[col_hist].quantile(0.001))&
        (df_ori['timestamp'].dt.month==month_hist)
    , col_hist]
    y_pred_tmp.hist(bins=40, alpha=0.5)
    plt.title('Month %2d, median true %.2f, median pred %.2f, diff %.2f'%(month_hist, y_pred_tmp.median(), df_ori_tmp.median(),
                                                               df_ori_tmp.median()-y_pred_tmp.median()))

diff_list = pd.DataFrame()
for i in range(3, 8):
    median_ori = df_ori.loc[(df_ori['timestamp'].dt.month == i), y_pred.columns].median()
    median_pred = y_pred.loc[(df_ori['timestamp'].dt.month == i), y_pred.columns].median()
    median_diff = median_ori-median_pred
    # (df_ori.loc[(df_ori['timestamp'].dt.month == i), y_pred.columns] - \
    #             y_pred.loc[(df_ori['timestamp'].dt.month == i), y_pred.columns]).median()
    diff_list = diff_list.append(median_diff, ignore_index=True)
plt.figure(figsize=(40,32))
for j in [0, 1, 2]:
    if j == 1:
        plt.title("median(original) - median(prediction)")
    plt.subplot(3, 1, j+1)
    diff_list[[i for i in diff_list.columns if '_r'+str(j) in i]].plot(ax=plt.gca())
    plt.xlabel("month")
    plt.ylabel("Frg - NN Bkg")
    if j == 1:
        plt.legend(loc="center left")
