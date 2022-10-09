from sklearn.metrics import mean_absolute_error as MAE, median_absolute_error as MeAE
import pandas as pd
from sklearn.model_selection import train_test_split
import logging


def report(y, pred):
  # Train and test split, read columns
  if 'met' in y.columns or 'timestamp' in y.columns:
    del y['met'], y['timestamp']
  if 'met' in pred.columns or 'timestamp' in pred.columns:
    del pred['met'], pred['timestamp']


  y = y.dropna()
  pred = pred.dropna().values
  pred_train, pred_test, y_train, y_test = train_test_split(pred, y, test_size=0.25, random_state=0, shuffle=True)

  col_range = y_train.columns

  # Median Absolute Error Computation
  mae = MAE(y_train, pred_train)
  logging.warning("MAE train : % f" % (mae))
  mae = MAE(y_test, pred_test)
  logging.warning("MAE test : % f" % (mae))
  diff_mean = (y_test - pred_test).median().mean()
  diff_std = (y_test - pred_test).median().std()
  logging.warning("diff test - pred : % f +- % f" % (diff_mean, diff_std))

  # Compute MAE per each detector and range
  idx = 0
  text_mae = ""
  dct_res = {
      'MAE train' : {},
      'MAE test' : {},
      'diff test' : {},
      'MeAE train' : {},
      'MeAE test' : {},
      'med diff test' : {}
  }
  for i in col_range:
      mae_tr = MAE(y_train.iloc[:, idx], pred_train[:, idx])
      mae_te = MAE(y_test.iloc[:, idx], pred_test[:, idx])
      diff_i = (y_test.iloc[:, idx] - pred_test[:, idx]).mean()
      meae_tr = MeAE(y_train.iloc[:, idx], pred_train[:, idx])
      meae_te = MeAE(y_test.iloc[:, idx], pred_test[:, idx])
      diff_i_m = (y_test.iloc[:, idx] - pred_test[:, idx]).median()
      text_tr = "MAE train of " + i + " : %0.3f" % (mae_tr)
      text_te = "MAE test of " + i + " : %0.3f" % (mae_te)
      test_diff_i = "diff test - pred " + i + " : %0.3f" % (diff_i)
      text_tr_m = "MeAE train of " + i + " : %0.3f" % (meae_tr)
      text_te_m = "MeAE test of " + i + " : %0.3f" % (meae_te)
      test_diff_i_m = "med diff test - pred " + i + " : %0.3f" % (diff_i_m)
      logging.warning(text_tr + '    ' + text_te + '    ' + test_diff_i + '    ' +\
                  text_tr_m + '    ' + text_te_m + '    ' + test_diff_i_m)
      text_mae += text_tr + '    ' + text_te + '    ' + test_diff_i + '    ' + \
                  text_tr_m + '    ' + text_te_m + '    ' + test_diff_i_m + '\n'
      idx = idx + 1
      # dictonary result
      dct_res['MAE train'][i] = mae_tr
      dct_res['MAE test'][i] = mae_te
      dct_res['diff test'][i] = diff_i
      dct_res['MeAE train'][i] = meae_tr
      dct_res['MeAE test'][i] = meae_te
      dct_res['med diff test'][i] = diff_i_m

  # Report performance
  dtf_res = pd.DataFrame(dct_res).T
  dtf_res
  for i in ['r0', 'r1', 'r2']:
    print("Energy range", i)
    print(dtf_res.loc[:, [j for j in dtf_res.columns if i in j]].mean(axis=1))
    print(dtf_res.loc[:, [j for j in dtf_res.columns if i in j]].std(axis=1))
    print("\n\n")


y = pd.read_csv('/beegfs/rcrupi/pred/frg_01-2014_01-2015.csv')
pred = pd.read_csv('/beegfs/rcrupi/pred/bkg_01-2014_01-2015.csv')
report(y, pred)
