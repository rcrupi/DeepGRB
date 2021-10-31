import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from connections.fermi_data_tools import df_trigger_catalog
from connections.utils.config import PATH_TO_SAVE
from sqlalchemy import create_engine
from datetime import datetime
# Preprocess
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error as MAE
from sklearn.preprocessing import StandardScaler
# NN
import tensorflow as tf
#from keras import backend as K
#from keras.models import Sequential
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

db_path = os.path.dirname(__file__)
db_path = db_path[0:(db_path.find('fermi_ml')+9)] + 'data/'
# df_trigger_catalog(db_path)
data_path = PATH_TO_SAVE + "bkg/"
pred_path = PATH_TO_SAVE + "pred/"
BOOL_TRAIN = False

# List csv files
list_csv = np.sort([i for i in os.listdir(data_path) if '.csv' in i and
                    (i >= '101101.csv' and i <= '110301.csv') ])
df_data = pd.DataFrame()

for csv_tmp in list_csv:
  # TODO add shift lag columns
  df_tmp = pd.read_csv(data_path + csv_tmp)
  df_data = df_data.append(df_tmp, ignore_index=True)

col_met = ['met']
col_range = ['n0_r0', 'n0_r1', 'n0_r2', 'n1_r0', 'n1_r1', 'n1_r2', 'n2_r0',
       'n2_r1', 'n2_r2', 'n3_r0', 'n3_r1', 'n3_r2', 'n4_r0', 'n4_r1', 'n4_r2',
       'n5_r0', 'n5_r1', 'n5_r2', 'n6_r0', 'n6_r1', 'n6_r2', 'n7_r0', 'n7_r1',
       'n7_r2', 'n8_r0', 'n8_r1', 'n8_r2', 'n9_r0', 'n9_r1', 'n9_r2', 'na_r0',
       'na_r1', 'na_r2', 'nb_r0', 'nb_r1', 'nb_r2']
col_sat_pos = ['pos_x', 'pos_y', 'pos_z', 'a', 'b', 'c', 'd', 'lat', 'lon', 'alt',
       'vx', 'vy', 'vz', 'w1', 'w2', 'w3', 'sun_vis', 'sun_ra', 'sun_dec',
       'earth_r', 'earth_ra', 'earth_dec', 'saa', 'l']
col_det_pos = ['n0_ra', 'n0_dec', 'n0_vis',
       'n1_ra', 'n1_dec', 'n1_vis', 'n2_ra', 'n2_dec', 'n2_vis', 'n3_ra',
       'n3_dec', 'n3_vis', 'n4_ra', 'n4_dec', 'n4_vis', 'n5_ra', 'n5_dec',
       'n5_vis', 'n6_ra', 'n6_dec', 'n6_vis', 'n7_ra', 'n7_dec', 'n7_vis',
       'n8_ra', 'n8_dec', 'n8_vis', 'n9_ra', 'n9_dec', 'n9_vis', 'na_ra',
       'na_dec', 'na_vis', 'nb_ra', 'nb_dec', 'nb_vis']
# Filter data within saa
df_data = df_data.loc[df_data['saa'] == 0, col_met+col_range+col_sat_pos+col_det_pos].reset_index(drop=True)
# TODO Filter zero counts
df_data.head()

# Add trigger event
engine = create_engine('sqlite:////'+db_path+'GBMdatabase.db')
# TODO update this table
gbm_tri = pd.read_sql_table('GBM_TRI', con=engine)
index_date = df_data['met'] >= 0
for index, row in gbm_tri.iterrows():
    if (index-1) % 1000 == 0:
        print('index reached: ', index)
    index_date = ((df_data['met'] <= row['met_time']) | (df_data['met'] >= row['met_end_time'])) & index_date

#### NN
# Load the data
# TODO trigger events
y = df_data.loc[index_date, col_range].astype('float32')
col_selected = col_sat_pos + col_det_pos
X = df_data.loc[index_date, col_selected].astype('float32')

# Splitting
X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.25, random_state=0, shuffle=True # False
   )

# Scale
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X = scaler.transform(X)


if BOOL_TRAIN:
    nn_input = keras.Input(
        shape=(X_train.shape[1],)
    )
    model_1 = Dense(2000*2, activation='relu')(nn_input)
    model_1 = Dropout(0.2)(model_1)
    nn_r = Dense(2000*2, activation='relu')(model_1)
    # nn_r = Dropout(0.2)(nn_r)
    # nn_r = Dense(2000, activation='relu')(nn_r)
    nn_r = Dropout(0.2)(nn_r)
    nn_r = Dense(1000*2, activation='relu')(nn_r)
    outputs = Dense(len(col_range), activation='relu')(nn_r)

    nn_r = keras.Model(
        inputs=[nn_input],
        outputs=outputs
    )

    opt = keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.7, beta_2=0.99, epsilon=1e-07)
    # import tensorflow.keras.losses as losses
    # loss_mae_none = losses.MeanAbsoluteError(reduction=losses.Reduction.NONE)
    # def loss_mae(y_true, y_pred):
    #     a = tf.math.reduce_max(loss_mae_none(y_true, y_pred)) # axis=0
    #     return a
    loss_mae = 'mae'

    nn_r.compile(loss=loss_mae, loss_weights=1, optimizer=opt)
    # nn_r.compile(loss=['mae']*36, loss_weights=[0.1,1,1]*12, optimizer=opt)

    # Fitting the model
    name_model = 'best_model_all_'+str(datetime.today())[0:10]+'.h5'
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=0.01, patience=8)
    mc = ModelCheckpoint(db_path + name_model, monitor='val_loss', mode='min', verbose=0, save_best_only=True)
    history = nn_r.fit(X_train, y_train, epochs=512, batch_size=2000,
                        validation_split=0.3, callbacks=[mc]) # es
    nn_r.save(db_path + name_model)

    # Predict the model
    pred_train = nn_r.predict(X_train)
    pred_test = nn_r.predict(X_test)

    # RMSE Computation
    mae = MAE(y_train, pred_train)
    print("MAE train : % f" %(mae))
    mae = MAE(y_test, pred_test)
    print("MAE test : % f" %(mae))

    idx = 0
    for i in col_range:
      mae_tr = MAE(y_train.iloc[:, idx], pred_train[:, idx])
      mae_te = MAE(y_test.iloc[:, idx], pred_test[:, idx])
      print("MAE test of " + i +" : %0.3f" %(mae_te), "MAE train of " + i +" : %0.3f" %(mae_te), sep='    ')
      idx = idx + 1


    # plot training history
    plt.plot(history.history['loss'][4:], label='train')
    plt.plot(history.history['val_loss'][4:], label='test')
    plt.legend()
    #plt.show()
    plt.savefig('val_loss_'+str(datetime.today())[0:10]+'.png')

else:
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(db_path) if isfile(join(db_path, f)) and 'best_model' in f]
    onlyfiles = np.sort(onlyfiles)
    nn_r = load_model(db_path + onlyfiles[-1])

    # Predict the model
    pred_train = nn_r.predict(X_train)
    pred_test = nn_r.predict(X_test)

    # RMSE Computation
    mae = MAE(y_train, pred_train)
    print("MAE train : % f" %(mae))
    mae = MAE(y_test, pred_test)
    print("MAE test : % f" %(mae))

    idx = 0
    for i in col_range:
      mae_tr = MAE(y_train.iloc[:, idx], pred_train[:, idx])
      mae_te = MAE(y_test.iloc[:, idx], pred_test[:, idx])
      print("MAE test of " + i +" : %0.3f" %(mae_te), "MAE train of " + i +" : %0.3f" %(mae_te), sep='    ')
      idx = idx + 1

from astropy.time import Time
print('Conversion from met to datetime')
ts = Time(df_data['met'], format='fermi').utc.to_datetime()

import gc
gc.collect()
# TODO too slow
pred_x_tot = nn_r.predict(scaler.transform(df_data.loc[:, col_selected].astype('float32')))
# if len(pred_x.shape)==2:
#   pred_x = pred_x.reshape(1,-1)[0]
y_tot = df_data.loc[:, col_range].astype('float32')
gc.collect()

### Generate a dataset for trigger algorithm
# Original bkg + ssa + met
# TODO add timestamp
df_ori = df_data.loc[:, col_range].astype('float32')
df_ori['met'] = df_data['met']
df_ori['timestamp'] = ts
# Prediction of the bkg
y_pred = pd.DataFrame(pred_x_tot, columns=y_tot.columns)

# Index where there is a time gap
index_saa = np.where((df_ori[['met']].diff() > 500).values)[0]
# range time to delete. Ex 150*4 seconds
time_to_del = 150
max_index = index_saa.max()
min_index = index_saa.min()
set_index = set()
for ind in index_saa:
  set_index = set_index.union(set(range(max(ind-time_to_del, min_index),
                                    min(ind+time_to_del, max_index))))
# df_ori = df_ori.drop(labels=set_index)
# y_pred = y_pred.drop(labels=set_index)
df_ori.loc[set_index, ['n0_r0', 'n0_r1', 'n0_r2', 'n1_r0', 'n1_r1', 'n1_r2', 'n2_r0', 'n2_r1',
       'n2_r2', 'n3_r0', 'n3_r1', 'n3_r2', 'n4_r0', 'n4_r1', 'n4_r2', 'n5_r0',
       'n5_r1', 'n5_r2', 'n6_r0', 'n6_r1', 'n6_r2', 'n7_r0', 'n7_r1', 'n7_r2',
       'n8_r0', 'n8_r1', 'n8_r2', 'n9_r0', 'n9_r1', 'n9_r2', 'na_r0', 'na_r1',
       'na_r2', 'nb_r0', 'nb_r1', 'nb_r2']] = np.nan
y_pred.loc[set_index] = np.nan

df_ori.reset_index(drop=True, inplace=True)
y_pred.reset_index(drop=True, inplace=True)
# time_r = range(0, 10000)
# plt.plot(df_ori.loc[time_r, 'timestamp'], df_ori.loc[time_r, 'n1_r1'], '.')
# plt.plot(df_ori.loc[time_r, 'timestamp'], y_pred.loc[time_r, 'n1_r1'], '.')

# Save the data
df_ori.to_csv(pred_path+'frg_101225.csv', index=False)
y_pred.to_csv(pred_path+'bkg_101225.csv', index=False)
pass