import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize, LogNorm
sns.set_theme()
import numpy as np
import pandas as pd
from gbm.data import TTE
from gbm.binning.unbinned import bin_by_time
from gbm.plot import Lightcurve
from gbm.finder import BurstCatalog
import pickle

tte_path = '/beegfs/rcrupi/zzz_other/tte_pkl/' # tte
list_tte = os.listdir(tte_path)
bool_pkl = False
if bool_pkl:
    burstcat = BurstCatalog()
    df_burst = pd.DataFrame(burstcat.get_table())

    def det_triggered(str_mask):
        list_det = np.array(['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7',
                             'n8', 'n9', 'na', 'nb'])
        try:
            idx_det = np.where(np.array([int(i) for i in list(str_mask)]) == 1)
            return list(list_det[idx_det])
        except:
            print("Error, not found detectors triggered.")
            return list(list_det)

    ds_train = []
    max_len_time = 8000
    for tte_tmp in list_tte:
        # Check if detector has event signal counts
        str_det = df_burst.loc[df_burst['trigger_name'] == tte_tmp.split('_')[3], 'bcat_detector_mask'].values[0]
        list_det = det_triggered(str_mask=str_det)
        if tte_tmp.split('_')[2] not in list_det:
            # print(tte_tmp + ' not used.')
            continue
        # read a tte file
        tte = TTE.open(tte_path+tte_tmp)
        print(tte)
        # bin in time
        phaii = tte.to_phaii(bin_by_time, 0.256, time_ref=0.0)
        type(phaii)
        # # plot the lightcurve
        # lcplot = Lightcurve(data=phaii.to_lightcurve())
        # plt.show()
        pdc = phaii.data.counts
        if pdc.shape[0] < max_len_time:
            diff_len = max_len_time - pdc.shape[0]
            pdc = np.pad(pdc, [(0, diff_len), (0, 0)], mode='constant', constant_values=0)
        else:
            print("An event is cut for the sake of dimensions. dim original: " + str(pdc.shape[0]))
            pdc = pdc[0:max_len_time, :]
        ds_train.append(pdc)
        # # Draw a heatmap with the numeric values in each cell
        # index = (phaii.data.time_centroids>=-15) & (phaii.data.time_centroids<15)
        # pd_ctime = pd.DataFrame(phaii.data.counts[index]).T
        # pd_ctime.index = np.around(phaii.data.energy_centroids)
        # pd_ctime.columns = np.around(phaii.data.time_centroids[index], 2)
        # f, ax = plt.subplots(figsize=(12, 8))
        # sns.heatmap(pd_ctime.loc[:, :], annot=False, fmt="d", linewidths=.5, ax=ax, norm=LogNorm()) # Normalize, LogNorm

    # mat_len = []
    # for i in ds_train:
    #     mat_len.append(i.shape)
    # mat_len = np.array(mat_len)
    # # max=8000, q75%=4000
    # plt.boxplot(mat_len[:, 0])

    ds_train = np.array(ds_train)

    for i in range(0, ds_train.shape[0]):
        with open(tte_path + 'ds_train' + str(i) + '.pickle', 'wb') as f:
            pickle.dump(ds_train[i, :, :], f)
    # with open(tte_path+'ds_train.pickle', 'wb') as f:
    #     pickle.dump(ds_train, f)
else:
    ds_train = []
    for i in os.listdir(tte_path):
        with open(tte_path+i, 'rb') as f:
            ds_train.append(pickle.load(f))
    ds_train = np.array(ds_train)

print('end')
pass

import tensorflow as tf
from tensorflow.keras import layers

input_img = tf.keras.Input(shape=(8000, 128, 1))

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (5, 5), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (7, 7), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = layers.Conv2D(8, (7, 7), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (5, 5), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = tf.keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Preprocessing
# max_ds_train = ds_train.max()
x_train = ds_train[:1000].astype('float32') / 4000
x_test = ds_train[1000:].astype('float32') / 4000

x_train = np.reshape(x_train, (len(x_train), 8000, 128, 1))
x_test = np.reshape(x_test, (len(x_test), 8000, 128, 1))

from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=4, # 128
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])




