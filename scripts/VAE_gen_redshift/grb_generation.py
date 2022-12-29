import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize, LogNorm
from pylab import imshow
sns.set_theme()
import numpy as np
import matplotlib.image
import pandas as pd
import pickle
from connections.utils.config import DB_PATH
from tqdm import tqdm
# the background fitter interface
from gbm.background import BackgroundFitter
# our fitting algorithm
from gbm.background.binned import Polynomial
from gbm.plot import Lightcurve

LEN_LC = 512 # 64
TIME_RES = 0.064
tte_path = '/beegfs/rcrupi/zzz_other/tte/' # tte
list_tte = os.listdir(tte_path)
tte_pkl_path = '/beegfs/rcrupi/zzz_other/tte_pkl_img/' # tte pkl
list_tte_pkl = [i for i in os.listdir(tte_pkl_path) if 'pickle' in i]
list_tte_bn_pkl = ["_".join(i.split('_')[0:2]) for i in list_tte_pkl]
bool_pkl = False
bool_download_gbm = False
if bool_pkl:
    # Preprocess data downloaded from FTP
    from gbm.data import TTE
    from gbm.binning.unbinned import bin_by_time
    # from gbm.plot import Lightcurve
    if bool_download_gbm:
        from gbm.finder import BurstCatalog
        burstcat = BurstCatalog()
        df_burst = pd.DataFrame(burstcat.get_table())
        df_burst.to_pickle(DB_PATH+'df_burst.pkl')
    else:
        df_burst = pd.read_pickle(DB_PATH+'df_burst.pkl')

    # Filter only the triggered detectors
    def det_triggered(str_mask):
        list_det = np.array(['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7',
                             'n8', 'n9', 'na', 'nb'])
        try:
            idx_det = np.where(np.array([int(i) for i in list(str_mask)]) == 1)
            # Do not consider BGO detectors
            idx_det = np.array([idx_tmp for idx_tmp in idx_det[0] if idx_tmp < 12])
            return list(list_det[idx_det])
        except:
            print("Warning, not found detectors triggered. det_mask: ", str_mask)
            return list(list_det)

    # Initialise dataset
    ds_train = []
    max_len_time = 8000
    np.random.seed(42)
    for tte_tmp in list_tte:
        if tte_tmp.split('_')[3] + "_" + tte_tmp.split('_')[2] in list_tte_bn_pkl:
            continue
        # Problems in loop: glg_tte_n3_bn090626707_v00, glg_tte_n8_bn090626707_v00, glg_tte_na_bn090626707_v00, glg_tte_n0_bn090626707_v00,
        # glg_tte_nb_bn090626707_v00, glg_tte_n6_bn090626707_v00, glg_tte_n4_bn090626707_v00, glg_tte_n2_bn090626707_v00,
        # glg_tte_n5_bn090626707_v00, glg_tte_n9_bn090626707_v00, glg_tte_n1_bn090626707_v00, glg_tte_n7_bn090626707_v00
        # Problem in fit: glg_tte_n0_bn121029350_v00, glg_tte_n2_bn120922939_v00, glg_tte_n0_bn121125469_v00,
        # glg_tte_n5_bn121029350_v00, glg_tte_n9_bn191130507_v00, glg_tte_n7_bn120415891_v00, glg_tte_n0_bn121125356_v00
        try:
            # Check if detector has event signal counts
            str_det = df_burst.loc[df_burst['trigger_name'] == tte_tmp.split('_')[3], 'bcat_detector_mask'].values[0]
            list_det = det_triggered(str_mask=str_det)
            t90_tmp = df_burst.loc[df_burst['trigger_name'] == tte_tmp.split('_')[3], 't90'].values[0]
            t90_start_tmp = df_burst.loc[df_burst['trigger_name'] == tte_tmp.split('_')[3], 't90_start'].values[0]
            t50_tmp = df_burst.loc[df_burst['trigger_name'] == tte_tmp.split('_')[3], 't50'].values[0]
            t90_e_tmp = df_burst.loc[df_burst['trigger_name'] == tte_tmp.split('_')[3], 't90_error'].values[0]
            # print(t90_tmp)
            if tte_tmp.split('_')[2] not in list_det:
                # print(tte_tmp + ' not used.')
                continue
            # read a tte file
            tte = TTE.open(tte_path+tte_tmp)
            print(tte)
            t_min, t_max = tte.time_range
            # bkg times
            # giovanni
            # [-15, -sigma], [t90+sigma, t90+75]
            # mio
            # GRB191130253 - set interval boundaries not with only t90_error (-15, +30)
            try:
                bkg_t_min_2 = max(t90_start_tmp - 3 * t90_e_tmp, t_min + 15)
                bkg_t_min_1 = max(bkg_t_min_2 - 15, t_min)
                bkg_t_max_1 = min(t90_start_tmp + t90_tmp + 3 * t90_e_tmp, t_max - 30)
                bkg_t_max_2 = min(t_max, bkg_t_max_1 + 30)
            except:
                print("Error: Background times ruins.")
                bkg_t_min_2 = -1
                bkg_t_min_1 = -2
                bkg_t_max_1 = 1
                bkg_t_max_2 = 2

            # Selection interval data
            bkg_t_sel_1 = max(t_min, bkg_t_min_2 - (bkg_t_max_1 - bkg_t_min_2) / 2)
            bkg_t_sel_2 = min(t_max, bkg_t_max_1 + (bkg_t_max_1 - bkg_t_min_2) / 2)

            bkgd_times = [(bkg_t_min_1, bkg_t_min_2), (bkg_t_max_1, bkg_t_max_2)]
            # bin in time (0.064s)
            flt_bin_time = max((bkg_t_sel_2 - bkg_t_sel_1)/(LEN_LC-1), TIME_RES/64) # max(t50_tmp/LEN_LC, TIME_RES)
            phaii = tte.to_phaii(bin_by_time, flt_bin_time, time_ref=0.0)
            type(phaii)
            # If the bin time is too small (less than TIME_RES) background estimation is not performed.
            # This is equivalent to require (bkg_t_sel_2 - bkg_t_sel_1) >= (LEN_LC-1)*TIME_RES
            if flt_bin_time >= TIME_RES:
                # we initialize our background fitter with the phaii object, the algorithm, and the time ranges to fit.
                # if we were using an unbinned algorithm, we'd call .from_tte() and give it tte instead of phaii
                backfitter = BackgroundFitter.from_phaii(phaii, Polynomial, time_ranges=bkgd_times)

                # and once initialized, we can run the fit with the fitting parameters appropriate for our algorithm.
                # here, we'll do a 1st order polynomial
                try:
                    backfitter.fit(order=1)
                except Exception as e:
                    print(e)
                    print("Errore fit", tte_tmp.split('_')[3], t90_tmp, bkgd_times)
                    continue

                bkgd = backfitter.interpolate_bins(phaii.data.tstart, phaii.data.tstop)
                type(bkgd)
                # bkgd.counts

                # Select lighturve in the selected background interval
                # This events have enough points to compute the background
                #phaii = phaii.slice_time((-5, t90_tmp*1.5))
                #bkgd = bkgd.slice_time(-5, t90_tmp*1.5)
                pdc = phaii.data.rates - bkgd.rates
                pdc = pdc[(phaii.data.time_centroids >= bkg_t_sel_1 - flt_bin_time/2) &
                          (phaii.data.time_centroids <= bkg_t_sel_2 + flt_bin_time/2)]
                print(pdc.shape)
                # plt.plot(pdc.sum(axis=1))
                # # plot the lightcurve
                # lcplot = Lightcurve(data=phaii.to_lightcurve(time_range=(bkg_t_sel_1, bkg_t_sel_2), energy_range=(8, 900)),
                #                     background=bkgd.integrate_energy(emin=8, emax=900))
                # lcplot.add_selection(phaii.to_lightcurve(time_range=(bkg_t_min_2, bkg_t_max_1), energy_range=(8, 900)))
                # plt.show()
            else:
                # pdc = phaii.data.rates - np.quantile(phaii.data.rates, q=0.5, axis=0, keepdims=True, interpolation='linear')
                # TODO make more robust the mean. Bkg order 0?
                pdc = phaii.data.rates - np.mean(phaii.data.rates, axis=0, keepdims=True)
                pdc = pdc[(phaii.data.time_centroids >= bkg_t_sel_1 - flt_bin_time / 2) &
                          (phaii.data.time_centroids <= bkg_t_sel_2 + flt_bin_time / 2)]
                print(pdc.shape)
                # plt.plot(pdc.sum(axis=1))
                # # plot the lightcurve
                # lcplot = Lightcurve(data=phaii.to_lightcurve(time_range=(bkg_t_sel_1, bkg_t_sel_2), energy_range=(8, 900)))
                # lcplot.add_selection(phaii.to_lightcurve(time_range=(bkg_t_min_2, bkg_t_max_1), energy_range=(8, 900)))
            #plt.show()
            #continue

            # Pad the values up to LEN_LC
            if pdc.shape[0] > LEN_LC:
                print("Warning. An event is cut for the sake of dimensions. dim original: " + str(pdc.shape[0]))
                pdc = pdc[0:LEN_LC, :]
            elif LEN_LC > pdc.shape[0]:
                if pdc.shape[0] == 0:
                    print("Error. No data in ", tte_tmp.split('_')[3], t90_tmp)
                # Padding with 0
                # pdc = np.pad(pdc, [(0, diff_len), (0, 0)], mode='edge')
                diff_len = LEN_LC - pdc.shape[0]
                pdc = np.pad(pdc, [(0, diff_len), (0, 0)], mode='constant', constant_values=0)
                print("Logging. Padded series.")

            # Save GRB image
            name_file = str(tte_tmp.split('_')[3]) + "_" + str(tte_tmp.split('_')[2]) + "_bin" + str(round(flt_bin_time, 4))
            with open(tte_pkl_path + name_file + '.pickle', 'wb') as f:
                pickle.dump(pdc, f)
            path_tmp = tte_pkl_path + name_file + '.svg'
            matplotlib.image.imsave(path_tmp, pdc.T, format='svg')
        except Exception as e:
            print(e)
            print("Error in loop.", tte_tmp.split('_')[3])
else:
    ds_train = []
    bln_all = False
    if bln_all:
        lst_tte_pkl_files = list_tte_pkl
    else:
        lst_tte_pkl_files = list_tte_pkl[:2000]
    for i in tqdm(lst_tte_pkl_files):
        with open(tte_pkl_path+i, 'rb') as f:
            ds_train.append(pickle.load(f))
    ds_train = np.array(ds_train)

# Scale dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
ds_train_scale = ds_train.copy()
for i in range(0, ds_train.shape[0]):
    ds_train_scale[i, :, :] = StandardScaler().fit_transform(ds_train[i, :, :])

idx_fig = 0
# Create two subplots and unpack the output array immediately
f, (ax1, ax2) = plt.subplots(2, 1)
ax1.imshow(ds_train[idx_fig, :, :].T)
ax1.set_title('Spectrogram')
ax1.axis('off')
ax2.plot(ds_train[idx_fig, :, :].sum(axis=1))
ax2.set_title('Lightcurve')

f2, (ax3, ax4) = plt.subplots(2, 1)
ax3.imshow(ds_train_scale[idx_fig, :, :].T)
ax3.set_title('Spectrogram scaled')
ax3.axis('off')
ax4.plot(ds_train_scale[idx_fig, :, :].sum(axis=1))
ax4.set_title('Lightcurve scaled')

print('end')
pass

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
N_HIDDEN = 32


def cnn_model(N_HIDDEN=32, loss='mae'):
    input = layers.Input(shape=(512, 128, 1))
    input_decoder = layers.Input(shape=(N_HIDDEN,))
    # Encoder
    x = layers.Conv2D(32, (2, 2), activation="relu", padding="same")(input)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(32, (4, 4), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((4, 4), padding="same")(x)
    x = layers.Conv2D(32, (8, 8), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((8, 8), padding="same")(x)
    x = layers.Flatten()(x)
    # Hidden layer
    h = layers.Dense(N_HIDDEN, activation='relu', name='hidden')(x)
    # Decoder
    y = layers.Dense(x.shape[1], activation='relu')(input_decoder)
    y = layers.Reshape((8, 2, 32), input_shape=(N_HIDDEN,))(y)
    y = layers.Conv2DTranspose(32, (9, 9), strides=2, activation="relu", padding="same")(y)
    y = layers.Conv2DTranspose(32, (6, 6), strides=2, activation="relu", padding="same")(y)
    y = layers.Conv2DTranspose(32, (6, 6), strides=2, activation="relu", padding="same")(y)
    y = layers.Conv2DTranspose(32, (3, 3), strides=4, activation="relu", padding="same")(y)
    y = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(y)
    y = layers.Conv2D(1, (3, 3), activation="linear", padding="same")(y)

    # Autoencoder
    encoder = Model(input, h)
    decoder = Model(input_decoder, y)
    autoencoder = Model(input, decoder(encoder(input)))

    opt = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    autoencoder.compile(optimizer=opt, loss=loss)
    autoencoder.summary()
    return autoencoder, encoder, decoder


def dense_model(N_HIDDEN=32):
    input = layers.Input(shape=(512, 128, 1))
    input_decoder = layers.Input(shape=(N_HIDDEN,))
    # Encoder
    x = layers.Flatten()(input)
    x = layers.Dense(N_HIDDEN*128, activation='relu')(x)
    # Hidden layer
    h = layers.Dense(N_HIDDEN, activation='relu', name='hidden')(x)
    # Decoder
    y = layers.Dense(N_HIDDEN*128, activation='linear')(input_decoder)
    y = layers.Dense(512*128, activation='linear')(y)
    y = layers.Reshape((512, 128, 1), input_shape=(y.shape[1],))(y)

    # Autoencoder
    encoder = Model(input, h)
    decoder = Model(input_decoder, y)
    autoencoder = Model(input, decoder(encoder(input)))

    opt = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    autoencoder.compile(optimizer=opt, loss="mse")
    autoencoder.summary()
    return autoencoder, encoder, decoder


autoencoder, encoder, decoder = cnn_model(512)

autoencoder.fit(
    x=ds_train_scale,
    y=ds_train_scale,
    epochs=4,
    batch_size=128,
    shuffle=True,
    validation_split=0.2,
)

emb_data = encoder.predict(ds_train_scale[0:1, :, :])
predictions = decoder.predict(emb_data)
plt.figure()
plt.imshow(predictions[0, :, :, 0].T)
plt.figure()
plt.imshow(ds_train_scale[0, :, :].T)
