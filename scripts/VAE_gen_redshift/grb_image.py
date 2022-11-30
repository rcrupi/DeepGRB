import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize, LogNorm
from pylab import imshow
sns.set_theme()
import numpy as np
import pandas as pd
import pickle
from connections.utils.config import DB_PATH
from tqdm import tqdm

tte_path = '/beegfs/rcrupi/zzz_other/tte/' # tte
list_tte = os.listdir(tte_path)
tte_pkl_path = '/beegfs/rcrupi/zzz_other/tte_pkl/' # tte pkl
list_tte_pkl = os.listdir(tte_pkl_path)
bool_pkl = True
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
            print("Error, not found detectors triggered. det_mask: ", str_mask)
            return list(list_det)

    # Initialise dataset
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
        # bin in time (0.256s)
        flt_bin_time = 0.256
        phaii = tte.to_phaii(bin_by_time, flt_bin_time, time_ref=0.0)
        type(phaii)
        # # plot the lightcurve
        # lcplot = Lightcurve(data=phaii.to_lightcurve())
        # plt.show()
        # TODO divide lightcurve 2400 and 1200
        # TODO select GRB only by T90 from 500 and 100
        pdc = phaii.data.counts
        # Pad the values up to max_len_time
        if pdc.shape[0] < max_len_time:
            diff_len = max_len_time - pdc.shape[0]
            # Padding with 0
            # pdc = np.pad(pdc, [(0, diff_len), (0, 0)], mode='constant', constant_values=0)
            # Padding with the averages of last values
            vet_pad = np.mean(pdc[-40:-5, :], axis=0)
            pdc = np.pad(pdc, [(0, diff_len), (0, 0)], mode='edge')
            pdc[max_len_time - diff_len:max_len_time, :] = vet_pad
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
        with open(tte_pkl_path + 'ds_train' + str(i) + '.pickle', 'wb') as f:
            pickle.dump(ds_train[i, :, :], f)
    # with open(tte_path+'ds_train.pickle', 'wb') as f:
    #     pickle.dump(ds_train, f)
else:
    ds_train = []
    bln_all = False
    if bln_all:
        lst_tte_pkl_files = os.listdir(tte_pkl_path)
    else:
        lst_tte_pkl_files = os.listdir(tte_pkl_path)[0:2000]
    for i in tqdm(lst_tte_pkl_files):  # TODO change to get all!!!
        with open(tte_pkl_path+i, 'rb') as f:
            ds_train.append(pickle.load(f))
    ds_train = np.array(ds_train)

print('end')
pass

import tensorflow as tf
from tensorflow.keras import layers

n_e_channel = 128

# input_img = tf.keras.Input(shape=(8000, n_e_channel, 1))
#
# x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
# x = layers.MaxPooling2D((20, 2), padding='same')(x)
# x = layers.Conv2D(8, (5, 5), activation='relu', padding='same')(x)
# x = layers.MaxPooling2D((20, 2), padding='same')(x)
# x = layers.Conv2D(8, (7, 7), activation='relu', padding='same')(x)
# encoded = layers.MaxPooling2D((20, 2), padding='same')(x)
#
# # at this point the representation is (4, 4, 8) i.e. 128-dimensional
#
# input_enc = tf.keras.Input(shape=(1, 16, 8))
#
# x = layers.Conv2D(8, (7, 7), activation='relu', padding='same')(input_enc)
# x = layers.UpSampling2D((20, 2))(x)
# x = layers.Conv2D(8, (5, 5), activation='relu', padding='same')(x)
# x = layers.UpSampling2D((20, 2))(x)
# x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
# x = layers.UpSampling2D((20, 2))(x)
# decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
#
# encoder = tf.keras.Model(input_img, encoded)
# decoder = tf.keras.Model(input_enc, decoded)
# autoencoder = tf.keras.Model(input_img, decoder(encoder(input_img)))
#
# opt = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.8, beta_2=0.8, epsilon=1e-07)
# autoencoder.compile(optimizer=opt, loss='mse')  # loss='mae' binary_crossentropy
#
# # Preprocessing
# # max_ds_train = ds_train.max()
# # Ignore last two channels
# ds_train[:, :, 126:] = 0
#
# x_train = ds_train[:400, :, 0:n_e_channel].astype('float32') / 4000  # TODO change filter
# x_test = ds_train[400:, :, 0:n_e_channel].astype('float32') / 4000  # TODO change filter
#
# x_train = np.reshape(x_train, (len(x_train), 8000, n_e_channel, 1))
# x_test = np.reshape(x_test, (len(x_test), 8000, n_e_channel, 1))
#
# from keras.callbacks import TensorBoard
#
# autoencoder.fit(x_train, x_train,
#                 epochs=1,
#                 batch_size=8, # 4, 128
#                 shuffle=True,
#                 validation_data=(x_test, x_test),)
#                 # callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
#
# print(encoder(x_test[0:1]).shape)
# print(autoencoder(x_test[0:1]).shape)
# print(decoder(encoder(x_test[0:1])).shape)
#
# imshow(x_train[0:1][0, 0:1300, :, 0].T, interpolation='nearest')
# imshow(autoencoder.predict(x_train[0:1])[0, 0:1300, :, 0].T, interpolation='nearest')
#
# imshow(x_test[0, 0:1300, :, 0].T, interpolation='nearest')
# imshow(autoencoder.predict(x_test[0:1])[0, 0:1300, :, 0].T, interpolation='nearest')
#
# rand_emb = np.random.uniform(low=0, high=1, size=(1, 1, 16, 8))
# imshow(decoder.predict(rand_emb)[0, 0:400, :, 0].T, interpolation='nearest')
#
# print("fine")


# # # Autoencoder recurrent

ds_train[:, :, 126:] = 0

x_train = ds_train[:9000, :, 0:n_e_channel].astype('float32') # / 4000  # TODO change filter
x_test = ds_train[9000:, :, 0:n_e_channel].astype('float32') # / 4000  # TODO change filter

x_train = np.reshape(x_train, (len(x_train), 8000, n_e_channel, 1))
x_test = np.reshape(x_test, (len(x_test), 8000, n_e_channel, 1))

x_train_l = x_train.mean(axis=2)
x_test_l = x_test.mean(axis=2)

x_train_l = x_train_l / x_train_l.max(axis=1)[:, None, :]
x_test_l = x_test_l / x_test_l.max(axis=1)[:, None, :]

n_e_channel = 128
max_dimension = 8000 #  8000
x_train_l = x_train_l[:, 0:max_dimension, :]
x_test_l = x_test_l[:, 0:max_dimension, :]
#
# # define model
# import tensorflow_addons as tfa
# model = tf.keras.Sequential()
# #model.add(tf.keras.layers.GRU(100, activation='relu', return_sequences=True))
# model.add(tfa.layers.ESN(16, activation='relu', input_shape=(max_dimension, 1), return_sequences=True))
# model.add(tfa.layers.ESN(16, activation='relu', input_shape=(max_dimension, 1), return_sequences=False))
# # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(1000, activation='relu', input_shape=(max_dimension, 1))))
# model.add(tf.keras.layers.RepeatVector(max_dimension))
# model.add(tf.keras.layers.Dropout(0.005))
# #model.add(tf.keras.layers.GRU(100, activation='relu', return_sequences=True))
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16, activation='relu', return_sequences=True, dropout=0.005, recurrent_dropout=0.005)))
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16, activation='relu', return_sequences=True, dropout=0.005, recurrent_dropout=0.005)))
# model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(8, activation='sigmoid')))
# model.add(tf.keras.layers.Dropout(0.005))
# model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid')))
# opt = tf.keras.optimizers.Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
#
# def qloss(y_true, y_pred):
#     qs = 0.75
#     e = y_true - y_pred
#     return tf.maximum(e, 0) # tf.maximum(qs*e, (qs-1)*e)
#
# model.compile(optimizer=opt, loss='mse')  # qloss 'mse'
#
# # fit model
# # model.layers[0].trainable = False
# model.fit(x_train_l, x_train_l, epochs=16,
#                 batch_size=16, # 4, 128
#                 shuffle=True,
#                 validation_data=(x_test_l, x_test_l))
# # demonstrate recreation
# m = np.random.choice(range(x_train_l.shape[0]))
# yhat = model.predict(x_train_l[m:m+1])
# #print(yhat[0, :, 0])
# plt.plot(x_train_l[m:m+1][0, :, 0])
# plt.plot(yhat[0, :, 0], 'x')
#
# m = np.random.choice(range(x_test_l.shape[0]))
# yhat = model.predict(x_test_l[m:m+1])
# #print(yhat[0, :, 0])
# plt.plot(x_test_l[m:m+1][0, :, 0])
# plt.plot(yhat[0, :, 0], 'x')
#
#
# encoder = tf.keras.Model(inputs=model.inputs, outputs=model.layers[0].output)

# # # plain NN
x_train_l = x_train_l[:, :, 0]
x_test_l = x_test_l[:, :, 0]
bln_plain = False
if bln_plain:
    input_img = tf.keras.Input(shape=(8000))

    x = layers.Dense(4000, activation='relu')(input_img)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.05)(x)
    x = layers.Dense(200, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.05)(x)
    x = layers.Dense(32, activation='relu')(x)
    encoded = layers.BatchNormalization()(x)

    input_enc = tf.keras.Input(shape=(32))
    x = layers.Dense(200, activation='relu')(input_enc)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.05)(x)
    x = layers.Dense(4000, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.05)(x)
    decoded = layers.Dense(8000, activation='sigmoid')(x)

    encoder = tf.keras.Model(input_img, encoded)
    decoder = tf.keras.Model(input_enc, decoded)
    autoencoder = tf.keras.Model(input_img, decoder(encoder(input_img)))

    opt = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', min_delta=0.01, patience=4)
    autoencoder.compile(optimizer=opt, loss='mse')  # qloss 'mse'

    autoencoder.fit(x_train_l, x_train_l, epochs=16,
                    batch_size=16, # 4, 128
                    shuffle=True,
              validation_split=0.2,
              callbacks=[es])

    # x_enc = encoder.predict(x_train_l)
    # np.median(x_enc, axis=0)
    # np.mean(x_enc, axis=0)
    # np.std(x_enc, axis=0)

    yhat = decoder.predict(np.random.normal(0, 1, size=(1, 32)))
    plt.plot(yhat[0, :], 'x-')

    m = np.random.choice(range(x_train_l.shape[0]))
    yhat = autoencoder.predict(x_train_l[m:m+1])[0, :]
    #print(yhat[0, :, 0])
    plt.plot(x_train_l[m:m+1][0, :])
    plt.plot(yhat, 'x-')

    m = np.random.choice(range(x_test_l.shape[0]))
    yhat = autoencoder.predict(x_test_l[m:m+1])
    #print(yhat[0, :, 0])
    plt.plot(x_test_l[m:m+1][0, :, 0])
    plt.plot(yhat[0, :, 0], 'x-')


# # # VAE
original_dim = 8000
intermediate_dim = 1000
latent_dim = 2

inputs = tf.keras.Input(shape=(original_dim,))
h = layers.Dense(intermediate_dim, activation='relu')(inputs)
#h = layers.BatchNormalization()(h)
#h = layers.Dense(intermediate_dim, activation='relu')(h)
z_mean = layers.Dense(latent_dim)(h)
z_log_sigma = layers.Dense(latent_dim)(h)
# We can use these parameters to sample new similar points from the latent space:

from tensorflow.keras import backend as K

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_sigma])
# Finally, we can map these sampled latent points back to reconstructed inputs:

# Create encoder
encoder = tf.keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

# Create decoder
latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
# x = layers.BatchNormalization()(x)
#x = layers.Dense(intermediate_dim*4, activation='relu')(x)
outputs = layers.Dense(original_dim, activation='sigmoid')(x)
decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = tf.keras.Model(inputs, outputs, name='vae_mlp')
# What we've done so far allows us to instantiate 3 models:

# an end-to-end autoencoder mapping inputs to reconstructions
# an encoder mapping inputs to the latent space
# a generator that can take points on the latent space and will output the corresponding reconstructed samples.
# We train the model using the end-to-end model, with a custom loss function: the sum of a reconstruction term, and the KL divergence regularization term.

reconstruction_loss = tf.keras.losses.mse(inputs, outputs)  # mse, mape, msle
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
opt = tf.keras.optimizers.Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', min_delta=0.01, patience=16)
mc = tf.keras.callbacks.ModelCheckpoint('vae', monitor='val_loss', mode='min', verbose=0, save_best_only=True)
vae.compile(optimizer=opt)
# We train our VAE on MNIST digits:
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# Data Agumentation
from scipy.ndimage.interpolation import shift
np.random.seed(42)
max_shift = 4000
for i in range(0, x_train_l.shape[0]):
    x_train_tmp_shift = shift(x_train_l[i, :], np.random.randint(max_shift), mode='nearest')
    x_train_l = np.append(x_train_l, [x_train_tmp_shift], axis=0)

bln_save = True
if bln_save:
    history = vae.fit(x_train_l, x_train_l,
            epochs=64, batch_size=32, validation_split=0.2, callbacks=[mc, es])
    # plt.plot(history.history['loss'][4:], label='train')
    # plt.plot(history.history['val_loss'][4:], label='val')
    plt.legend()
    vae.save("vae_trained" + '.h5')
else:
    vae = tf.keras.models.load_model("vae_trained" + '.h5')
# Because our latent space is two-dimensional, there are a few cool visualizations that can be done at this point.
# One is to look at the neighborhoods of different classes on the latent 2D plane:

x_enc = encoder.predict(x_train_l)[2]
print("men latent: ", np.mean(x_enc))
mean_lat_var = np.mean(x_enc, axis=0)
print("mean std: ", np.std(x_enc))
std_lat_var = np.std(x_enc, axis=0)

# n_plots = 6
# vet_dim_1 = [mean_lat_var[0] + std_lat_var[0]*l for l in np.linspace(-1,1,n_plots)]
# vet_dim_2 = [mean_lat_var[1] + std_lat_var[1]*l for l in np.linspace(-1,1,n_plots)]
#
# fig, axs = plt.subplots(n_plots, n_plots)
# for idx_1, lat_var_1 in enumerate(vet_dim_1):
#     for idx_2, lat_var_2 in enumerate(vet_dim_2):
#         yhat = decoder.predict(np.array([[lat_var_1, lat_var_2]]))
#         axs[idx_1, idx_2].plot(yhat[0, :], 'x-')
#         axs[idx_1, idx_2].set_title(f"l1: {round(lat_var_1, 2)}. l2: {round(lat_var_2, 2)}")


for i in range(0, 20):
    yhat = decoder.predict(np.random.normal(mean_lat_var, std_lat_var, size=(1, latent_dim)))
    plt.figure()
    plt.plot(yhat[0, :], 'x-')
    plt.savefig('not_real_'+str(i)+'.png')

    m = np.random.choice(range(x_train_l.shape[0]))
    yhat = vae.predict(x_train_l[m:m+1])[0, :]
    plt.figure()
    plt.plot(x_train_l[m:m+1][0, :])
    plt.plot(yhat, 'x-')
    plt.savefig('reconstruct_' + str(i) + '.png')
