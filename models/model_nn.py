# import utils
from connections.utils.config import PATH_TO_SAVE, FOLD_PRED, FOLD_BKG, GBM_BURST_DB, FOLD_NN
import logging
# Standard packages
import matplotlib.pyplot as plt
import seaborn as sns
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
from sqlalchemy import create_engine
import gc
from astropy.time import Time
# Preprocess
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as MAE, median_absolute_error as MeAE
from sklearn.preprocessing import StandardScaler
# Tensorflow, Keras
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
# Custom losses
from models.utils.losses import loss_median, loss_max
# Explainability
import shap


class ModelNN:
    def __init__(self, start_month, end_month):
        """
        Define NN class.
        :param start_month: str, the starting month to consider background (included). E.g. '08-2012' for August 2012.
        :param end_month: str, the ending month to consider background (excluded). E.g. '09-2012' for August 2012.
        """
        # Data and index filter initialised
        self.df_data = None
        self.index_date = None
        # Start and end month of dataset
        self.start_month = start_month
        self.end_month = end_month
        # Model nn
        self.scaler = None
        self.nn_r = None
        # Columns to select:
        # -) met datetime
        self.col_met = ['met']
        # -) Counts of photons: y
        self.col_range = ['n0_r0', 'n0_r1', 'n0_r2', 'n1_r0', 'n1_r1', 'n1_r2', 'n2_r0',
                          'n2_r1', 'n2_r2', 'n3_r0', 'n3_r1', 'n3_r2', 'n4_r0', 'n4_r1', 'n4_r2',
                          'n5_r0', 'n5_r1', 'n5_r2', 'n6_r0', 'n6_r1', 'n6_r2', 'n7_r0', 'n7_r1',
                          'n7_r2', 'n8_r0', 'n8_r1', 'n8_r2', 'n9_r0', 'n9_r1', 'n9_r2', 'na_r0',
                          'na_r1', 'na_r2', 'nb_r0', 'nb_r1', 'nb_r2']
        # -) satellite info
        self.col_sat_pos = ['pos_x', 'pos_y', 'pos_z', 'a', 'b', 'c', 'd', 'lat', 'lon', 'alt',
                            'vx', 'vy', 'vz', 'w1', 'w2', 'w3', 'sun_vis', 'sun_ra', 'sun_dec',
                            'earth_r', 'earth_ra', 'earth_dec', 'saa', 'l']
        # -) detectors info
        self.col_det_pos = ['n0_ra', 'n0_dec', 'n0_vis',
                            'n1_ra', 'n1_dec', 'n1_vis', 'n2_ra', 'n2_dec', 'n2_vis', 'n3_ra',
                            'n3_dec', 'n3_vis', 'n4_ra', 'n4_dec', 'n4_vis', 'n5_ra', 'n5_dec',
                            'n5_vis', 'n6_ra', 'n6_dec', 'n6_vis', 'n7_ra', 'n7_dec', 'n7_vis',
                            'n8_ra', 'n8_dec', 'n8_vis', 'n9_ra', 'n9_dec', 'n9_vis', 'na_ra',
                            'na_dec', 'na_vis', 'nb_ra', 'nb_dec', 'nb_vis']
        # Columns of input: X
        self.col_selected = self.col_sat_pos + self.col_det_pos

    def prepare(self, bool_del_trig=True):
        """
        Load all csv files, filter the data in SSA, delete trigger events if specified by bool_del_trig.
        :param bool_del_trig: boolean, if True the triggered events in GBM catalogue are deleted in the train set.
        :return: None
            self.df_data -> dataset
            self.index_date -> index to filter dataset
        """
        logging.info("Start preparing dataframe input to NN model.")
        start_month_day = self.start_month.split('-')[1][2:4] + self.start_month.split('-')[0] + '01' + '.csv'
        end_month_day = self.end_month.split('-')[1][2:4] + self.end_month.split('-')[0] + '01' + '.csv'
        logging.info('Load ' + start_month_day + ' to ' + end_month_day)
        # List csv files
        list_csv = np.sort([i_day for i_day in os.listdir(PATH_TO_SAVE + '/' + FOLD_BKG) if '.csv' in i_day and
                            (start_month_day <= i_day < end_month_day)])
        # Define final data table. This goes in input to the NN
        df_data = pd.DataFrame()
        # Load each csv files
        logging.info("Loading csv files.")
        yymm = ''
        for csv_tmp in list_csv:
            if csv_tmp[0:4] != yymm:
                logging.info("Loading: " + csv_tmp)
                yymm = csv_tmp[0:4]
            df_tmp = pd.read_csv(PATH_TO_SAVE + FOLD_BKG + '/' + csv_tmp)
            # import datetime
            # df_tmp['day'] = datetime.datetime(int('20'+csv_tmp[0:2]), int(csv_tmp[2:4]), int(csv_tmp[4:6])).timetuple().tm_yday
            df_data = df_data.append(df_tmp, ignore_index=True)
        del df_tmp

        # Filter data within saa
        logging.info("Filtering data when Fermi is in SAA.")
        df_data = df_data.loc[df_data['saa'] == 0, self.col_met + self.col_range + self.col_sat_pos +
                              self.col_det_pos].reset_index(drop=True)  # + ['day']
        logging.info(df_data.head())
        if bool_del_trig:
            logging.info("Deleting events already present in GBM calalogue in Train Set.")
            # Take index of the time where triggers were identified
            engine = create_engine('sqlite:////' + GBM_BURST_DB + 'gbm_burst_catalog.db')
            # TODO update this table. Updated up to 2021-01
            gbm_tri = pd.read_sql_table('GBM_TRI', con=engine)
            index_date = df_data['met'] >= 0
            index_tmp = 1
            for _, row in gbm_tri.iterrows():
                if (index_tmp - 1) % 1000 == 0:
                    print('index reached: ', index_tmp)
                index_date = ((df_data['met'] <= row['met_time']) | (
                            df_data['met'] >= row['met_end_time'])) & index_date
                index_tmp += 1
        else:
            index_date = df_data.index

        # Filter zero counts in frg
        index_date = (df_data[self.col_range] > 0).all(axis=1) & index_date

        logging.info("End prepare data")
        self.df_data = df_data
        self.index_date = index_date

        self.col_selected = self.col_selected  # + ['day']

    def build_model_hype(self, hp):
        nn_input = tf.keras.Input(shape=(len(self.col_selected),))
        hp_units = hp.Choice('units', values=[32, 256, 512, 1024, 2048, 4096])
        dropout = hp.Choice('dropout', values=[0.01, 0.05, 0.1, 0.2, 0.4])
        # First layers
        model_1 = Dense(hp_units, activation='relu')(nn_input)
        model_1 = Dropout(dropout)(model_1)
        # Second layer
        nn_r = Dense(hp_units, activation='relu')(model_1)
        nn_r = Dropout(dropout)(nn_r)
        # Third layer
        nn_r = Dense(int(hp_units / 2), activation='relu')(nn_r)
        # Fourth (last) layer output
        outputs = Dense(len(self.col_range), activation='relu')(nn_r)
        nn_r = tf.keras.Model(inputs=[nn_input], outputs=outputs)
        # Optimizer
        lr = hp.Choice('learning_rate', values=[0.01, 0.007, 0.004, 0.003, 0.002, 0.001, 0.0005, 0.0001])
        beta_1 = hp.Choice('beta_1', values=[0.5, 0.7, 0.8, 0.9, 0.99])
        beta_2 = hp.Choice('beta_2', values=[0.5, 0.7, 0.8, 0.9, 0.99])
        opt = tf.keras.optimizers.Nadam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, epsilon=1e-07)
        loss = 'mae'
        # Compile nn model
        nn_r.compile(loss=loss, loss_weights=1, optimizer=opt)
        return nn_r

    def train(self, bool_train=True, bool_hyper=False, loss_type='mean', units=4000, epochs=512, lr=0.001, bs=2000,
              model_pretrain=None, do=0.05, modelcheck=True):
        """
        Train a neural network to estimate counts of detectors.
        In FOLD_NN is saved: the NN model, train and validation performance during epochs,
         performance per each detector_range counts.
        :param bool_train: boolean, if True the nn is trained otherwise it is loaded in FOLD_NN folder
            (same months but minimum loss).
        :param bool_hyper: boolean, train with keras_tuner. Find best hyperparameters but very slow.
        :param loss_type: str, if 'mean' the loss is the average of MeanAEs, if 'max' is the max of MeanAEs, if 'median'
            the loss is Median Absolute Error
        :param units: number of nodes in the first and second layer, the third is halved.
        :param epochs: number of epochs of the NN.
        :param lr: learning rate of the NN during training.
        :param bs: batch size of the NN during training.
        :param model_pretrain: string, if founded the pretrained model is loaded e re-trained.
        :param do: parameters for the dropout between layers.
        :param modelcheck: if True in the fitting model, the best one in the validation set is selected.
        :return: None
            self.scaler -> Standard scaler object for input X.
            self.nn_r -> The model Neural Network that is a regressor from X to y.

        """
        # # # NN
        # Load the data
        logging.info("Define X (input) as satellite info and detectors info, y (target) as detector counts.")
        y = self.df_data.loc[self.index_date, self.col_range].astype('float32')
        X = self.df_data.loc[self.index_date, self.col_selected].astype('float32')
        # Splitting
        logging.info("Split and scale X, y.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, shuffle=True)
        # Scale
        logging.info("Standard scaling X.")
        scaler = StandardScaler()
        scaler.fit(X_train)
        self.scaler = scaler
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        if bool_train:
            logging.info("NN define with Tensorflow.")
            # Num of inputs as columns of table
            nn_input = tf.keras.Input(shape=(X_train.shape[1],))
            # First layers
            model_1 = Dense(units, activation='relu')(nn_input)
            model_1 = Dropout(do)(model_1)
            # Second layer
            nn_r = Dense(units, activation='relu')(model_1)
            nn_r = Dropout(do)(nn_r)
            # Third layer
            nn_r = Dense(int(units / 2), activation='relu')(nn_r)
            nn_r = Dropout(do)(nn_r)
            # Fourth (last) layer output
            outputs = Dense(len(self.col_range), activation='relu',
                            # kernel_regularizer=tf.keras.regularizers.l2(l2=1e-1),
                            # bias_regularizer=tf.keras.regularizers.l2(1e-1),
                            # activity_regularizer=tf.keras.regularizers.l2(1e-5)
                            )(nn_r)
            nn_r = tf.keras.Model(inputs=[nn_input], outputs=outputs)
            # Optimizer
            opt = tf.keras.optimizers.Nadam(learning_rate=lr, beta_1=0.8, beta_2=0.8, epsilon=1e-07)
            # opt = tf.keras.optimizers.RMSprop( learning_rate=0.002, rho=0.6, momentum=0.0, epsilon=1e-07)

            if loss_type == 'max':
                logging.info('Loss chosen: Max Mean Absolute Error.')
                # Define Loss as max_i(det_ran_error)
                loss = loss_max
            elif loss_type == 'median':
                logging.info('Loss chosen: Median Absolute Error.')
                # Define Loss as average of Median Absolute Error for each detector_range
                loss = loss_median

            elif loss_type == 'mean' or loss_type == 'mae':
                # Define Loss as average of Mean Absolute Error for each detector_range
                logging.info('Loss chosen: Mean Absolute Error.')
                loss = 'mae'
                # loss = tf.keras.losses.MeanAbsoluteError()
            elif loss_type == 'huber':
                loss = tf.keras.losses.Huber(delta=1)
            else:
                # Define Loss as average of Mean Squared Error for each detector_range
                logging.info('Loss chosen: Mean Squared Error.')
                loss = 'mse'

            # Load pretrain model if specified
            if model_pretrain is not None:
                logging.info("Pretrained model: " + model_pretrain)
                try:
                    nn_r = load_model(PATH_TO_SAVE + FOLD_NN + '/' + model_pretrain, compile=False)
                except Exception as e:
                    logging.error(e)
                    logging.warning("Can't import model " + model_pretrain + ". Train a NN from scratch.")

            # Compile nn model
            nn_r.compile(loss=loss, loss_weights=1, optimizer=opt)

            if not bool_hyper:
                # Fitting the model
                if modelcheck:
                    es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.01, patience=32)
                    mc = ModelCheckpoint(GBM_BURST_DB + 'm_check', monitor='val_loss', mode='min',
                                         verbose=0, save_best_only=True)
                else:
                    es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.01, patience=32,
                                       restore_best_weights=True)

                logging.info("Fitting the model.")
                # callbacks=[es, mc]
                if modelcheck:
                    history = nn_r.fit(X_train, y_train, epochs=epochs, batch_size=bs,
                                       validation_split=0.3, callbacks=[es, mc])
                else:
                    history = nn_r.fit(X_train, y_train, epochs=epochs, batch_size=bs,
                                       validation_split=0.3, callbacks=[es])
            else:
                tuner = kt.Hyperband(self.build_model_hype,
                                     objective='val_loss',
                                     max_epochs=640,
                                     factor=3,
                                     directory='my_dir',
                                     project_name='intro_to_kt')
                stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=64)
                tuner.search(X_train, y_train, epochs=50, validation_split=0.3, batch_size=bs, callbacks=[stop_early])

                # Get the optimal hyperparameters
                best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
                nn_r = tuner.hypermodel.build(best_hps)
                history = nn_r.fit(X_train, y_train, epochs=50, validation_split=0.3)

            # Insert loss result in model name
            loss_test = round(nn_r.evaluate(X_test, y_test), 2)
            name_model = 'model_' + self.start_month + '_' + self.end_month + '_' + str(loss_test)

            # Predict the model
            logging.info("NN predict on train and test set.")
            pred_train = nn_r.predict(X_train)
            pred_test = nn_r.predict(X_test)

            # Median Absolute Error Computation
            mae = MAE(y_train, pred_train)
            logging.info("MAE train : % f" % (mae))
            mae = MAE(y_test, pred_test)
            logging.info("MAE test : % f" % (mae))
            diff_mean = (y_test - pred_test).median().mean()
            logging.info("diff test - pred : % f" % (diff_mean))

            # Compute MAE per each detector and range
            idx = 0
            text_mae = ""
            for i in self.col_range:
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
                logging.info(text_tr + '    ' + text_te + '    ' + test_diff_i + '    ' +\
                             text_tr_m + '    ' + text_te_m + '    ' + test_diff_i_m)
                text_mae += text_tr + '    ' + text_te + '    ' + test_diff_i + '    ' + \
                            text_tr_m + '    ' + text_te_m + '    ' + test_diff_i_m + '\n'
                idx = idx + 1

            # plot training history
            plt.plot(history.history['loss'][4:], label='train')
            plt.plot(history.history['val_loss'][4:], label='test')
            plt.legend()

            logging.info("Saving model with name: " + name_model)
            nn_r.save(PATH_TO_SAVE + FOLD_NN + "/" + name_model + '.h5')
            self.nn_r = nn_r
            # Save figure of performance
            plt.savefig(PATH_TO_SAVE + FOLD_NN + "/" + name_model + '.png')
            # open text file and write mae performance
            text_file = open(PATH_TO_SAVE + FOLD_NN + "/" + name_model + '.txt', "w")
            text_file.write(text_mae)
            text_file.close()

        # If NN already trained load the one with the same months and minimum loss
        else:
            if model_pretrain is not None:
                try:
                    logging.info("Try to load " + model_pretrain)
                    self.nn_r = load_model(PATH_TO_SAVE + FOLD_NN + '/' + model_pretrain, compile=False)
                except Exception as e:
                    logging.warning(e)
                    logging.warning("Can't load model " + model_pretrain)
            else:
                onlyfiles = [f for f in listdir(PATH_TO_SAVE + FOLD_NN) if
                             isfile(join(PATH_TO_SAVE + FOLD_NN, f)) and '.h5' in f and
                             self.start_month in f and self.end_month in f]
                if len(onlyfiles) == 0:
                    logging.error("Model not found. Try to train the model.")
                    raise
                # Sort to take the best 'loss' model
                index_min_loss = int(np.argmin([float(i.split('_')[-1].split('.h5')[0]) for i in onlyfiles]))
                logging.info("Try to load " + onlyfiles[index_min_loss])
                self.nn_r = load_model(PATH_TO_SAVE + FOLD_NN + '/' + onlyfiles[index_min_loss], compile=False)
                pass

    def predict(self, time_to_del: int = 150):
        """
        Predict the bkg for all dataset, save frg and bkg in FOLD_PRED.
        :param time_to_del: int, if True delete time_to_del * 4s before and after SSA.
        :return: None
            df_ori -> original counts for each det_rng + met time
            y_pred -> predict counts for each det_rng + met time
        """
        logging.info('Conversion from met to datetime')
        ts = Time(self.df_data['met'], format='fermi').utc.to_datetime()
        # Clear RAM with garbage collector
        gc.collect()
        # Predict y for all dataset TODO too slow
        pred_x_tot = self.nn_r.predict(self.scaler.transform(self.df_data.loc[:, self.col_selected].astype('float32')))
        gc.collect()

        # # # Generate a dataset for trigger algorithm
        # Original bkg + ssa + met
        df_ori = self.df_data.loc[:, self.col_range].astype('float32').reset_index(drop=True)
        df_ori['met'] = self.df_data['met'].values
        df_ori['timestamp'] = ts
        # Prediction of the bkg
        y_pred = pd.DataFrame(pred_x_tot, columns=self.col_range)
        y_pred['met'] = self.df_data['met'].values
        y_pred['timestamp'] = ts

        if time_to_del > 0:
            logging.info("Delete data near to SSA.")
            # Index where there is a time gap (more than 500 is considered SSA)
            index_saa = np.where((df_ori[['met']].diff() > 500).values)[0]
            # range time to delete. E.g. time_to_del*4 seconds
            max_index = index_saa.max()
            min_index = index_saa.min()
            set_index = set()
            for ind in index_saa:
                set_index = set_index.union(set(range(max(ind - time_to_del, min_index),
                                                      min(ind + time_to_del, max_index))))
            # Set counts in SSA as NaN. The algorithm of triggering will ignore those
            df_ori.loc[set_index, self.col_range] = np.nan
            y_pred.loc[set_index] = np.nan

        # Set zero counts (in frg) to np.nan
        for col in self.col_range:
            df_ori.loc[(df_ori[col] == 0), col] = np.nan  # | (y_pred[col] == 0)
            y_pred.loc[(df_ori[col] == 0), col] = np.nan  # | (y_pred[col] == 0)
        # Check if some prediction are 0
        if (y_pred.loc[:, self.col_range] == 0).any().any():
            logging.error("A prediction count rate is 0. Check the input data.")
            logging.error(str((y_pred.loc[:, self.col_range] == 0).sum()))

        df_ori.reset_index(drop=True, inplace=True)
        y_pred.reset_index(drop=True, inplace=True)

        # Save the data
        logging.info("Save foreground and background in csv files.")
        df_ori.to_csv(PATH_TO_SAVE + FOLD_PRED + "/" + 'frg_' + self.start_month + '_' + self.end_month + '.csv',
                      index=False)
        y_pred.to_csv(PATH_TO_SAVE + FOLD_PRED + "/" + 'bkg_' + self.start_month + '_' + self.end_month + '.csv',
                      index=False)

    def plot(self, time_r=range(0, 10000), orbit_bin=None, det_rng='n1_r1'):
        """
        Methods to plot frb, bkg and residuals.
        :param time_r: index of dataframe to plot the bkg and frg
        :param orbit_bin: int. The number of orbit to bin the lightcurve e.g. if global_bin=10 the time bin of the
                signals is 96*10*60s
        :param det_rng: detector to select in the plot
        :return: None
        """
        # Plot a particular zone and det_rng
        df_ori = pd.read_csv(PATH_TO_SAVE + FOLD_PRED + "/" + 'frg_' + self.start_month + '_' + self.end_month + '.csv')
        y_pred = pd.read_csv(PATH_TO_SAVE + FOLD_PRED + "/" + 'bkg_' + self.start_month + '_' + self.end_month + '.csv')

        # # Plot y_true and y_pred
        # plt.figure()
        # plt.plot(df_ori.loc[:, det_rng], y_pred.loc[:, det_rng], '.', alpha=0.2)
        # plt.plot([0, 2000], [0, 2000], '-')

        if orbit_bin is not None:
            # # Drop na for average counts
            # df_ori = df_ori.dropna(axis=0)
            # y_pred = y_pred.dropna(axis=0)
            # Downsample the signals averaging by orbit_bin orbit slots. 96.5*60s are the seconds of orbit.
            df_ori_downsample = df_ori.groupby(df_ori.met // (96*orbit_bin*60)).mean()
            df_ori_downsample['timestamp'] = df_ori['timestamp'].groupby(df_ori.met // (96*orbit_bin*60)).first()
            y_pred_downsample = y_pred.groupby(df_ori.met // (96*orbit_bin*60)).mean()
            y_pred_downsample['timestamp'] = df_ori_downsample['timestamp']

            # df_ori_downsample = df_ori.groupby(df_ori.index//1000).sum()
            # df_ori_downsample['timestamp'] = df_ori['timestamp'].groupby(df_ori.index//1000).first()
            # y_pred_downsample = y_pred.groupby(y_pred.index//1000).sum()
            # y_pred_downsample['timestamp'] = y_pred['timestamp'].groupby(y_pred.index//1000).first()

            df_ori = df_ori_downsample
            y_pred = y_pred_downsample
            time_r = df_ori.index

        # Plot frg, bkg and residual for det_rng
        with sns.plotting_context("talk"):
            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(40, 60))
            # Remove horizontal space between axes
            fig.subplots_adjust(hspace=0)
            fig.suptitle(det_rng + " " + str(pd.to_datetime(df_ori.loc[time_r, 'timestamp']).iloc[0]))

            # Plot each graph, and manually set the y tick values
            axs[0].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']), df_ori.loc[time_r, det_rng], 'k-.')
            axs[0].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']), y_pred.loc[time_r, det_rng], 'r-')

            # axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
            # axs[0].set_ylim(-1, 1)
            axs[0].set_title('foreground and background')
            axs[0].set_xlabel('time')
            axs[0].set_ylabel('Count Rate')

            axs[1].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']),
                        df_ori.loc[time_r, det_rng] - y_pred.loc[time_r, det_rng], 'k-.')
            axs[1].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']).fillna(method='ffill'),
                        df_ori.loc[time_r, 'met'].fillna(0) * 0, 'k-')
            # axs[1].set_yticks(np.arange(0.1, 1.0, 0.2))
            # axs[1].set_ylim(0, 1)
            axs[1].set_xlabel('time')
            axs[1].set_ylabel('Residuals')

        # TODO to delete
        # plt.plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']), df_ori.loc[time_r, det_rng], '.')
        # plt.plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']), y_pred.loc[time_r, det_rng], '.')

        # Plot y_pred vs y_true
        with sns.plotting_context("talk"):
            fig = plt.figure()
            fig.set_size_inches(24, 12)
            plt.plot(df_ori.loc[time_r, self.col_range], y_pred.loc[time_r, self.col_range], '.', alpha=0.2)
            plt.plot([0, 600], [0, 600], '-')
            plt.xlim([0, 600])
            plt.ylim([0, 600])
            plt.xlabel('True signal')
            plt.ylabel('Predicted signal')
        plt.legend(self.col_range)

    def explain(self, time_r=range(0, 10), det_rng=None):
        """
        Explanation for instances in time range 'time_r'.
        :param time_r: index timestamp.
        :param det_rng: detector and range to explain.
        :return: None
            Plot summary_plot more than one instances.
            Plot waterfall is one instance.
        """
        # Define dataset of background and explanation
        X_back = shap.sample(self.df_data.loc[:, self.col_selected].astype('float32'), nsamples=42, random_state=42)
        X_back_std =  pd.DataFrame(self.scaler.transform(X_back),  columns=self.col_selected)
        X_expl = self.df_data.loc[time_r, self.col_selected].astype('float32')
        X_expl_std = pd.DataFrame(self.scaler.transform(X_expl), columns=self.col_selected)
        # Explainer shap
        e = shap.KernelExplainer(self.nn_r, X_back_std)
        shap_values = e.shap_values(X_expl_std, n_sample=20)
        # Gradient based
        # e = shap.GradientExplainer(self.nn_r, X_back_std)
        # shap_values = e.shap_values(X_expl_std.values)
        if len(time_r) > 1:
            plt.figure()
            shap.summary_plot(shap_values, X_expl_std)
        else:
            if det_rng is not None:
                idx = np.where(np.array(self.col_range) == det_rng)[0][0]
                plt.figure()
                shap.plots._waterfall.waterfall_legacy(e.expected_value[idx], shap_values[idx][0],
                                                       feature_names=self.col_selected)
            else:
                logging.warning("No Detector and range specified.")
                plt.figure()
                shap.summary_plot(shap_values, X_expl_std)
