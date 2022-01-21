# import utils
from connections.utils.config import PATH_TO_SAVE, FOLD_PRED, FOLD_BKG, DB_PATH, FOLD_NN
import logging
# Standard packages
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
from sqlalchemy import create_engine
import gc
from astropy.time import Time
# from datetime import datetime
# Preprocess
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error as MAE
from sklearn.preprocessing import StandardScaler
# Tensorflow, Keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import tensorflow.keras.losses as losses


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
        for csv_tmp in list_csv:
            # logging.info("Loading: " + csv_tmp)
            df_tmp = pd.read_csv(PATH_TO_SAVE + FOLD_BKG + '/' + csv_tmp)
            df_data = df_data.append(df_tmp, ignore_index=True)
        del df_tmp

        # Filter data within saa
        logging.info("Filtering data when Fermi is in SAA.")
        df_data = df_data.loc[df_data['saa'] == 0, self.col_met + self.col_range + self.col_sat_pos +
                              self.col_det_pos].reset_index(drop=True)
        # TODO Filter zero counts
        logging.info(df_data.head())
        if bool_del_trig:
            logging.info("Deleting events already present in GBM calalogue in Train Set.")
            # Take index of the time where triggers were identified
            engine = create_engine('sqlite:////' + DB_PATH + 'GBMdatabase.db')
            # TODO update this table
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
        logging.info("End prepare data")
        self.df_data = df_data
        self.index_date = index_date

    def train(self, bool_train=True, loss_robust=False, units=4000, epochs=512, lr=0.001, bs=2000, model_pretrain=None):
        """
        Train a neural network to estimate counts of detectors.
        In FOLD_NN is saved: the NN model, train and validation performance during epochs,
         performance per each detector_range counts.
        :param bool_train: boolean, if True the nn is trained otherwise it is loaded in FOLD_NN folder
            (same months but minimum loss).
        :param loss_robust: boolean, if False the loss is the average of MAEs, if True is the max of MAEs.
        :param units: number of nodes in the first and second layer, the third is halved.
        :param epochs: number of epochs of the NN.
        :param lr: learning rate of the NN during training.
        :param bs: batch size of the NN during training.
        :param model_pretrain: string, if founded the pretrained model is loaded e re-trained.
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
            model_1 = Dropout(0.2)(model_1)
            # Second layer
            nn_r = Dense(units, activation='relu')(model_1)
            nn_r = Dropout(0.2)(nn_r)
            # Third layer
            nn_r = Dense(int(units / 2), activation='relu')(nn_r)
            # Fourth (last) layer output
            outputs = Dense(len(self.col_range), activation='relu')(nn_r)
            nn_r = tf.keras.Model(inputs=[nn_input], outputs=outputs)
            # Optimizer
            opt = tf.keras.optimizers.Nadam(learning_rate=lr, beta_1=0.7, beta_2=0.99, epsilon=1e-07)

            if loss_robust:
                # Define Loss as max_i(det_ran_error)
                loss_mae_none = losses.MeanAbsoluteError(reduction=losses.Reduction.NONE)

                def loss_mae(y_true, y_predict):
                    """
                    Take the maximum of the MAE detectors.
                    :param y_true: y target
                    :param y_predict: y predicted by the NN
                    :return: max_i(MAE_i)
                    """
                    a = tf.math.reduce_max(loss_mae_none(y_true, y_predict))  # axis=0
                    return a
            else:
                # Define Loss as average of MAE for each detector_range
                loss_mae = 'mae'

            # Load pretrain model if specified
            if model_pretrain is not None:
                logging.info("Pretrained model: " + model_pretrain)
                try:
                    nn_r = load_model(PATH_TO_SAVE + FOLD_NN + '/' + model_pretrain)
                except Exception as e:
                    logging.error(e)
                    logging.warning("Can't import model " + model_pretrain + ". Train a NN from scratch.")

            # Compile nn model
            nn_r.compile(loss=loss_mae, loss_weights=1, optimizer=opt)

            # Fitting the model
            # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=0.01, patience=32)
            mc = ModelCheckpoint(DB_PATH + 'm_check', monitor='val_loss', mode='min', verbose=0, save_best_only=True)

            logging.info("Fitting the model.")
            # callbacks=[es, mc]
            history = nn_r.fit(X_train, y_train, epochs=epochs, batch_size=bs, validation_split=0.3, callbacks=[mc])
            # Insert loss result in model name
            loss_test = round(nn_r.evaluate(X_test, y_test), 2)
            name_model = 'model_' + self.start_month + '_' + self.end_month + '_' + str(loss_test)
            logging.info("Saving model with name: " + name_model)
            nn_r.save(PATH_TO_SAVE + FOLD_NN + "/" + name_model + '.h5')
            self.nn_r = nn_r

            # Predict the model
            logging.info("NN predict on train and test set.")
            pred_train = nn_r.predict(X_train)
            pred_test = nn_r.predict(X_test)

            # MAE Computation
            mae = MAE(y_train, pred_train)
            logging.info("MAE train : % f" % (mae))
            mae = MAE(y_test, pred_test)
            logging.info("MAE test : % f" % (mae))

            # Compute MAE per each detector and range
            idx = 0
            text_mae = ""
            for i in self.col_range:
                mae_tr = MAE(y_train.iloc[:, idx], pred_train[:, idx])
                mae_te = MAE(y_test.iloc[:, idx], pred_test[:, idx])
                text_tr = "MAE train of " + i + " : %0.3f" % (mae_tr)
                text_te = "MAE test of " + i + " : %0.3f" % (mae_te)
                logging.info(text_tr + '    ' + text_te)
                text_mae += text_tr + '    ' + text_te + '\n'
                idx = idx + 1
            # open text file and write mae performance
            text_file = open(PATH_TO_SAVE + FOLD_NN + "/" + name_model + '.txt', "w")
            text_file.write(text_mae)
            text_file.close()

            # plot training history
            plt.plot(history.history['loss'][4:], label='train')
            plt.plot(history.history['val_loss'][4:], label='test')
            plt.legend()
            plt.savefig(PATH_TO_SAVE + FOLD_NN + "/" + name_model + '.png')

        # If NN already trained load the one with the same months and minimum loss
        else:
            onlyfiles = [f for f in listdir(PATH_TO_SAVE + FOLD_NN) if
                         isfile(join(PATH_TO_SAVE + FOLD_NN, f)) and '.h5' in f and
                         self.start_month in f and self.end_month in f]
            if len(onlyfiles) == 0:
                logging.error("Model not found. Try to train the model.")
                raise
            # Sort to take the best 'loss' model
            index_min_loss = int(np.argmin([float(i.split('_')[-1].split('.h5')[0]) for i in onlyfiles]))
            self.nn_r = load_model(PATH_TO_SAVE + FOLD_NN + '/' + onlyfiles[index_min_loss])

    def predict(self, time_to_del: int = 150):
        """
        Predict the bkg for all dataset, save frg and bkg in FOLD_PRED.
        :param time_to_del: boolean, if True delete time_to_del * 4s before and after SSA.
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
        # if len(pred_x.shape)==2:
        #   pred_x = pred_x.reshape(1,-1)[0]
        y_tot = self.df_data.loc[:, self.col_range].astype('float32')
        gc.collect()

        # # # Generate a dataset for trigger algorithm
        # Original bkg + ssa + met
        df_ori = self.df_data.loc[:, self.col_range].astype('float32')
        df_ori['met'] = self.df_data['met']
        df_ori['timestamp'] = ts
        # Prediction of the bkg
        y_pred = pd.DataFrame(pred_x_tot, columns=y_tot.columns)
        y_pred['met'] = self.df_data['met'].values

        if time_to_del > 0:
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

        df_ori.reset_index(drop=True, inplace=True)
        y_pred.reset_index(drop=True, inplace=True)

        # Save the data
        df_ori.to_csv(PATH_TO_SAVE + FOLD_PRED + "/" + 'frg_' + self.start_month + '_' + self.end_month + '.csv',
                      index=False)
        y_pred.to_csv(PATH_TO_SAVE + FOLD_PRED + "/" + 'bkg_' + self.start_month + '_' + self.end_month + '.csv',
                      index=False)
        pass

    def plot(self, time_r=range(0, 10000), det_rng='n1_r1'):
        df_ori = pd.read_csv(PATH_TO_SAVE + FOLD_PRED + "/" + 'frg_' + self.start_month + '_' + self.end_month + '.csv')
        y_pred = pd.read_csv(PATH_TO_SAVE + FOLD_PRED + "/" + 'bkg_' + self.start_month + '_' + self.end_month + '.csv')
        plt.plot(df_ori.loc[time_r, 'timestamp'], df_ori.loc[time_r, det_rng], '.')
        plt.plot(df_ori.loc[time_r, 'timestamp'], y_pred.loc[time_r, det_rng], '.')
        plt.show()
