import pandas as pd
import numpy as np
from gbm.finder import BurstCatalog, TriggerCatalog
from connections.utils.config import PATH_GRB_TABLE, GBM_BURST_DB, GBM_TRIG_DB
import logging
from gbm.time import Met
import sqlite3


def df_burst_catalog_raw(download=True):
    """
    :param download: boolean condition to donwload online or load the table offline
    :return: the table in DataFrame Pandas format
    """
    try:
        if download:
            # Download GBM table burst
            burstcat = BurstCatalog()
            # Define a dataframe pandas of the GRB GBM table
            df_grb = pd.DataFrame(burstcat.get_table())
            # Save table
            df_grb.to_csv(PATH_GRB_TABLE, index=False)
        else:
            # Load table if already saved
            df_grb = pd.read_csv(PATH_GRB_TABLE)
        return df_grb
    except Exception as e:
        logging.error("Can't save, download or load the table GRB GBM.")
        logging.error(e)


def map_det_num(list_det):
    # Map the detector numbers into symbols
    list_det_new = []
    for i in list_det[0]:
        i = int(i)
        if i >= 0 and i < 10:
            list_det_new.append('n' + str(i))
        if i == 10:
            list_det_new.append('na')
        if i == 11:
            list_det_new.append('nb')
        if i == 12:
            list_det_new.append('b0')
        if i == 13:
            list_det_new.append('b1')
    return list_det_new

def map_det_num_burst(list_det):
    # Map the detector numbers into symbols
    list_det_new = []
    for i in list_det[0]:
        i = int(i)
        if i >= 0 and i < 10:
            list_det_new.append('NAI_0' + str(i))
        if i == 10:
            list_det_new.append('NAI_10')
        if i == 11:
            list_det_new.append('NAI_11')
        if i == 12:
            list_det_new.append('b0')
        if i == 13:
            list_det_new.append('b1')
    return list_det_new


def df_burst_catalog(db_path=GBM_BURST_DB):
    # Define trigger Fermi GBM catalogue burst
    burstcat = BurstCatalog()
    df_grb = pd.DataFrame(burstcat.get_table())
    # Convert datetime to MET
    print('Conversion from met to datetime')
    df_grb['tTrigger'] = df_grb.loc[:, 'trigger_time'].apply(lambda x: Met(0).from_iso(x.replace(' ', 'T')).met).values

    df_grb['id'] = df_grb['trigger_name'].str.slice(2)
    # Detector mask now is a list of detectors
    df_grb['trig_det'] = df_grb['bcat_detector_mask'].apply(
        lambda x: map_det_num_burst(np.where(np.array(list(x)) == '1'))).astype(str)
    df_grb['T90'] = df_grb['t90']
    df_grb['T90_err'] = df_grb['t90_error']
    df_grb['T50'] = df_grb['t50']
    df_grb['T50_err'] = df_grb['t50_error']
    df_grb['tStart'] = df_grb['tTrigger'] + df_grb['t90_start']
    df_grb['tStop'] = df_grb['tStart'] + df_grb['T90']
    df_grb['flux'] = df_grb['fluence']/df_grb['T90']
    df_grb['fluxb'] = df_grb['flux_batse_1024']
    df_grb['fluence_err'] = df_grb['fluence_error']
    df_grb['fluenceb'] = df_grb['fluence_batse']
    df_grb['fluenceb_err'] = df_grb['fluence_batse_error']
    # Select necessary columns
    df_grb = df_grb[['id', 'T90', 'T90_err', 'T50', 'T50_err', 'tStart', 'tStop', 'tTrigger', 'trig_det', 'fluence',
                     'flux']]

    # Save GRB table
    df_grb.to_sql('GBM_GRB', sqlite3.connect(str(db_path)), if_exists='replace')
    # df_grb.to_csv(db_path)


def df_trigger_catalog(db_path=GBM_TRIG_DB):
    # Define trigger Fermi GBM catalogue trigger
    trigcat = TriggerCatalog()
    df_trigcat = pd.DataFrame(trigcat.get_table())
    # Convert datetime to MET
    print('Conversion from met to datetime')
    df_trigcat['met_time'] = df_trigcat.loc[:, 'time'].apply(lambda x: Met(0).from_iso(x.replace(' ', 'T')).met).values
    df_trigcat['met_end_time'] = df_trigcat.loc[:, 'end_time'].apply(lambda x: Met(0).from_iso(x.replace(' ', 'T')).met).values
    df_trigcat = df_trigcat[['name', 'met_time', 'met_end_time', 'time', 'end_time', 'trigger_type', 'detector_mask']]

    # Detector mask now is a list of detectors
    df_trigcat['detector_mask'] = df_trigcat['detector_mask'].apply(lambda x: map_det_num(np.where(np.array(list(x))=='1')))
    # Save triggers table
    df_trigcat.to_csv(db_path)
