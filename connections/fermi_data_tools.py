import pandas as pd
from gbm.finder import BurstCatalog, TriggerCatalog
from connections.utils.config import PATH_GRB_TABLE
import logging


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


def df_trigger_catalog(db_path):
    # Define trigger Fermi GBM catalogue trigger
    trigcat = TriggerCatalog()
    df_trigcat = pd.DataFrame(trigcat.get_table())
    # Convert datetime to MET
    print('Conversion from met to datetime')
    df_trigcat['met_time'] = df_trigcat.loc[:, 'time'].apply(lambda x: t.Met(0).from_iso(x.replace(' ', 'T')).met).values
    df_trigcat['met_end_time'] = df_trigcat.loc[:, 'end_time'].apply(lambda x: t.Met(0).from_iso(x.replace(' ', 'T')).met).values
    df_trigcat = df_trigcat[['name', 'met_time', 'met_end_time', 'time', 'end_time', 'trigger_type', 'detector_mask']]
    # Map the detector numbers into symbols
    def map_det_num(list_det):
      list_det_new=[]
      for i in list_det[0]:
        i = int(i)
        if i>=0 and i<10:
          list_det_new.append('n'+str(i))
        if i==10:
          list_det_new.append('na')
        if i==11:
          list_det_new.append('nb')
        if i==12:
          list_det_new.append('b0')
        if i==13:
          list_det_new.append('b1')
      return list_det_new
    # Detector mask now is a list of detectors
    df_trigcat['detector_mask'] = df_trigcat['detector_mask'].apply(lambda x: map_det_num(np.where(np.array(list(x))=='1')))
    # Save triggers table
    df_trigcat.to_csv(db_path)
