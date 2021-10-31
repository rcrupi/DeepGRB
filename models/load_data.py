from connections.fermi_data_tools import df_burst_catalog_raw
from models.utils.config import list_grb_table_col
import logging


def df_burst_catalog(download=False, dropna=True, select_col=list_grb_table_col):
    """
    :param download: boolean condition to donwload online or load the table offline
    :param dropna: boolean condition to keep or not na wors in the dataset
    :param select_col: columns to select
    :return:
    """
    try:
        # Load table GRB GBM raw
        df_grb_raw = df_burst_catalog_raw(download=download)
        if dropna:
            # Drop rows if at least one NaN value
            df_grb_raw = df_grb_raw.dropna(axis=0)
        else:
            df_grb_raw = df_grb_raw.fillna(df_grb_raw.mean())
        # Select the proper columns
        df_grb = df_grb_raw[select_col]
        return df_grb
    except Exception as e:
        logging.error("Can't dropna, select columns in the table GRB GBM.")
        logging.error(e)
