# import utils
import os
import logging
# GBM data tools
from gbm.finder import ContinuousFtp
# Standard packages
import pandas as pd
import datetime, calendar
from dateutil.relativedelta import relativedelta
from connections.utils.config import PATH_TO_SAVE, FOLD_CSPEC_POS


def download_spec(start_month, end_month, bool_overwrite=False):
    """
    This function download in the folder FOLD_CSPEC_POS the cspec and poshist files in the time range speficied in the
     params.
    :param start_month: str, the starting month to consider background (included). E.g. '08-2012' for August 2012.
    :param end_month: str, the ending month to consider background (excluded). E.g. '09-2012' for August 2012.
    :param bool_overwrite: bool, if True overwrite the files
    :return: pandas DataFrame, table of the days downloaded.
    """

    # Initialise dataframe with the list of days
    df_days = pd.DataFrame({'id': [], 'tStart': []})
    # Convert string to datetime.date format
    date_start = datetime.date(int(start_month.split('-')[1]), int(start_month.split('-')[0]), 1)
    date_end = datetime.date(int(end_month.split('-')[1]), int(end_month.split('-')[0]), 1)
    date_tmp = date_start
    # Add each day to download in df_days until end_month is reached
    while date_end > date_tmp:
        year = date_tmp.year
        month = date_tmp.month
        # days in the particular year and month
        num_days = calendar.monthrange(year, month)[1]
        # days is in "%y%m%d" format, tStart in '%Y-%m-%dT%H:%M:%S.00' and it is needed for download. 12:00 am default.
        days = [datetime.date(year, month, day).strftime("%y%m%d") for day in range(1, num_days + 1)]
        tStart = [datetime.datetime(year, month, day, 12).strftime('%Y-%m-%dT%H:%M:%S.00') for day in
                  range(1, num_days + 1)]
        # Dataframe with the list of days
        df_days = df_days.append(pd.DataFrame({'id': days, 'tStart': tStart}), ignore_index=True)
        date_tmp = date_tmp + relativedelta(months=1)
    logging.info('End list days bkg.')

    # Run 4 times the download to be sure that a day is downloaded. Can happen that a download fails.
    for round in [0, 1, 2, 3]:
        logging.info('round #: {}'.format(round))
        # List files, e.g. name: glg_cspec_nb_210929_v00.pha
        # split for '_' and take third position -> 210929
        list_days = [i.split('_')[3] for i in os.listdir(PATH_TO_SAVE + FOLD_CSPEC_POS)]
        # per each day count the number of files
        df_days_count = pd.Series(list_days).value_counts().reset_index()
        df_days_count = df_days_count.rename(columns={'index': 'id', 0: 'num_files'})
        # Select only the days to download
        df_days_count = pd.merge(df_days, df_days_count, how='left', on='id')
        df_days_count = df_days_count.fillna(0)
        # Cycle for each Burst day
        for _, row in df_days_count.iterrows():
            try:
                # Check if files is already downloaded. 15 files are needed.
                if row['num_files'] < 15 or bool_overwrite:
                    # Delete the files that are being downloaded again
                    lst_files_to_delete = [i for i in os.listdir(PATH_TO_SAVE + FOLD_CSPEC_POS) if row['id'] in i]
                    [os.remove(PATH_TO_SAVE + FOLD_CSPEC_POS + "/" + i) for i in lst_files_to_delete]
                    # Define what day download
                    # ftp_daily = ContinuousFtp(met=row['tStart'])
                    logging.info('Initialise connection FTP for time UTC: ' + row['tStart'])
                    ftp_daily = ContinuousFtp(utc=row['tStart'], gps=None)
                    # Download in the folder chosen
                    ftp_daily.get_cspec(PATH_TO_SAVE + FOLD_CSPEC_POS)
                    # Download poshist
                    ftp_daily.get_poshist(PATH_TO_SAVE + FOLD_CSPEC_POS)
            except Exception as e:
                logging.error(e)
                logging.error('Error for file: ' + row['id'][0:6])
    logging.info("End download.")
    return df_days_count
