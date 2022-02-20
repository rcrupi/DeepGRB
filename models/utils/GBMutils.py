# import utils
from connections.utils.config import PATH_TO_SAVE, FOLD_PRED, DB_PATH
import logging
# Standard packages
import pandas as pd
from sqlalchemy import create_engine


def add_trig_gbm_to_frg(start_month, end_month, inter_time=4.096):
    """
    Add trigger events of catalogue GBM as new columns.
    :param start_month: the start month of the FRG file
    :param end_month: the end month of the FRG file
    :param inter_time: the interval time of binning in foreground
    :return: None
    Update the frg.csv file with the new column. If the event is present the row has the name of it.
    """
    # Read foreground file
    df_data = pd.read_csv(PATH_TO_SAVE + FOLD_PRED + "/" + 'frg_' + start_month + '_' + end_month + '.csv')
    # Add column event, 0 if no event, the name of the event otherwise
    df_data['event'] = 0
    # Load trigger events
    logging.info("Reading events already present in GBM calalogue.")
    # Take index of the time where triggers were identified
    # TODO update this table
    # update_gbm_db()
    engine = create_engine('sqlite:////' + DB_PATH + 'GBMdatabase.db')
    gbm_tri = pd.read_sql_table('GBM_TRI', con=engine)
    # select only events in frg timeline
    gbm_tri = gbm_tri.loc[(gbm_tri['met_end_time'] >= df_data['met'].min()) &
                          (gbm_tri['met_time'] <= df_data['met'].max()), :]
    index_tmp = 1
    for _, row in gbm_tri.iterrows():
        if (index_tmp - 1) % 100 == 0:
            logging.info('triggers added: ', index_tmp)
        # Search met time event
        # |----|---(-|----|--)--|----|
        # The start event "(" and end ")" are moved by inter_time
        # |---(-|----|----|----|--)--|
        # So the timestamp met taken are
        # |---(-X----X----X----X--)--|
        df_data.loc[(df_data['met'] >= row['met_time'] - inter_time) &
                    (df_data['met'] <= row['met_end_time'] + inter_time), 'event'] = row['name']
        index_tmp += 1
    logging.info('Overwriting file ' + 'frg_' + start_month + '_' + end_month + '.csv')
    df_data.to_csv(PATH_TO_SAVE + FOLD_PRED + "/" + 'frg_' + start_month + '_' + end_month + '.csv', index=False)


def update_gbm_db():
    pass
