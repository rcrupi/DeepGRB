# import utils
import os
import logging
from joblib import Parallel, delayed
# GBM data tools
from gbm.data import Ctime, Cspec
from gbm.binning.binned import rebin_by_time
from gbm.data import PosHist
from gbm import coords
# Standard packages
import numpy as np
import pandas as pd
from connections.utils.config import PATH_TO_SAVE, FOLD_CSPEC_POS, FOLD_BKG


def build_table(df_days, erange, bool_overwrite=False, bool_parallel=False):
    """
    :param df_days: pandas DataFrame, table of the days downloaded in FOLD_BKG folder.
    :param erange: dict, dictionary of list of energy range for NaI and Bi detectors.
        E.g. {'n': [(28, 50), (50, 300), (300, 500)], 'b': [(756, 5025), (5025, 50000)]}
    :param bool_overwrite: bool, if True overwrite the tables (csv files).
    :param bool_parallel: choose if use all CPU for computing the lightcurve (rebinning phase is long).
    :return:
    """
    logging.info("Begin build table (csv files).")
    for _, row in df_days.iterrows():
        try:
            # Sort list file to have cspec + poshist in FOLD_CSPEC_POS
            list_file = os.listdir(PATH_TO_SAVE + FOLD_CSPEC_POS)
            # List of csv of background: lightcurve + satellate features
            list_csv = os.listdir(PATH_TO_SAVE + FOLD_BKG)
            if (row['id'][0:6]+'.csv' not in list_csv) or bool_overwrite:
                # Initialise the data dictionary
                dic_data = {}
                list_pha = [i for i in list_file if '.pha' in i and row['id'][0:6] in i]
                list_pha = np.sort(list_pha)
                # If .pha files are less than 12 NaI + 2 Bi don't proceed
                if len(list_pha) < 14:
                    logging.warning('Not enough detectors (.pha file) in day: ' + row['id'][0:6])
                    continue

                if bool_parallel:
                    # Define the generator for Parallel
                    def fun_lightcurve_param(file_tmp):
                        print('Processing file: ' + file_tmp)
                        res = fun_lightcurve(dic_data={}, file_tmp=file_tmp, erange=erange)
                        print('End processing file: ' + file_tmp)
                        return res
                    # Parallelize
                    results = Parallel(n_jobs=-1, verbose=1)(delayed(fun_lightcurve_param)(file_tmp)
                                                             for file_tmp in list_pha)
                    # Build dic_data inserting each detector_range values and met timestamp
                    for res_dec_i in results:
                        name_dets_i = [i for i in list(res_dec_i.keys()) if i != 'met']
                        # Add detector_rage in dic_data
                        for det_tmp in name_dets_i:
                            dic_data[det_tmp] = res_dec_i[det_tmp]
                        # Add met timestamp if not present
                        if 'met' not in list(dic_data.keys()):
                            dic_data['met'] = res_dec_i['met']
                else:
                    for file_tmp in list_pha:
                        # Transform counts data
                        dic_data = fun_lightcurve(dic_data, file_tmp, erange)

                # Add poshist variables
                file_pos = [i for i in list_file if 'poshist' in i and row['id'][0:6] in i]
                if len(file_pos) > 0:
                    file_pos = np.sort(file_pos)[::-1]
                    file_pos = file_pos[0]
                else:
                    logging.warning('Not poshist file in day: ' + row['id'][0:6])
                    continue
                dic_data = fun_poshist(dic_data, file_pos)
                # Create final dataset
                df_data = pd.DataFrame(dic_data)
                if df_data.isna().sum().sum() > 0:
                    logging.warning('NaN values in csv table.')
                    logging.warning(df_data.isna().sum())
                logging.info('Saving file: ' + row['id'][0:6])
                df_data.to_csv(PATH_TO_SAVE + FOLD_BKG + '/' + row['id'][0:6] + '.csv', index=False)

        except Exception as e:
            logging.error(e)
            logging.error('Error for file: ' + row['id'][0:6])
    logging.info("End preprocess csv files.")


def fun_lightcurve(dic_data, file_tmp, erange):
    """
    Function that operate on cspec or ctime. Return the lightcurve of file_tmp in the energy range of erange.
    :param dic_data: dictionary of data counts (lightcurve) of the detectors in various energy range.
    :param file_tmp: the name of the daily file .pha.
    :param erange: dict, dictionary of list of energy range for NaI and Bi detectors.
    :return:
    """
    if '.pha' in file_tmp:
        logging.info('Lightcurve execution.')
        # read a cspec file
        logging.info('Reading file: ' + file_tmp)
        # ctime or cspec?
        c_tmp = Cspec.open(PATH_TO_SAVE + FOLD_CSPEC_POS + '/' + file_tmp)
        # integrate over range of energy
        if '_n' in file_tmp:
            type_detector = 'n'
        elif '_b' in file_tmp:
            type_detector = 'b'
        else:
            logging.error('Error. NaI or Bi detector if file.')
            raise
        # num_detector = file_tmp.headers['PRIMARY']['DETNAM']
        num_detector = file_tmp[(file_tmp.find(type_detector)+1):(file_tmp.find(type_detector)+2)]
        # rebin the data to 4096 ms resolution
        logging.info('Start binning phase')
        rebinned_cspec = c_tmp.rebin_time(rebin_by_time, 4.096)
        logging.info('End binning phase')
        lightcurve = None
        for idx, erange_tmp in enumerate(erange[type_detector]):
            # integrate over the four range keV
            lightcurve = rebinned_cspec.to_lightcurve(energy_range=erange_tmp)
            # the lightcurve bin centroids and count rates
            dic_data[type_detector+num_detector+'_r'+str(idx)] = lightcurve.rates
        if 'met' not in dic_data.keys():
            # Set the timestamp as the first centroid of the lightcurve
            if lightcurve is not None:
                dic_data['met'] = lightcurve.centroids
            else:
                logging.error('Lightcurve not computed correctly.')
                raise
        # Remove file if all the data are saved in dic_data
        # os.remove(PATH_TO_SAVE + FOLD_CSPEC_POS + '/' + file_tmp)
        return dic_data
    else:
        logging.error('Error. Not a .pha file.')
        raise


def fun_poshist(dic_data, file_tmp):
    """
    Function that operate on poshist file
    :param dic_data:
    :param file_tmp:
    :return:
    """
    if 'poshist' in file_tmp:
        logging.info('Poshist execution.')
        # read a poshist file
        logging.info('Reading file: ' + file_tmp)
        # open poshist
        p_tmp = PosHist.open(PATH_TO_SAVE + FOLD_CSPEC_POS + '/' + file_tmp)
        # Select only times for the interpolation
        met_ts = dic_data['met']
        time_filter = (met_ts >= p_tmp._times.min()) & (met_ts <= p_tmp._times.max())
        for key in dic_data.keys():
            dic_data[key] = dic_data[key][time_filter]
        met_ts = dic_data['met']
        # # # Add feature columns
        # TODO average the position over 4 seconds
        # Position and rotation
        var_tmp = p_tmp.get_eic(met_ts)
        dic_data['pos_x'] = var_tmp[0]
        dic_data['pos_y'] = var_tmp[1]
        dic_data['pos_z'] = var_tmp[2]
        var_tmp = p_tmp.get_quaternions(met_ts)
        dic_data['a'] = var_tmp[0]
        dic_data['b'] = var_tmp[1]
        dic_data['c'] = var_tmp[2]
        dic_data['d'] = var_tmp[3]
        dic_data['lat'] = p_tmp.get_latitude(met_ts)
        dic_data['lon'] = p_tmp.get_longitude(met_ts)
        dic_data['alt'] = p_tmp.get_altitude(met_ts)
        # Velocity
        var_tmp = p_tmp.get_velocity(met_ts)
        dic_data['vx'] = var_tmp[0]
        dic_data['vy'] = var_tmp[1]
        dic_data['vz'] = var_tmp[2]
        var_tmp = p_tmp.get_angular_velocity(met_ts)
        dic_data['w1'] = var_tmp[0]
        dic_data['w2'] = var_tmp[1]
        dic_data['w3'] = var_tmp[2]
        # Sun and Earth visibility
        dic_data['sun_vis'] = p_tmp.get_sun_visibility(met_ts)
        var_tmp = coords.get_sun_loc(met_ts)
        dic_data['sun_ra'] = var_tmp[0]
        dic_data['sun_dec'] = var_tmp[1]
        dic_data['earth_r'] = p_tmp.get_earth_radius(met_ts)
        var_tmp = p_tmp.get_geocenter_radec(met_ts)
        dic_data['earth_ra'] = var_tmp[0]
        dic_data['earth_dec'] = var_tmp[1]
        # Detectors pointing and visibility
        for det_name in ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb', 'b0', 'b1']:
            # Equatorial pointing for each detector
            var_tmp = p_tmp.detector_pointing(det_name, met_ts)
            dic_data[det_name + '_' + 'ra'] = var_tmp[0]
            dic_data[det_name + '_' + 'dec'] = var_tmp[1]
            # Obscured by earth
            dic_data[det_name + '_' + 'vis'] = p_tmp.location_visible(var_tmp[0], var_tmp[1], met_ts)
        # Magnetic field
        dic_data['saa'] = p_tmp.get_saa_passage(met_ts)
        dic_data['l'] = p_tmp.get_mcilwain_l(met_ts)
        # # # End add columns
        # Remove file if all the data are saved in dic_data
        # os.remove(PATH_TO_SAVE + FOLD_CSPEC_POS + '/' + file_tmp)
        return dic_data
    else:
        logging.error('Error. Not a poshist file.')
