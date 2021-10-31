# import utils
import os
import shutil
# GBM data tools
from gbm.finder import ContinuousFtp
from gbm.data import Ctime, Cspec
from gbm.binning.binned import rebin_by_time
from gbm.data import PosHist
from gbm import coords
# Standard packages
import pandas as pd
import datetime, calendar
from connections.utils.config import PATH_TO_SAVE
data_path = PATH_TO_SAVE + "bkg/"
# Define range of energy
erange = {}
erange['n'] = [(28, 50), (50, 300), (300, 500)]
erange['b'] = [(756, 5025), (5025, 50000)]

grb_top = pd.DataFrame({'id': [], 'tStart': []})
year = 2010
list_month = [11, 12]
for month in list_month:
    num_days = calendar.monthrange(year, month)[1]
    days = [datetime.date(year, month, day).strftime("%y%m%d") for day in range(1, num_days+1)]
    tStart = [datetime.datetime(year, month, day, 12).strftime('%Y-%m-%dT%H:%M:%S.00') for day in range(1, num_days+1)]
    # Dataframe with the list of days
    grb_top = grb_top.append(pd.DataFrame({'id': days, 'tStart': tStart}), ignore_index=True)
print('End list days bkg.')


### Function that operate on cspec or ctime
def fun_lightcurve(dic_data, data_path, file_tmp):
  if '.pha' in file_tmp:
    print('Lightcurve execution.')
    # read a cspec file
    print('Reading file: ', file_tmp)
    # ctime or cspec?
    c_tmp = Cspec.open(data_path + 'tmp/' + file_tmp)
    # integrate over range of energy
    if '_n' in file_tmp:
      type_detector = 'n'
    elif '_b' in file_tmp:
      type_detector = 'b'
    # num_detector = file_tmp.headers['PRIMARY']['DETNAM']
    num_detector = file_tmp[(file_tmp.find(type_detector)+1):(file_tmp.find(type_detector)+2)]
    # rebin the data to 4096 ms resolution
    rebinned_cspec = c_tmp.rebin_time(rebin_by_time, 4.096)
    for idx, erange_tmp in enumerate(erange[type_detector]):
      # integrate over the four range keV
      lightcurve = rebinned_cspec.to_lightcurve(energy_range=erange_tmp)
      # the lightcurve bin centroids and count rates
      dic_data[type_detector+num_detector+'_r'+str(idx)] = lightcurve.rates
    if 'met' not in dic_data.keys():
      # Set the timestamp as the first centroid of the lightcurve
      dic_data['met'] = lightcurve.centroids
    # Remove file if all the data are saved in dic_data
    os.remove(data_path + 'tmp/' + file_tmp)
    return dic_data
  else:
    print('Error. Not a .pha file.')

### Function that operate on poshist file
def fun_poshist(dic_data, file_tmp):
  if 'poshist' in file_tmp:
    print('Poshist execution.')
    # read a poshist file
    print('Reading file: ', file_tmp)
    # open poshist
    p_tmp = PosHist.open(data_path + 'tmp/' + file_tmp)
    # Select only times for the interpolation
    met_ts = dic_data['met']
    time_filter = (met_ts >= p_tmp._times.min()) & (met_ts <= p_tmp._times.max())
    for key in dic_data.keys():
      dic_data[key]=dic_data[key][time_filter]
    met_ts = dic_data['met']
    ### Add columns
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
    ### End add columns
    # Remove file if all the data are saved in dic_data
    os.remove(data_path + 'tmp/' + file_tmp)
    return dic_data
  else:
    print('Error. Not a poshist file.')


# Overwrite csv dataframe
bool_overwrite = False

# Run 4 times the download to be sure that a day is downloaded
for round in [0, 1, 2, 3]:
    # Delete folder tmp of daily data
    if os.path.exists(data_path + 'tmp'):
        shutil.rmtree(data_path + 'tmp')
    # List csv files
    list_csv = [i.split('.')[0] for i in os.listdir(data_path) if '.csv' in i]
    # Cycle for each Burst day
    for _, row in grb_top.iterrows():
        try:
            # Check if dataset csv is already computed
            if row['id'][0:6] not in list_csv and not bool_overwrite:
                # Define what day download
                # ftp_daily = ContinuousFtp(met=row['tStart'])
                print('Initialise connection FTP for time UTC: ', row['tStart'])
                ftp_daily = ContinuousFtp(utc=row['tStart'], gps=None)
                # Download in the folder chosen
                ftp_daily.get_cspec(data_path + 'tmp')
                # Download poshist
                ftp_daily.get_poshist(data_path + 'tmp')
                # Sort list file to have poshist at the end
                list_file = os.listdir(data_path + 'tmp')
                # Initialise the data dictionary
                dic_data = {}
                for file_tmp in [i for i in list_file if '.pha' in i]:
                    # Transform counts data
                    dic_data = fun_lightcurve(dic_data, data_path, file_tmp)
                # Add poshist variables
                file_tmp = [i for i in list_file if 'poshist' in i][0]
                dic_data = fun_poshist(dic_data, file_tmp)
                # Create final dataset
                df_data = pd.DataFrame(dic_data)
                # df_data['y'] = 1*((dic_data['met']>=row['tStart'])&(dic_data['met']<=row['tStop']))
                print('Saving file: ', row['id'][0:6])
                df_data.to_csv(data_path + row['id'][0:6] + '.csv', index=False)
                # Delete folder tmp of daily data
                if os.path.exists(data_path + 'tmp'):
                    shutil.rmtree(data_path + 'tmp')

        except Exception as e:
            print(e)
            print('Error for file: ' + row['id'][0:6])
