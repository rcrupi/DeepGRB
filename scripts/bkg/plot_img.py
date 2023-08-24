import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PATH_TO_FOLD = "/beegfs/rcrupi/pred"
PATH_TO_SAVE = "/home/rcrupi/Downloads/"
col_range = ['n0_r0', 'n0_r1', 'n0_r2', 'n1_r0', 'n1_r1', 'n1_r2', 'n2_r0',
                          'n2_r1', 'n2_r2', 'n3_r0', 'n3_r1', 'n3_r2', 'n4_r0', 'n4_r1', 'n4_r2',
                          'n5_r0', 'n5_r1', 'n5_r2', 'n6_r0', 'n6_r1', 'n6_r2', 'n7_r0', 'n7_r1',
                          'n7_r2', 'n8_r0', 'n8_r1', 'n8_r2', 'n9_r0', 'n9_r1', 'n9_r2', 'na_r0',
                          'na_r1', 'na_r2', 'nb_r0', 'nb_r1', 'nb_r2']


# # # n4_r1_2019_05_21.png \label{fig:residual}
start_month = "01-2019"
end_month = "07-2019"
orbit_bin = None
det_rng = 'n4_r1'

# Plot a particular zone and det_rng
df_ori = pd.read_csv(PATH_TO_FOLD + "/" + 'frg_' + start_month + '_' + end_month + '_MAE' + '.csv')
y_pred = pd.read_csv(PATH_TO_FOLD + "/" + 'bkg_' + start_month + '_' + end_month + '_MAE' + '.csv')

time_r_min = '2019-05-21 00:00:00'
time_r_max = '2019-05-22 00:00:00'
time_r = df_ori[
    (pd.to_datetime(df_ori.timestamp) >= pd.to_datetime(time_r_min)) &
    (pd.to_datetime(df_ori.timestamp) < pd.to_datetime(time_r_max))
].index



# Plot frg, bkg and residual for det_rng
with sns.plotting_context("talk"):
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(24, 12))
    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)
    fig.suptitle(det_rng + " " + str(pd.to_datetime(df_ori.loc[time_r, 'timestamp']).iloc[0]))

    # Set NaN non nominal count rates
    hhh = df_ori.loc[time_r, det_rng].copy()
    hhh.loc[hhh < 250] = None

    # Plot each graph, and manually set the y tick values
    axs[0].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']), hhh, 'k-.')
    axs[0].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']), y_pred.loc[time_r, det_rng], 'r-')

    # axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
    # axs[0].set_ylim(-1, 1)
    axs[0].set_title('foreground and background')
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('Count Rate')

    axs[1].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']),
                hhh - y_pred.loc[time_r, det_rng],
                'k-.')
    axs[1].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']).fillna(method='ffill'),
                df_ori.loc[time_r, 'met'].fillna(0) * 0, 'k-')
    # axs[1].set_yticks(np.arange(0.1, 1.0, 0.2))
    # axs[1].set_ylim(0, 1)
    axs[1].set_xlabel('time (month-day hour)')
    axs[1].set_ylabel('Residuals')

    axs[2].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']),
                (hhh - y_pred.loc[time_r, det_rng]) / y_pred.loc[time_r, det_rng] * 100, # sqrt
                 'k-.')
    axs[2].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']).fillna(method='ffill'),
                df_ori.loc[time_r, 'met'].fillna(0) * 0, 'k-')
    # axs[1].set_yticks(np.arange(0.1, 1.0, 0.2))
    # axs[1].set_ylim(0, 1)
    axs[2].set_xlabel('time (month-day hour)')
    axs[2].set_ylabel('Residuals %') #
plt.savefig(PATH_TO_SAVE + 'n4_r1_2019_05_21_S.png')

# # # bkg_est.png \label{fig:bkg_est}
# Plot y_pred vs y_true
with sns.plotting_context("talk"):
    fig = plt.figure()
    fig.set_size_inches(16, 10)
    plt.plot(df_ori.loc[:, ['n4_r0', 'n4_r1', 'n4_r2']], y_pred.loc[:,  ['n4_r0', 'n4_r1', 'n4_r2']], '.', alpha=0.2)
    plt.plot([0, 600], [0, 600], '-')
    plt.xlim([0, 600])
    plt.ylim([0, 600])
    plt.xlabel('True signal')
    plt.ylabel('Predicted signal')
    plt.legend(['n4_r0', 'n4_r1', 'n4_r2'])
plt.savefig(PATH_TO_SAVE + 'bkg_est.png')


# # # n5_r0_2020_orbit_1.png \label{fig:solarmin2020_orbit1}
start_month = "01-2020"
end_month = "01-2021"
orbit_bin = 1
det_rng = 'n5_r0'

# Plot a particular zone and det_rng
df_ori = pd.read_csv(PATH_TO_FOLD + "/" + 'frg_' + start_month + '_' + end_month + '.csv')
y_pred = pd.read_csv(PATH_TO_FOLD + "/" + 'bkg_' + start_month + '_' + end_month + '.csv')

time_r = df_ori.index

# # Drop na for average counts
# df_ori = df_ori.dropna(axis=0)
# y_pred = y_pred.dropna(axis=0)
# Downsample the signals averaging by orbit_bin orbit slots. 96.5*60s are the seconds of orbit.
df_ori_downsample = df_ori.groupby(df_ori.met // (96 * orbit_bin * 60)).mean()
df_ori_downsample['timestamp'] = df_ori['timestamp'].groupby(df_ori.met // (96 * orbit_bin * 60)).first()
y_pred_downsample = y_pred.groupby(df_ori.met // (96 * orbit_bin * 60)).mean()
y_pred_downsample['timestamp'] = df_ori_downsample['timestamp']

df_ori = df_ori_downsample
y_pred = y_pred_downsample
time_r = df_ori.index

# Plot frg, bkg and residual for det_rng
with sns.plotting_context("talk"):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(24, 12))
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
    axs[1].set_xlabel('time (year-month)')
    axs[1].set_ylabel('Residuals')
plt.savefig(PATH_TO_SAVE + 'n5_r0_2020_orbit_1.png')

# # # n5_r0_2020_orbit16.png \label{fig:solarmin2020_orbit16}
start_month = "01-2020"
end_month = "01-2021"
orbit_bin = 16
det_rng = 'n5_r0'

# Plot a particular zone and det_rng
df_ori = pd.read_csv(PATH_TO_FOLD + "/" + 'frg_' + start_month + '_' + end_month + '.csv')
y_pred = pd.read_csv(PATH_TO_FOLD + "/" + 'bkg_' + start_month + '_' + end_month + '.csv')

time_r = df_ori.index

# # Drop na for average counts
# df_ori = df_ori.dropna(axis=0)
# y_pred = y_pred.dropna(axis=0)
# Downsample the signals averaging by orbit_bin orbit slots. 96.5*60s are the seconds of orbit.
df_ori_downsample = df_ori.groupby(df_ori.met // (96 * orbit_bin * 60)).mean()
df_ori_downsample['timestamp'] = df_ori['timestamp'].groupby(df_ori.met // (96 * orbit_bin * 60)).first()
y_pred_downsample = y_pred.groupby(df_ori.met // (96 * orbit_bin * 60)).mean()
y_pred_downsample['timestamp'] = df_ori_downsample['timestamp']

df_ori = df_ori_downsample
y_pred = y_pred_downsample
time_r = df_ori.index

# Plot frg, bkg and residual for det_rng
with sns.plotting_context("talk"):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(24, 12))
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
    axs[1].set_xlabel('time (year-month)')
    axs[1].set_ylabel('Residuals')
plt.savefig(PATH_TO_SAVE + 'n5_r0_2020_orbit16.png')

# # # n5_r0_2014_orbit_1_zoom.png \label{fig:solarmaxzoom2014_orbit1}
start_month = "01-2014"
end_month = "01-2015"
orbit_bin = 1
det_rng = 'n5_r0'

# Plot a particular zone and det_rng
df_ori = pd.read_csv(PATH_TO_FOLD + "/" + 'frg_' + start_month + '_' + end_month + '.csv')
y_pred = pd.read_csv(PATH_TO_FOLD + "/" + 'bkg_' + start_month + '_' + end_month + '.csv')

time_r = df_ori.index

# # Drop na for average counts
# df_ori = df_ori.dropna(axis=0)
# y_pred = y_pred.dropna(axis=0)
# Downsample the signals averaging by orbit_bin orbit slots. 96.5*60s are the seconds of orbit.
df_ori_downsample = df_ori.groupby(df_ori.met // (96 * orbit_bin * 60)).mean()
df_ori_downsample['timestamp'] = df_ori['timestamp'].groupby(df_ori.met // (96 * orbit_bin * 60)).first()
y_pred_downsample = y_pred.groupby(df_ori.met // (96 * orbit_bin * 60)).mean()
y_pred_downsample['timestamp'] = df_ori_downsample['timestamp']

df_ori = df_ori_downsample
y_pred = y_pred_downsample
time_r = df_ori.index

# Plot frg, bkg and residual for det_rng
with sns.plotting_context("talk"):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(24, 12))
    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)
    fig.suptitle(det_rng + " " + str(pd.to_datetime(df_ori.loc[time_r, 'timestamp']).iloc[0]))

    # Plot each graph, and manually set the y tick values
    axs[0].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']), df_ori.loc[time_r, det_rng], 'k-.')
    axs[0].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']), y_pred.loc[time_r, det_rng], 'r-')

    # axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
    axs[0].set_ylim(130, 350)
    axs[0].set_title('foreground and background')
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('Count Rate')

    axs[1].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']),
                df_ori.loc[time_r, det_rng] - y_pred.loc[time_r, det_rng], 'k-.')
    axs[1].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']).fillna(method='ffill'),
                df_ori.loc[time_r, 'met'].fillna(0) * 0, 'k-')
    # axs[1].set_yticks(np.arange(0.1, 1.0, 0.2))
    axs[1].set_ylim(-15, 100)
    axs[1].set_xlabel('time (year-month)')
    axs[1].set_ylabel('Residuals')
plt.savefig(PATH_TO_SAVE + 'n5_r0_2014_orbit_1_zoom.png')

# # # n5_r0_2014_orbit_1.png \label{fig:solarmax2014_orbit1}
# Plot frg, bkg and residual for det_rng
with sns.plotting_context("talk"):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(24, 12))
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
    axs[1].set_xlabel('time (year-month)')
    axs[1].set_ylabel('Residuals')
plt.savefig(PATH_TO_SAVE + 'n5_r0_2014_orbit_1.png')

# # # n5_r0_2014_orbit16.png \label{fig:solarmmaxzoom2014_orbit16}
start_month = "01-2014"
end_month = "01-2015"
orbit_bin = 16
det_rng = 'n5_r0'

# Plot a particular zone and det_rng
df_ori = pd.read_csv(PATH_TO_FOLD + "/" + 'frg_' + start_month + '_' + end_month + '.csv')
y_pred = pd.read_csv(PATH_TO_FOLD + "/" + 'bkg_' + start_month + '_' + end_month + '.csv')

time_r = df_ori.index

# # Drop na for average counts
# df_ori = df_ori.dropna(axis=0)
# y_pred = y_pred.dropna(axis=0)
# Downsample the signals averaging by orbit_bin orbit slots. 96.5*60s are the seconds of orbit.
df_ori_downsample = df_ori.groupby(df_ori.met // (96 * orbit_bin * 60)).mean()
df_ori_downsample['timestamp'] = df_ori['timestamp'].groupby(df_ori.met // (96 * orbit_bin * 60)).first()
y_pred_downsample = y_pred.groupby(df_ori.met // (96 * orbit_bin * 60)).mean()
y_pred_downsample['timestamp'] = df_ori_downsample['timestamp']

df_ori = df_ori_downsample
y_pred = y_pred_downsample
time_r = df_ori.index

# Plot frg, bkg and residual for det_rng
with sns.plotting_context("talk"):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(24, 12))
    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)
    fig.suptitle(det_rng + " " + str(pd.to_datetime(df_ori.loc[time_r, 'timestamp']).iloc[0]))

    # Plot each graph, and manually set the y tick values
    axs[0].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']), df_ori.loc[time_r, det_rng], 'k-.')
    axs[0].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']), y_pred.loc[time_r, det_rng], 'r-')

    # axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
    axs[0].set_ylim(180, 260)
    axs[0].set_title('foreground and background')
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('Count Rate')

    axs[1].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']),
                df_ori.loc[time_r, det_rng] - y_pred.loc[time_r, det_rng], 'k-.')
    axs[1].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']).fillna(method='ffill'),
                df_ori.loc[time_r, 'met'].fillna(0) * 0, 'k-')
    # axs[1].set_yticks(np.arange(0.1, 1.0, 0.2))
    axs[1].set_ylim(-5, 22)
    axs[1].set_xlabel('time (year-month)')
    axs[1].set_ylabel('Residuals')
plt.savefig(PATH_TO_SAVE + 'n5_r0_2014_orbit16.png')

# # # GRB 091024
start_month = "09-2009"
end_month = "12-2009"
orbit_bin = None

# Plot a particular zone and det_rng
df_ori = pd.read_csv(PATH_TO_FOLD + "/" + 'frg_' + start_month + '_' + end_month + '.csv')
y_pred = pd.read_csv(PATH_TO_FOLD + "/" + 'bkg_' + start_month + '_' + end_month + '.csv')

for det_rng in ['n0_r0', 'n0_r1', 'n0_r2', 'n6_r0', 'n6_r1', 'n6_r2', 'n8_r0', 'n8_r1', 'n8_r2']:

    time_r_min = 278065500
    time_r_max = 278071000
    time_r = df_ori[(df_ori.met >= time_r_min) & (df_ori.met < time_r_max)].index

    # Plot frg, bkg and residual for det_rng
    with sns.plotting_context("talk"):
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
        # Remove horizontal space between axes
        fig.subplots_adjust(hspace=0)
        fig.suptitle(det_rng + " " + str(pd.to_datetime(df_ori.loc[time_r, 'timestamp']).iloc[0]))

        # Plot each graph, and manually set the y tick values
        axs[0].plot(df_ori.loc[time_r, 'met'], df_ori.loc[time_r, det_rng], 'k-.')
        axs[0].plot(df_ori.loc[time_r, 'met'], y_pred.loc[time_r, det_rng], 'r-')

        # axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
        # axs[0].set_ylim(180, 260)
        axs[0].set_title('foreground and background')
        axs[0].set_xlabel('time')
        axs[0].set_ylabel('Count Rate')

        axs[1].plot(df_ori.loc[time_r, 'met'],
                    df_ori.loc[time_r, det_rng] - y_pred.loc[time_r, det_rng], 'k-.')
        axs[1].plot(df_ori.loc[time_r, 'met'].fillna(method='ffill'),
                    df_ori.loc[time_r, 'met'].fillna(0) * 0, 'k-')
        # axs[1].set_yticks(np.arange(0.1, 1.0, 0.2))
        # axs[1].set_ylim(-5, 22)
        axs[1].set_xlabel('met')
        axs[1].set_ylabel('Residuals')
        plt.savefig(PATH_TO_SAVE + det_rng + '.png')

# # # GRB 190507970
start_month = "01-2019"
end_month = "07-2019"
orbit_bin = None

# Plot a particular zone and det_rng
df_ori = pd.read_csv(PATH_TO_FOLD + "/" + 'frg_' + start_month + '_' + end_month + '_MAE' + '.csv')
y_pred = pd.read_csv(PATH_TO_FOLD + "/" + 'bkg_' + start_month + '_' + end_month + '_MAE' + '.csv')

det_rng = 'n8_r1'
time_r_min = '2019-05-07 22:55:00'
time_r_max = '2019-05-07 23:52:59'
time_r = df_ori[
    (pd.to_datetime(df_ori.timestamp) >= pd.to_datetime(time_r_min)) &
    (pd.to_datetime(df_ori.timestamp) < pd.to_datetime(time_r_max))
].index

# Plot frg, bkg and residual for det_rng
with sns.plotting_context("talk"):
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(24, 12))
    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)
    fig.suptitle(det_rng + " " + str(pd.to_datetime(df_ori.loc[time_r, 'timestamp']).iloc[0]))

    # Set NaN non nominal count rates
    hhh = df_ori.loc[time_r, det_rng].copy()
    hhh.loc[hhh < 250] = None

    # Plot each graph, and manually set the y tick values
    axs[0].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']), hhh, 'k-.')
    axs[0].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']), y_pred.loc[time_r, det_rng], 'r-')

    # axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
    # axs[0].set_ylim(-1, 1)
    axs[0].set_title('foreground and background')
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('Count Rate')

    axs[1].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']),
                hhh - y_pred.loc[time_r, det_rng],
                'k-.')
    axs[1].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']).fillna(method='ffill'),
                df_ori.loc[time_r, 'met'].fillna(0) * 0, 'k-')
    # axs[1].set_yticks(np.arange(0.1, 1.0, 0.2))
    axs[1].set_ylim(-27, 87)
    axs[1].set_xlabel('time (month-day hour)')
    axs[1].set_ylabel('Residuals')

    axs[2].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']),
                (hhh - y_pred.loc[time_r, det_rng]) / y_pred.loc[time_r, det_rng] * 100, # sqrt
                 'k-.')
    axs[2].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']).fillna(method='ffill'),
                df_ori.loc[time_r, 'met'].fillna(0) * 0, 'k-')
    # axs[1].set_yticks(np.arange(0.1, 1.0, 0.2))
    # axs[1].set_ylim(0, 1)
    axs[2].set_xlabel('time (month-day hour)')
    axs[2].set_ylabel('Residuals %') #
plt.savefig(PATH_TO_SAVE + 'residuals_GRB190507970.png')
