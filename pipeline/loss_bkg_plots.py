import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PATH_TO_SAVE = "/home/rcrupi/Downloads/"

# # # Quantile regressor
y_pred_q90 = pd.read_csv("/home/rcrupi/Desktop/rcrupi/pred/bkg_03-2019_07-2019_q90.csv")
df_ori = pd.read_csv("/home/rcrupi/Desktop/rcrupi/pred/frg_03-2019_07-2019_q90.csv")
y_pred_q10 = pd.read_csv("/home/rcrupi/Desktop/rcrupi/pred/bkg_03-2019_07-2019_q10.csv")
# frg_q10 = pd.read_csv("/home/rcrupi/Desktop/rcrupi/pred/frg_03-2019_07-2019_q10.csv")
# y_pred = pd.read_csv("/home/rcrupi/Desktop/rcrupi/pred/bkg_03-2019_07-2019_current.csv")
y_pred = pd.read_csv("/home/rcrupi/Desktop/rcrupi/pred/bkg_03-2019_07-2019_classic.csv")


col_range = ['n0_r0', 'n0_r1', 'n0_r2', 'n1_r0', 'n1_r1', 'n1_r2', 'n2_r0',
                          'n2_r1', 'n2_r2', 'n3_r0', 'n3_r1', 'n3_r2', 'n4_r0', 'n4_r1', 'n4_r2',
                          'n5_r0', 'n5_r1', 'n5_r2', 'n6_r0', 'n6_r1', 'n6_r2', 'n7_r0', 'n7_r1',
                          'n7_r2', 'n8_r0', 'n8_r1', 'n8_r2', 'n9_r0', 'n9_r1', 'n9_r2', 'na_r0',
                          'na_r1', 'na_r2', 'nb_r0', 'nb_r1', 'nb_r2']

orbit_bin = None
det_rng = 'n7_r1'

time_r_min = '2019-04-20 21:00:00'
time_r_max = '2019-04-21 00:00:00'
time_r = df_ori[
    (pd.to_datetime(df_ori.timestamp) >= pd.to_datetime(time_r_min)) &
    (pd.to_datetime(df_ori.timestamp) < pd.to_datetime(time_r_max))
].index

# Plot frg, bkg and residual for det_rng
with sns.plotting_context("talk"):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(24, 12))
    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)
    fig.suptitle(det_rng + " " + str(pd.to_datetime(df_ori.loc[time_r, 'timestamp']).iloc[0]))

    # Plot each graph, and manually set the y tick values
    axs[0].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']), df_ori.loc[time_r, det_rng], 'k-.')
    axs[0].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']), y_pred.loc[time_r, det_rng], 'r-')
    axs[0].fill_between(pd.to_datetime(df_ori.loc[time_r, 'timestamp']),
                        y_pred_q10.loc[time_r, det_rng],
                        y_pred_q90.loc[time_r, det_rng], alpha=1)
    # axs[0].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']), y_pred_q10.loc[time_r, det_rng], 'b-')
    # axs[0].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']), y_pred_q90.loc[time_r, det_rng], 'g-')

    # axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
    # axs[0].set_ylim(-1, 1)
    axs[0].set_title('foreground and background')
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('Count Rate')

    axs[1].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']),
                df_ori.loc[time_r, det_rng] - y_pred.loc[time_r, det_rng], 'k-.')
    axs[1].fill_between(pd.to_datetime(df_ori.loc[time_r, 'timestamp']),
                        - y_pred.loc[time_r, det_rng] + y_pred_q90.loc[time_r, det_rng], #+ df_ori.loc[time_r, det_rng],
                        - y_pred.loc[time_r, det_rng] + y_pred_q10.loc[time_r, det_rng] ,#+ df_ori.loc[time_r, det_rng],
                        alpha=1)
    axs[1].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']).fillna(method='ffill'),
                df_ori.loc[time_r, 'met'].fillna(0) * 0, 'k-')

    # axs[1].set_yticks(np.arange(0.1, 1.0, 0.2))
    # axs[1].set_ylim(0, 1)
    axs[1].set_xlabel('time (month-day hour)')
    axs[1].set_ylabel('Residuals')
plt.savefig(PATH_TO_SAVE + 'n7_r1_2019_04_20_quantile.png')


# # # MSE
y_pred_mse = pd.read_csv("/home/rcrupi/Desktop/rcrupi/pred/bkg_03-2019_07-2019_MSE.csv")
df_ori = pd.read_csv("/home/rcrupi/Desktop/rcrupi/pred/frg_03-2019_07-2019_q90.csv")
orbit_bin = None
det_rng = 'n7_r1'

time_r_min = '2019-04-20 21:00:00'
time_r_max = '2019-04-21 00:00:00'
time_r = df_ori[
    (pd.to_datetime(df_ori.timestamp) >= pd.to_datetime(time_r_min)) &
    (pd.to_datetime(df_ori.timestamp) < pd.to_datetime(time_r_max))
].index

# Plot frg, bkg and residual for det_rng
with sns.plotting_context("talk"):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(24, 12))
    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)
    fig.suptitle(det_rng + " " + str(pd.to_datetime(df_ori.loc[time_r, 'timestamp']).iloc[0]))

    # Plot each graph, and manually set the y tick values
    axs[0].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']), df_ori.loc[time_r, det_rng], 'k-.')
    axs[0].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']), y_pred_mse.loc[time_r, det_rng], 'r-')
    # axs[0].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']), y_pred_q10.loc[time_r, det_rng], 'b-')
    # axs[0].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']), y_pred_q90.loc[time_r, det_rng], 'g-')

    # axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
    # axs[0].set_ylim(-1, 1)
    axs[0].set_title('foreground and background')
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('Count Rate')

    axs[1].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']),
                df_ori.loc[time_r, det_rng] - y_pred_mse.loc[time_r, det_rng], 'k-.')
    axs[1].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']).fillna(method='ffill'),
                df_ori.loc[time_r, 'met'].fillna(0) * 0, 'k-')

    # axs[1].set_yticks(np.arange(0.1, 1.0, 0.2))
    # axs[1].set_ylim(0, 1)
    axs[1].set_xlabel('time (month-day hour)')
    axs[1].set_ylabel('Residuals')
plt.savefig(PATH_TO_SAVE + 'n7_r1_2019_04_20_MSE.png')

# # # MAE con eventi
y_pred_mae_eventi = pd.read_csv("/home/rcrupi/Desktop/rcrupi/pred/bkg_03-2019_07-2019_con_eventi.csv")
df_ori = pd.read_csv("/home/rcrupi/Desktop/rcrupi/pred/frg_03-2019_07-2019_q90.csv")
orbit_bin = None
det_rng = 'n7_r1'

time_r_min = '2019-04-20 21:00:00'
time_r_max = '2019-04-21 00:00:00'
time_r = df_ori[
    (pd.to_datetime(df_ori.timestamp) >= pd.to_datetime(time_r_min)) &
    (pd.to_datetime(df_ori.timestamp) < pd.to_datetime(time_r_max))
].index

# Plot frg, bkg and residual for det_rng
with sns.plotting_context("talk"):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(24, 12))
    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)
    fig.suptitle(det_rng + " " + str(pd.to_datetime(df_ori.loc[time_r, 'timestamp']).iloc[0]))

    # Plot each graph, and manually set the y tick values
    axs[0].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']), df_ori.loc[time_r, det_rng], 'k-.')
    axs[0].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']), y_pred_mae_eventi.loc[time_r, det_rng], 'r-')
    # axs[0].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']), y_pred_q10.loc[time_r, det_rng], 'b-')
    # axs[0].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']), y_pred_q90.loc[time_r, det_rng], 'g-')

    # axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
    # axs[0].set_ylim(-1, 1)
    axs[0].set_title('foreground and background')
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('Count Rate')

    axs[1].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']),
                df_ori.loc[time_r, det_rng] - y_pred_mae_eventi.loc[time_r, det_rng], 'k-.')
    axs[1].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']).fillna(method='ffill'),
                df_ori.loc[time_r, 'met'].fillna(0) * 0, 'k-')

    # axs[1].set_yticks(np.arange(0.1, 1.0, 0.2))
    # axs[1].set_ylim(0, 1)
    axs[1].set_xlabel('time (month-day hour)')
    axs[1].set_ylabel('Residuals')
plt.savefig(PATH_TO_SAVE + 'n7_r1_2019_04_20_MAE_con_eventi.png')

# # # MSE con eventi
y_pred_mse_eventi = pd.read_csv("/home/rcrupi/Desktop/rcrupi/pred/bkg_03-2019_07-2019_MSE_event.csv")
df_ori = pd.read_csv("/home/rcrupi/Desktop/rcrupi/pred/frg_03-2019_07-2019_MSE_event.csv")
orbit_bin = None
det_rng = 'n7_r1'

time_r_min = '2019-04-20 21:00:00'
time_r_max = '2019-04-21 00:00:00'
time_r = df_ori[
    (pd.to_datetime(df_ori.timestamp) >= pd.to_datetime(time_r_min)) &
    (pd.to_datetime(df_ori.timestamp) < pd.to_datetime(time_r_max))
].index

# Plot frg, bkg and residual for det_rng
with sns.plotting_context("talk"):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(24, 12))
    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)
    fig.suptitle(det_rng + " " + str(pd.to_datetime(df_ori.loc[time_r, 'timestamp']).iloc[0]))

    # Plot each graph, and manually set the y tick values
    axs[0].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']), df_ori.loc[time_r, det_rng], 'k-.')
    axs[0].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']), y_pred_mse_eventi.loc[time_r, det_rng], 'r-')
    # axs[0].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']), y_pred_q10.loc[time_r, det_rng], 'b-')
    # axs[0].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']), y_pred_q90.loc[time_r, det_rng], 'g-')

    # axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
    # axs[0].set_ylim(-1, 1)
    axs[0].set_title('foreground and background')
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('Count Rate')

    axs[1].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']),
                df_ori.loc[time_r, det_rng] - y_pred_mse_eventi.loc[time_r, det_rng], 'k-.')
    axs[1].plot(pd.to_datetime(df_ori.loc[time_r, 'timestamp']).fillna(method='ffill'),
                df_ori.loc[time_r, 'met'].fillna(0) * 0, 'k-')

    # axs[1].set_yticks(np.arange(0.1, 1.0, 0.2))
    # axs[1].set_ylim(0, 1)
    axs[1].set_xlabel('time (month-day hour)')
    axs[1].set_ylabel('Residuals')
plt.savefig(PATH_TO_SAVE + 'n7_r1_2019_04_20_MSE_con_eventi.png')

