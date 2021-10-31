# import modules
import os
import numpy as np
import pandas as pd
from pyswarms.single.global_best import GlobalBestPSO
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
# Fermi SkyPlot
from gbm.data import HealPix
from gbm.data import PosHist
from gbm.plot import SkyPlot
from gbm.finder import ContinuousFtp

class localization():
    def __init__(self, list_ra, list_dec, counts_frg, counts_bkg):
        self.list_ra = list_ra
        self.list_dec = list_dec
        self.counts_frg = counts_frg
        self.counts_bkg = counts_bkg
        self.counts = np.maximum(counts_frg - counts_bkg, 0)
        self.data = list(zip(list_ra, list_dec, counts))
        self.res = None
        self.list_pos = None

    # Cosine between vector defined by a=(ra_a, dec_a) and b=(ra_b, dec_b)
    # considering polar coordinates
    @staticmethod
    def vect_cos(a, b):
        cos = np.cos(a[0]) * np.cos(a[1]) * np.cos(b[0]) * np.cos(b[1]) + \
              np.sin(a[0]) * np.cos(a[1]) * np.sin(b[0]) * np.cos(b[1]) + \
              np.sin(a[1]) * np.sin(b[1])
        return np.maximum(cos, 0)

    # Define the loss between the measured counts per each detector and the target
    # counts multiplied by the unkown (ra, dec) on the event
    def loss_position(self, x):
        f = 0
        ra = x[:, 0]
        dec = x[:, 1]
        counts = x[:, 2]
        for row in self.data:
            f += (counts * self.vect_cos((ra, dec), (row[0], row[1])) - row[2]) ** 2
        return f

    def fit(self, iters=1000):
        # instatiate the optimizer
        # dec: [-pi/2, pi/2], ra: [0, pi]
        bounds = ([0, -np.pi / 2, min(self.counts)], [2 * np.pi, np.pi / 2, max(self.counts) * 2])
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        optimizer = GlobalBestPSO(n_particles=100, dimensions=3, options=options, bounds=bounds)

        # now run the optimization
        cost, pos = optimizer.optimize(self.loss_position, iters=iters)
        self.res = pos
        return {'ra': pos[0] / np.pi * 180, 'dec': pos[1] / np.pi * 180, 'res_counts': pos[2]}

    def fit_conf_int(self, iters=1000):
        # Use the previous point as initialiser
        x0 = self.res
        percentage_loop = 0
        np.random.seed(42)
        list_pos = []
        for i in range(0, iters):
            # now run the optimization
            counts = np.maximum(
                np.random.poisson(np.maximum(self.counts_frg.astype(int), 0)) -
                np.random.poisson(np.maximum(self.counts_bkg.astype(int), 0))
                , 0)
            data = list(zip(self.list_ra, self.list_dec, counts))

            def loss_position_singular(x):
                f = 0
                ra = x[0]
                dec = x[1]
                counts = x[2]
                for row in data:
                    f += (counts * self.vect_cos((ra, dec), (row[0], row[1])) - row[2]) ** 2
                return f

            # cost, pos_1 = optimizer.optimize(loss_position, iters=20)
            res = minimize(loss_position_singular, x0, method='L-BFGS-B',
                           bounds=((0, 2 * np.pi), (-np.pi / 2, np.pi / 2),
                                   (min(counts), max(counts) * 2)),
                           options={'gtol': 1e-6, 'disp': False})
            pos_1 = res.x
            # print('ra', pos_1[0]/np.pi*180, 'dec', pos_1[1]/np.pi*180)
            list_pos.append(pos_1[0:2])
            if i / iters >= percentage_loop:
                print("Iteration percentage confidence interval fit: ", int(i / iters * 100))
                percentage_loop += 25 / 100
        self.list_pos = list_pos

        return list_pos

        # # Plot points
        # plt.figure(figsize=(15,10))
        # plt.plot(np.array(list_pos)[:,0]/np.pi*180, np.array(list_pos)[:,1]/np.pi*180, '.', alpha=0.2)
        # plt.plot(x0[0]/np.pi*180, x0[1]/np.pi*180, 'gx', markersize=15)

    def plot(self, ori_pos=[None, None], n_std=1):
        def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
            """
            Create a plot of the covariance confidence ellipse of *x* and *y*.

            Parameters
            ----------
            x, y : array-like, shape (n, )
                Input data.

            ax : matplotlib.axes.Axes
                The axes object to draw the ellipse into.

            n_std : float
                The number of standard deviations to determine the ellipse's radiuses.

            **kwargs
                Forwarded to `~matplotlib.patches.Ellipse`

            Returns
            -------
            matplotlib.patches.Ellipse
            """
            if x.size != y.size:
                raise ValueError("x and y must be the same size")

            cov = np.cov(x, y)
            pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
            # Using a special case to obtain the eigenvalues of this
            # two-dimensionl dataset.
            ell_radius_x = np.sqrt(1 + pearson)
            ell_radius_y = np.sqrt(1 - pearson)
            ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                              facecolor=facecolor, **kwargs)

            # Calculating the stdandard deviation of x from
            # the squareroot of the variance and multiplying
            # with the given number of standard deviations.
            scale_x = np.sqrt(cov[0, 0]) * n_std
            mean_x = np.mean(x)

            # calculating the stdandard deviation of y ...
            scale_y = np.sqrt(cov[1, 1]) * n_std
            mean_y = np.mean(y)

            transf = transforms.Affine2D() \
                .rotate_deg(45) \
                .scale(scale_x, scale_y) \
                .translate(mean_x, mean_y)

            ellipse.set_transform(transf + ax.transData)
            return ax.add_patch(ellipse)

        # Fit MV normal
        mean = np.mean(np.array(self.list_pos) / np.pi * 180, axis=0)
        cov = np.cov(np.array(self.list_pos) / np.pi * 180, rowvar=0)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(np.array(self.list_pos)[:, 0] / np.pi * 180, np.array(self.list_pos)[:, 1] / np.pi * 180, s=0.5)
        # ax.axvline(c='grey', lw=1)
        # ax.axhline(c='grey', lw=1)
        confidence_ellipse((np.array(self.list_pos) / np.pi * 180)[:, 0],
                           (np.array(self.list_pos) / np.pi * 180)[:, 1], ax,
                           n_std=n_std, edgecolor='red')
        ax.scatter(self.res[0] * 180 / np.pi, self.res[1] * 180 / np.pi, c='green', marker='*', s=3)
        ax.scatter(mean[0], mean[1], c='red', marker='+', s=3)
        ax.scatter(ori_pos[0], ori_pos[1], c='black', marker='x', s=3)
        plt.show()

        return mean, cov

# Load catalogue of triggered events
df_trig = pd.read_csv('/beegfs/rcrupi/pred/'+'trigs_table.csv')
# Not in catalogue Fermi
df_trig = df_trig.loc[df_trig['catalog_triggers'].isna()].reset_index(drop=True)
# Dataset xxx_101225.csv xxx_2014.csv xxx_19_01-06.csv
df_frg = pd.read_csv('/beegfs/rcrupi/pred/'+'frg_19_01-06.csv')
df_bkg = pd.read_csv('/beegfs/rcrupi/pred/'+'bkg_19_01-06.csv')
# 101111, 140102, 140112, 190404, 190420
df_event = pd.read_csv('/beegfs/rcrupi/bkg/'+'190420.csv')
# 311194700, 410351700, 411228000, 576076200, 577492400
met_event = 577492400

df_bkg['met'] = df_frg['met'].values
df_frg_bkg = pd.merge(df_event, df_bkg, how='left', on=['met'], suffixes=('_frg', '_bkg'))

col_ra = np.sort([i for i in df_frg_bkg.columns if '_ra' in i and len(i) == 5 and 'n' in i])
col_dec = np.sort([i for i in df_frg_bkg.columns if '_dec' in i and len(i) == 6 and 'n' in i])
# select energy range to 1
col_count_frg = np.sort([i for i in df_frg_bkg.columns if '_frg' in i and 'n' in i and '_r0_' in i])
col_count_bkg = np.sort([i for i in df_frg_bkg.columns if '_bkg' in i and 'n' in i and '_r0_' in i])
# Define columns for residual counts
col_det = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb']
# Calculate the residual
df_frg_bkg[col_det] = df_frg_bkg[col_count_frg].values - df_frg_bkg[col_count_bkg].values

df_frg_bkg_event = df_frg_bkg.loc[(df_frg_bkg['met'] > met_event - 1000) & (df_frg_bkg['met'] < met_event + 1000), col_det]
df_frg_bkg_event['n6'].plot()
max_fin = 0
for i in col_det:
  max_tmp = df_frg_bkg_event.loc[:, i].max()
  if max_tmp>max_fin:
    max_fin=max_tmp
    ind_max=df_frg_bkg_event.loc[:, i].idxmax()
print('The index of the peak is: ', ind_max)

col_filter = range(0, 12)
list_ra = df_frg_bkg.loc[ind_max, np.array(col_ra)[col_filter]].values/180*np.pi
list_dec = df_frg_bkg.loc[ind_max, np.array(col_dec)[col_filter]].values/180*np.pi
counts = np.maximum(df_frg_bkg.loc[ind_max, np.array(col_det)[col_filter]].values, 0)
counts_frg = df_frg_bkg.loc[ind_max, np.array(col_count_frg)[col_filter]].values
counts_bkg = df_frg_bkg.loc[ind_max, np.array(col_count_bkg)[col_filter]].values

loc = localization(list_ra, list_dec, counts_frg, counts_bkg)
res = loc.fit()
print(res)
rnd_res = loc.fit_conf_int(500)
mean, cov = loc.plot()

# initialize the continuous data finder with a time (Fermi MET, UTC, or GPS)
cont_finder = ContinuousFtp(met=met_event)
cont_finder.get_poshist('tmp')
# open a poshist file
poshist = PosHist.open("tmp/"+os.listdir("tmp")[0])
os.remove("tmp/"+os.listdir("tmp")[0])
# initialize plot
skyplot = SkyPlot()
# plot the orientation of the detectors and Earth blockage at our time of interest
skyplot.add_poshist(poshist, trigtime=met_event)
gauss_map = HealPix.from_gaussian(np.round(res['ra']), np.round(res['dec']), 10)
skyplot.add_healpix(gauss_map)
plt.show()

pass
