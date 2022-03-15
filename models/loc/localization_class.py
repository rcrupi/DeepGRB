import numpy as np
import pandas as pd
from pyswarms.single.global_best import GlobalBestPSO
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


class localization():
    def __init__(self, list_ra, list_dec, counts_frg, counts_bkg):
        self.list_ra = list_ra
        self.list_dec = list_dec
        self.counts_frg = counts_frg
        self.counts_bkg = counts_bkg
        self.counts = np.maximum(counts_frg - counts_bkg, 0)
        self.res = None
        self.list_pos = None
        if list_ra.shape[0] == list_dec.shape[0] == counts_bkg.shape[0] == counts_frg.shape[0]:
            self.dim = list_ra.shape[0]
        else:
            print('Error, dimensions not matched.')

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
    def loss_position(self, data, x):
        f = 0
        ra = x[:, 0]
        dec = x[:, 1]
        counts = x[:, 2]
        for row in data:
            f += (counts * self.vect_cos((ra, dec), (row[0], row[1])) - row[2]) ** 2
        return f

    def _fit_core(self, list_ra, list_dec, counts, iters=1000):
        # instatiate the optimizer
        # dec: [-pi/2, pi/2], ra: [0, pi]
        bounds = ([0, -np.pi / 2, min(counts)], [2 * np.pi, np.pi / 2, max(counts) * 2])
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        optimizer = GlobalBestPSO(n_particles=100, dimensions=3, options=options, bounds=bounds)
        data = list(zip(list_ra, list_dec, counts))
        def loss_position_data(x):
            return self.loss_position(data, x)
        # now run the optimization
        cost, pos = optimizer.optimize(loss_position_data, iters=iters)
        return {'ra': pos[0], 'dec': pos[1], 'res_counts': pos[2]}

    def fit(self, iters=1000):
        list_pos = pd.DataFrame()
        for i in range(0, self.dim):
            list_pos = list_pos.append(
                self._fit_core(self.list_ra[i], self.list_dec[i], self.counts[i], iters=iters),
                ignore_index=True)
        self.res = list_pos.loc[:, ['ra', 'dec', 'res_counts']].median().values
        self.list_pos = list_pos.loc[:, ['ra', 'dec']].values
        return {'ra': self.res[0] / np.pi * 180, 'dec': self.res[1] / np.pi * 180, 'res_counts': self.res[2]}

    def fit_conf_int(self, iters=1000):
        if self.dim > 1:
            print('Already computed, call plot method.')
            return None
        # Use the previous point as initialiser
        x0 = self.res
        percentage_loop = 0
        np.random.seed(42)
        list_pos = []
        for i in range(0, iters):
            # now run the optimization
            counts = np.maximum(
                np.random.poisson(np.maximum(self.counts_frg.astype(int)[0], 0)) -
                np.random.poisson(np.maximum(self.counts_bkg.astype(int)[0], 0))
                , 0)
            data = list(zip(self.list_ra[0], self.list_dec[0], counts))

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
                           # dec: [-pi/2, pi/2], ra: [0, pi]
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

    def plot(self, plot_show=False, ori_pos=[None, None], n_std=1):
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
        if plot_show:
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
