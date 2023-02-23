import numpy as np
from math import sqrt, log


class Curve:
    '''
    From the original python implementation of
    FOCuS Poisson by Kester Ward (2021). All rights reserved.
    '''

    def __init__(self, k_T, lambda_1, t=0):
        self.a = k_T
        self.b = -lambda_1
        self.t = t

    def __repr__(self):
        return "({:d}, {:.2f}, {:d})".format(self.a, self.b, self.t)

    def evaluate(self, mu):
        return max(self.a * log(mu) + self.b * (mu - 1), 0)

    def update(self, k_T, lambda_1):
        return Curve(self.a + k_T, -self.b + lambda_1, self.t - 1)

    def ymax(self):
        return self.evaluate(self.xmax())

    def xmax(self):
        return -self.a / self.b

    def is_negative(self):
        # returns true if slope at mu=1 is negative (i.e. no evidence for positive change)
        return (self.a + self.b) <= 0

    def dominates(self, other_curve):
        return (self.a + self.b >= other_curve.a + other_curve.b) and (self.a * other_curve.b <= other_curve.a * self.b)


def focus_step(curve_list, k_T, lambda_1):
    '''
    From the original python implementation of
    FOCuS Poisson by Kester Ward (2021). All rights reserved.
    '''
    if not curve_list:  # list is empty
        if k_T <= lambda_1:
            return [], 0., 0
        else:
            updated_c = Curve(k_T, lambda_1, t=-1)
            return [updated_c], updated_c.ymax(), updated_c.t

    else:  # list not empty: go through and prune

        updated_c = curve_list[0].update(k_T, lambda_1)  # check leftmost quadratic separately
        if updated_c.is_negative():  # our leftmost quadratic is negative i.e. we have no quadratics
            return [], 0., 0,
        else:
            new_curve_list = [updated_c]
            global_max = updated_c.ymax()
            time_offset = updated_c.t

            for c in curve_list[1:] + [Curve(0, 0)]:  # add on new quadratic to end of list
                updated_c = c.update(k_T, lambda_1)
                if new_curve_list[-1].dominates(updated_c):
                    break
                else:
                    new_curve_list.append(updated_c)
                    ymax = updated_c.ymax()
                    if ymax > global_max:  # we have a new candidate for global maximum
                        global_max = ymax
                        time_offset = updated_c.t

    return new_curve_list, global_max, time_offset


def set(mu_min=1, t_max: int = 0):
    '''
    :param mu_min: float > 1. kills faint cp
    :param t_max: int > 1. kills old cp
    :return: a function
    '''

    def run(xs, bs):
        '''
        params
        :param xs: counts generator or sequence object
        :param bs: background generator or sequence object
        :return: a list
        '''

        out = []
        out_offset = []
        curve_list = []

        for T, (x_t, lambda_t) in enumerate(zip(xs, bs)):
            if not np.isnan(lambda_t):
                # mu_min and t_max curves cut
                if (
                        curve_list
                    and ((ab_crit and curve_list[0].a <= ab_crit * curve_list[0].b)
                    or  (t_max and curve_list[0].t < t_max))
                ):
                    curve_list = curve_list[1:]

                # main step
                curve_list, global_max, offset = focus_step(curve_list, x_t, lambda_t)
                out.append(sqrt(2*global_max))
                out_offset.append(offset)
            else:
                # expects np.nan when SAA turn-off
                curve_list = [] # reset changepoints
                out.append(np.nan)
                out_offset.append(np.nan)
        return out, out_offset

    assert mu_min >= 1.
    assert t_max >= 0
    ab_crit = (lambda mu_min: (1 - mu_min) / log(mu_min) if mu_min > 1 else None)(mu_min)
    t_max = -t_max

    return run
