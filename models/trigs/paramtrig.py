'''
Credits Giuseppe Dilillo.
'''

from functools import reduce
from math import log, sqrt


def sign(n, b):
    if n > b:
        return n * log(n / b) - (n - b)
    else:
        return 0


# a min function which support None as infinity
nmin = (lambda a, b: min(map((lambda x: float("inf") if x is None else x), (a, b))))


def set(threshold, bg_len, fg_len, hs, gs):
    def run(X):
        obsbuf = []  # observation buffer
        bkg_rate = 0

        for T, X_T in enumerate(X):
            global_max = 0
            time_offset = 0
            obsbuf.append(X_T)

            if T >= buflen:
                bkg_rate += obsbuf[bg_len] / bg_len
                bkg_rate -= obsbuf.pop(0) / bg_len
                schedule = allchecks
            elif T >= bg_len:
                schedule = [(h, g) for (h, g) in zip(hs, gs) if h <= T - bg_len + 1]
            else:
                bkg_rate += obsbuf[-1] / bg_len
                schedule = []

            for h, g in schedule:
                if (T + 1) % h == g:
                    S = sign(sum(obsbuf[-h:]), bkg_rate * h)
                    if S > global_max:
                        global_max = S
                        time_offset = -h

            if global_max > threshold:
                return sqrt(2 * global_max), T + time_offset + 1, T

        return 0, T + 1, T  # no change found by end of signal

    # check all gs are smaller than respective hs
    assert reduce((lambda x, y: x * y), [g < h for (h, g) in zip(hs, gs)])
    assert len(hs) == len(gs)
    buflen = fg_len + bg_len
    allchecks = [(h, g) for (h, g) in zip(hs, gs)]
    threshold = threshold ** 2 / 2

    return run


set_gbm = (lambda threshold: set(threshold,
                                 bg_len=1062,
                                 fg_len=250,
                                 hs=[1, 2, 2, 4, 4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256],
                                 gs=[0, 0, 1, 0, 2, 0, 4, 0, 8, 0, 16, 0, 32, 0, 64, 0, 128])
           )
