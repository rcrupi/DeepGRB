import pandas as pd
import numpy as np
from itertools import groupby
from operator import itemgetter


dets = [
    'n0_r0',
    'n0_r1',
    'n1_r0',
    'n1_r1',
    'n2_r0',
    'n2_r1',
]

KDETS = ('0', '1', '2')
KRANGES = ('0', '1')


def get_keys(ns=KDETS, rs=KRANGES):
    """
    build lists like ['n1_r0', 'n3_r0']
    :param ns: sequence representing dets
    :param rs: sequence representing ranges
    :return: list of strings
    """
    out = ['n' + str(i) + '_r' + j for i in ns for j in rs]
    return out

def fetch_triggers(table, threshold, min_dets_num=2, max_dets_num=3):
    '''
    returns a list of the triggers objects
    from focus fildata.
    :param threshold:
    :return:
    '''
    out = {}
    for i in ['0', '1', '2']:
        table_ni = table[get_keys(ns=[i])]
        out[i] = table_ni[table_ni > threshold].any(axis=1)
    merged_ranges_df = pd.DataFrame(out)
    dets_over_trig = merged_ranges_df[merged_ranges_df == True].count(axis=1)
    data = dets_over_trig[dets_over_trig >= min_dets_num]

    trig_segs = []
    for k, g in groupby(enumerate(data.index), lambda ix: ix[0] - ix[1]):
        tup = tuple(map(itemgetter(1), g))
        start, end = tup[0], tup[-1] + 1 # changed right extrema
        if (dets_over_trig[start:end + 1] < max_dets_num).all(): # <= now is <
            trig_segs.append((start, end))
    return trig_segs


testdataA = np.array([
    [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [5.1, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 3.2, 5.1, 0.0, 0.0, 0.0],
    [5.1, 3.2, 5.1, 0.0, 0.0, 0.0],
    [5.1, 0.0, 5.1, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 5.1, 5.1,  5.1, 5.1],
])

testdata1 = testdataA
pars1 = {
    'threshold': 5,
}
results1 = (
    (6,9),
)

testdata2 = testdataA
pars2 = {
    'threshold': 3,
}
results2 = (
    (6,9),
)

testdata3 = testdataA
pars3 = {
    'threshold': 3,
    'min_dets_num': 1,
    'max_dets_num': 4,
}
results3 = (
    (2,3),
    (4,9),
    (10,11)
)

testdataB = np.array([
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
])

testdata4 = testdataB
pars4 = {
    'threshold': 5,
    'min_dets_num': 2,
    'max_dets_num': 3,
}
results4 = ()

testdata5 = testdataB
pars5 = {
    'threshold': 5,
    'min_dets_num': 1,
}
results5 = (
    (0, 11),
)

testdataC = np.array([
    [np.nan, np.nan, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
])

testdata6 = testdataC
pars6 = {
    'threshold': 5,
    'min_dets_num': 1,
}
results6 = (
    (1, 11),
)

testdata7 = testdataC
pars7 = {
    'threshold': 5,
    'min_dets_num': 2,
}
results7 = ()

testdataD = np.array([
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
    [5.1, 5.1, 0.0, 0.0, 0.0, 0.0],
])

testdata8 = testdataD
pars8 = {
    'threshold': 5,
    'min_dets_num': 1,
}
results8 = (
    (0, 4),
    (5, 11),
)

if __name__ == '__main__':
    df = pd.DataFrame(testdata1, columns = dets)
    results = fetch_triggers(df, **pars1)
    print("test passed: {}".format(tuple(results) == results1))

    df = pd.DataFrame(testdata2, columns = dets)
    results = fetch_triggers(df, **pars2)
    print("test passed: {}".format(tuple(results) == results2))

    df = pd.DataFrame(testdata3, columns = dets)
    results = fetch_triggers(df, **pars3)
    print("test passed: {}".format(tuple(results) == results3))

    df = pd.DataFrame(testdata4, columns = dets)
    results = fetch_triggers(df, **pars4)
    print("test passed: {}".format(tuple(results) == results4))

    df = pd.DataFrame(testdata5, columns = dets)
    results = fetch_triggers(df, **pars5)
    print("test passed: {}".format(tuple(results) == results5))

    df = pd.DataFrame(testdata6, columns = dets)
    results = fetch_triggers(df, **pars6)
    print("test passed: {}".format(tuple(results) == results6))

    df = pd.DataFrame(testdata7, columns = dets)
    results = fetch_triggers(df, **pars7)
    print("test passed: {}".format(tuple(results) == results7))

    df = pd.DataFrame(testdata8, columns = dets)
    results = fetch_triggers(df, **pars8)
    print("test passed: {}".format(tuple(results) == results8))