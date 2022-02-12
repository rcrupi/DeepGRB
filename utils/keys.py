KDETS = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b')
KRANGES = ('0', '1', '2')


def get_keys(ns=KDETS, rs=KRANGES):
    """
    build lists like ['n1_r0', 'n3_r0']
    :param ns: sequence representing dets
    :param rs: sequence representing ranges
    :return: list of strings
    """
    out = ['n' + str(i) + '_r' + j for i in ns for j in rs]
    return out


def filter_keys(ls, ns, rs=None):
    """
    :param ls: list of string keys
    :param ns: detectors to keep
    :param rs: ranges to keep
    :return: filtered list of strings
    """
    if rs is None:
        rs = ['0', '1', '2']
    index_labels = set([i[1] for i in ls])
    range_labels = set([i[-1] for i in ls])

    out_index = index_labels.intersection(ns)
    out_range = range_labels.intersection(rs)
    return sorted(get_keys(out_index, out_range))
