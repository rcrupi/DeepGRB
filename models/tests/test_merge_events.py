dataA = [(1,4), (5,9), (10, 11), (12, 13), (20, 24), (25, 26)]

data1 = dataA
pars1 = {
    'length': 10
}
results1 = ((1,9), (10, 13), (20, 26))

data2 = dataA
pars2 = {
    'length': 3
}
results2 = ((1,4), (5,9), (10, 11), (12, 13), (20, 24), (25, 26))

data3 = dataA
pars3 = {
    'length': 4
}
results3 = ((1,4), (5,9), (10, 13), (20, 24), (25, 26))

data4 = dataA
pars4 = {
    'length': 666
}
results4 = ((1,26),)

dataB = [(32,56), (57,58), (60, 64), (99,100)]

data5 = dataB
pars5 = {
    'length': 10
}
results5 = ((32, 56), (57, 64), (99, 100))


def merge(data, length = 10):
    out = []
    i = 0
    while i < len(data):
        j = 0
        while (
            (i + j < len(data))
            and (data[i + j][1] - data[i][0] < length)
        ):
            j += 1

        if j == 0:
            out.append((data[i][0], data[i][1]))
            i += 1
        else:
            out.append((data[i][0], data[i + j - 1][1]))
            i = i + j
    return out

if __name__ == '__main__':
    results = merge(data1, **pars1)
    print("test passed: {}".format(tuple(results) == results1))

    results = merge(data2, **pars2)
    print("test passed: {}".format(tuple(results) == results2))

    results = merge(data3, **pars3)
    print("test passed: {}".format(tuple(results) == results3))

    results = merge(data4, **pars4)
    print("test passed: {}".format(tuple(results) == results4))

    results = merge(data5, **pars5)
    print("test passed: {}".format(tuple(results) == results5))