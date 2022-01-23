'''
a compact, functional implementation of FOCuS normal.
requires python 3.8.
'''


def update(q, x_t):
    return (q[0] - 1, q[1] + 2 * x_t)

def dominates(q, p):
    return not ((q[1] < p[1]) or (q[1] / q[0] > p[1] / p[0]))

def ymax(q):
    return - q[1] ** 2 / (4 * q[0])

def qtocp(q):
    return (q[0] and ymax(q) or 0, q[0])

# def focus_rstep(qs, x_t, q):
#     if qs and not dominates(q, p := update(qs[0], x_t)):
#         return [p] + focus_rstep(qs[1:], x_t, p)
#     return [(0, 0.)]

def focus_rstep(cs, x_t, lambda_t, c):
    if cs:
        k = update(cs[0], x_t, lambda_t)
        if not dominates(c, k):
            return [k] + focus_rstep(cs[1:], x_t, lambda_t, k)
        return [(0, 0., 0)]

def focus(X, threshold):
    qs = [(0, 0.)]

    for t, x_t in enumerate(X):
        qs = focus_rstep(qs, x_t, (1, 0.))
        global_max, time_offset = max(map(qtocp, qs))

        if global_max > threshold:
            return global_max, t + time_offset + 1, t

    return 0., t + 1, t