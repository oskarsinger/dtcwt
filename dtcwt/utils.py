import numpy as np

import os

def reflect(X, minx, maxx):

    Y = np.copy(X)
    t = np.nonzero(Y > maxx)
    Y[t] = 2 * maxx - Y[t]

    t = np.nonzero(Y < minx)

    while np.count_nonzero(t) > 0:
        Y[t] = 2 * minx - Y[t]
        t = np.nonzero(Y > maxx)

        if not np.count_nonzero(t) > 0:
            Y[t] = 2 * maxx - Y[t]

        t = np.nonzero(Y < minx)

    return Y - 1

def get_wavelet_basis(wavelet_name):

    path_items = [
        os.path.split(os.path.realpath(__file__))[0],
        os.pardir,
        'constants', 
        wavelet_name + '.csv']
    path = os.path.join(*path_items)

    with open(path) as f:
        titles = [t.strip() 
                  for t in f.readline().split(',')]
        val_lists = {t : [] for t in titles}

        for l in f:
            vals = l.split(',')

            for (t, v) in zip(titles, vals):
                if len(v) > 0:
                    val_lists[t].append(float(v))

    return {t : np.array(vec)[:,np.newaxis]
            for (t, vec) in val_lists.items()}

def get_padded_wavelets(Yh, Yl):

    hi_and_lo = Yh# + [Yl]
    num_rows = hi_and_lo[0].shape[0]
    basis = np.zeros(
        (num_rows, len(hi_and_lo)),
        dtype=complex)
    basis[:,0] = np.copy(hi_and_lo[0][:,0])

    for (i, y) in enumerate(hi_and_lo[1:]):
        interval = int(num_rows / y.shape[0])

        for j in range(y.shape[0]):
            begin = interval * j
            end = begin + interval
            basis[begin:end,i+1] = y[j]

    return basis

def get_sampled_wavelets(Yh, Yl):

    hi_and_lo = Yh + [Yl]
    k = None

    for (i, y) in enumerate(hi_and_lo):
        if y.shape[0] > i:
            k = i+1
        else:
            break

    hi_and_lo = hi_and_lo[:k]
    num_rows = hi_and_lo[-1].shape[0]
    basis = np.zeros(
        (num_rows, k),
        dtype=complex)

    for (i, y) in enumerate(hi_and_lo):
        power = k - i - 1
        sample = np.copy(y[::2**power,0])
        basis[:,i] = sample

    return basis
