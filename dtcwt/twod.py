import numpy as np

import filters

def dtwavexfm2(
    X, 
    nlevels, 
    biorthogonal, 
    q_shift):

    (Yl, Yh, Y_scale) = [None]*3

    (h0a, h0b, h1a, h1b, h0o, h1o) = (
        q_shift['h0a'],
        q_shift['h0b'],
        q_shift['h1a'],
        q_shift['h1b'],
        biorthogonal['h0o'],
        biorthogonal['h1o'])

    original_size = X.shape

    if X.ndim >= 3:
        raise ValueError(
            'Input must have at most 2 modes.')

    initial_row_extend = 0
    initial_col_extend = 0

    if original_size[0] % 2 > 0:
        X = np.vstack([X, X[:,-1]]).T
        initial_col_extend = 1

    extended_size = X.shape

    if nlevels == 0:
        return (Yl, Yh, Y_scale)

    Yh = [None] * nlevels
    Y_scale = [None] * nlevels
    S = None

    if nlevels >= 1:

        # Do odd top-level filters on cols
        Lo = filters.get_column_filtered(X, h0o).T
        Hi = filters.get_column_filtered(X, h1o).T

        # Do odd top-level filters on rows
        LoLo = filters.get_column_filtered(Lo, h0o).T
        Yh[0] = np.zeros([i/2 for i in LoLo.shape]+[6])
        Yh[0][:,:,[0,5]] = q2c(filters.get_column_filtered(Hi, h0o).T)
        Yh[0][:,:,[2,3]] = q2c(filters.get_column_filtered(Lo, h1o).T)
        Yh[0][:,:,[1,4]] = q2c(filters.get_column_filtered(Hi, h1o).T)
        S = np.hstack([np.array(LoLo.shape), S])
        Y_scale[0] = np.copy(LoLo)

    if nlevels >= 2:
        for level in xrange(1, nlevels):
            (n, p) = LoLo.shape

            # Extend by 2 rows if no. of rows of LoLo are divisable by 4
            if n % 4 > 0:
                LoLo = np.hstack([LoLo[:,0], LoLo, LoLo[-1,:]])

            # Extend by 2 cols of no. of cols of LoLo are divisable by 4
            if p % 4 > 0:
                LoLo = np.vstack([LoLo[:,0], LoLo, LoLo[:,-1]]).T

            print LoLo.shape

            # Do even Qshift filters on rows
            Lo = filters.get_column_d_filtered(LoLo, h0b, h0a).T
            Hi = filters.get_column_d_filtered(LoLo, h1b, h1a).T

            # Do even Qshift filters on columns
            LoLo = filters.get_column_d_filtered(Lo, h0b, h0a).T
            print LoLo.shape
            (n, p) = LoLo.shape
            Yh[level] = np.zeros((n/2, p/2, 6))
            Yh[level][:,:,[0,5]] = q2c(
                filters.get_column_d_filtered(Hi,h0b,h0a).T)
            Yh[level][:,:,[2,3]] = q2c(
                filters.get_column_d_filtered(Lo,h1b,h1a).T)
            Yh[level][:,:,[1,4]] = q2c(
                filters.get_column_d_filtered(Hi,h1b,h1a).T)
            S = np.hstack([np.array(LoLo.shape), S])
            Y_scale[level] = np.copy(LoLo)

    Yl = np.copy(LoLo)

    return (Yl, Yh, Y_scale)

def dtwaveifm2(
    Yl, 
    Yh, 
    biorthogonal, 
    q_shift, 
    gain_mask=None):

    a = Yh.shape[0]

    if gain_mask is None:
        gain_mask = np.ones(6, a)

    (g0a, g0b, g1a, g1b, g0o, g1o) = (
        q_shift['g0a'],
        q_shift['g0b'],
        q_shift['g1a'],
        q_shift['g1b'],
        biorthogonal['g0o'],
        biorthogonal['g1o'])
        
    current_level = a - 1;

    Z = np.copy(Yl)
    get_c2q = lambda idxs,lvl: c2q(Yh, gain_mask, idxs, lvl)

    while current_level >= 1:

        lh = get_c2q(np.array([0,5]), current_level)
        hl = get_c2q(np.array([2,3]), current_level)
        hh = get_c2q(np.array([1,4]), current_level)

        # Do even Qshift filters on columns.
        y1 = filters.get_column_i_filtered(Z, g0b, g0a) + \
            filters.get_column_i_filtered(lh, g1b, g1a)
        y2 = filters.get_column_i_filtered(hl, g0b, g0a) + \
            filters.get_column_i_filtered(hh, g1b, g1a)

        # Do even Qshift filters on rows.
        Z = (filters.get_column_i_filtered(y1.T, g0b, g0a) +
            filters.get_column_i_filtered(y2.T, g1b, g1a)).T

        # Check size of Z and crop as required
        (n, p) = Z.shape
        S = [2 * i for i in Yh[current_level-1].shape]

        if not n == S[0]:
            Z = Z[1:n-1,:]

        if not p == S[1]:
            Z = Z[:,1:p-1]

        not_equal = [not i == j
                     for (i,j) in zip(Z.shape, S[0:3])]

        if any(not_equal):
            raise ValueError(
                'Sizes of subbands are not valid for DTWAVEIFM2!')

        current_level -= 1

    lh = get_c2q(np.array([0,5]), 0)
    hl = get_c2q(np.array([2,3]), 0)
    hh = get_c2q(np.array([1,4]), 0)

    y1 = filters.get_column_filtered(Z, g0o) + \
        filters.get_column_filtered(lh, g1o)
    y2 = filters.get_column_filtered(hl, g0o) + \
        filters.get_column_filtered(hh, g1o)
    Z = (filters.get_column_filtered(y1.T, g0o) +
        filters.get_column_filtered(y2.T, g1o)).T

    return Z

def q2c(y):

    sy = y.shape
    t1 = np.arange(0, sy[0], 2)
    t2 = np.arange(0, sy[1], 2)
    j2 = np.array([0.5**(0.5), 1j*0.5**(0.5)])
    p = (y[0:sy[0]:2, 0:sy[1]:2]*j2[0] + 
        y[0:sy[0]:2, 1:sy[1]+1:2]*j2[1])[:,:,np.newaxis]
    q = (y[1:sy[0]+1:2, 1:sy[1]+1:2]*j2[0] - 
        y[1:sy[0]+1:2, 0:sy[1]:2]*j2[1])[:,:,np.newaxis]

    return np.concatenate([p-q,p+q], axis=2)

def c2q(
    Yh, 
    gain_mask, 
    indexes, 
    level):

    w = Yh[level][:,:,indexes]
    gain = gain_mask[indexes,level]

    sw = w.shape
    x = np.zeros((2*sw[0], 2*sw[1]))
    nonzeros = [np.count_nonzero(a)
                for a in [w, gain]]

    if sum(nonzeros) > 0:
        sc = 0.5**(0.5) * gain
        P = w[:,:,0] * sc[0] + w[:,:,1] * sc[1]
        Q = w[:,:,0] * sc[0] - w[:,:,1] * sc[1]

        t1 = np.arange(0, x.shape[0], 2)
        t2 = np.arange(0, x.shape[1], 2)

        x[t1, t2] = np.real(P)
        x[t1, t2+1] = np.imag(P)
        x[t1+1, t2] = np.image(Q)
        x[t1+1, t2+1] = -np.real(Q)

    return x
