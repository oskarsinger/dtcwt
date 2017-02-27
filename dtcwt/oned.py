import numpy as np

import filters

def get_partial_reconstructions(
    Yh, Yl, biorthogonal, qshift):

    prs = []
    Ylz = np.zeros_like(Yl)
    mask = np.zeros((1,len(Yh)))

    for i in xrange(len(Yh)):
        mask = mask * 0
        mask[0,i] = 1
        pr = dtwaveifm(
            Ylz,
            Yh,
            biorthogonal,
            qshift,
            gain_mask=mask)
        
        prs.append(pr)

    Yl_pr = dtwaveifm(
        Yl,
        Yh,
        biorthogonal,
        qshift,
        gain_mask=mask * 0)
    prs.append(Yl_pr)

    return prs

def dtwavexfm(
    X, nlevels, biorthogonal, q_shift):

    (Yl, Yh, Y_scale) = [None] * 3

    (h0a, h0b, h1a, h1b, h0o, h1o) = (
        q_shift['h0a'],
        q_shift['h0b'],
        q_shift['h1a'],
        q_shift['h1b'],
        biorthogonal['h0o'],
        biorthogonal['h1o'])

    L = X.shape

    if L[0] % 2 > 0:
        X = np.vstack([
            X,
            np.ones((1,1)) * np.nan])
        """
        raise ValueError(
            'Size of X must be a multiple of 2!')
        """

    if nlevels == 0:
        return (Yl, Yh, Y_scale)

    Yh = [None] * nlevels
    Y_scale = [None] * nlevels
    j = 1j
    Hi = filters.get_column_filtered(X, h1o)
    Lo = filters.get_column_filtered(X, h0o)
    end = Hi.shape[0]
    Yh[0] = Hi[0:end:2,:] + j * Hi[1:end+1:2,:]
    Y_scale[0] = np.copy(Lo)

    if nlevels >= 2:
        for level in range(1, nlevels):

            if Lo.shape[0] % 4 > 0:
                Lo = np.vstack([
                    np.copy(Lo[0,:]), 
                    np.copy(Lo), 
                    np.copy(Lo[-1,:])])

            Hi = filters.get_column_d_filtered(Lo, h1b, h1a)
            Lo = filters.get_column_d_filtered(Lo, h0b, h0a)
            end = Hi.shape[0]
            Yh[level] = Hi[0:end:2,:] + j * Hi[1:end+1:2,:]
            Y_scale[level] = np.copy(Lo)

    Yl = np.copy(Lo)

    return (Yl, Yh, Y_scale)

def dtwaveifm(
    Yl, Yh, biorthogonal, q_shift, 
    gain_mask=None):

    nlevels = len(Yh)

    if gain_mask is None:
        gain_mask = np.ones((1,nlevels))

    (g0a, g0b, g1a, g1b, g0o, g1o) = (
        q_shift['g0a'],
        q_shift['g0b'],
        q_shift['g1a'],
        q_shift['g1b'],
        biorthogonal['g0o'],
        biorthogonal['g1o'])
        
    Lo = np.copy(Yl)

    for level in reversed(xrange(1,nlevels)): 
        Hi = c2q1d(Yh[level] * gain_mask[:,level])
        Lo_filt = filters.get_column_i_filtered(
            Lo, g0b, g0a)
        Hi_filt = filters.get_column_i_filtered(
            Hi, g1b, g1a)
        Lo = Lo_filt + Hi_filt

        (Lo_n, Lo_p) = Lo.shape
        (Yh_n, Yh_p) = Yh[level-1].shape

        if not Lo_n == 2 * Yh_n:
            Lo = Lo[1:-1,:]
            (Lo_n, Lo_p) = Lo.shape

        if not (Lo_n == 2 * Yh_n and Lo_p == Yh_p):
            raise ValueError(
                'Yh sizes are not valid for DTWAVEIFM')

    Hi = c2q1d(Yh[0] * gain_mask[:,0])

    return filters.get_column_filtered(Lo, g0o) + \
        filters.get_column_filtered(Hi, g1o)

def c2q1d(X):

    (n, p) = X.shape
    z = np.zeros((n*2, p))
    skip = np.arange(0, n*2, 2)
    z[skip,:] = np.real(X)
    z[skip+1,:] = np.imag(X)

    return z
