import numpy as np

from scipy.signal import convolve2d as conv2
import utils as dtcwtu

# TODO: double check indexing to translate from Matlab to numpy

def get_column_filtered(X, h):

    (n, p) = X.shape
    m = h.shape[0]
    m2 = int(m/2)
    Y = None
    
    if np.count_nonzero(X) > 0:
        xe = dtcwtu.reflect(
            np.arange(1-m2, n+m2+1), 0.5, n+0.5)

        Y = conv2(X[xe,:], h, mode='valid')
    else:
        Y = np.zeros((n+1-(m % 2), p))

    return Y

def get_column_i_filtered(X, ha, hb):

    (n, p) = X.shape

    if n % 2 > 0:
        raise ValueError(
            'No. of rows in X must be multiple of 2!')

    m = ha.shape[0]

    if not m == hb.shape[0]:
        raise ValueError(
            'Lengths of ha and hb must be the same!')

    if m % 2 > 0:
        raise ValueError(
            'Lengths of ha and hb must be even!')

    m2 = int(m/2)
    Y = np.zeros((n*2, p))

    xe = dtcwtu.reflect(
        np.arange(1-m2, n+m2+1), 0.5, n+0.5)
    hao = ha[0:m:2]
    hae = ha[1:m:2]
    hbo = hb[0:m:2]
    hbe = hb[1:m:2]
    begin_s = 0
    end_s = n*2

    if m2 % 2 == 0:

        begin_b = 3
        end_b = n + m
        begin_a = 2
        end_a = n + m - 1

        if np.sum(ha * hb) > 0:
            begin_b = 2
            end_b = n + m - 1
            begin_a = 3
            end_a = n + m

        #print 'X[xe[begin_b-2:end_b-2:2],:]', X[xe[begin_b-2:end_b-2:2],:]
        Y[begin_s:end_s:4,:] = conv2(
            X[xe[begin_b-2:end_b-2:2],:], hae, 'valid')
        #print 'Y[begin_s:end_s:4,:]', Y[begin_s:end_s:4,:]
        Y[begin_s+1:end_s+1:4,:] = conv2(
            X[xe[begin_a-2:end_a-2:2],:], hbe, 'valid')
        #print 'Y[begin_s+1:end_s+1:4,:]', Y[begin_s+1:end_s+1:4,:]
        Y[begin_s+2:end_s+2:4,:] = conv2(
            X[xe[begin_b:end_b:2],:], hao, 'valid')
        #print 'Y[begin_s+2:end_s+2:4,:]', Y[begin_s+2:end_s+2:4,:]
        Y[begin_s+3:end_s+3:4,:] = conv2(
            X[xe[begin_a:end_a:2],:], hbo, 'valid')
        #print 'Y[begin_s+3:end_s+3:4,:]', Y[begin_s+3:end_s+3:4,:]
    else:

        begin_b = 2
        end_b = n + m - 1
        begin_a = 1
        end_a = n + m - 2

        if np.sum(ha * hb) > 0:
            begin_a = 2
            end_a = n + m - 1
            begin_b = 1
            end_b = n + m - 2

        #print 'X[xe[begin_b:end_b:2],:]', X[xe[begin_b:end_b:2],:]
        #print 'hao', hao
        Y[begin_s:end_s:4,:] = conv2(
            X[xe[begin_b:end_b:2],:], hao, 'valid')
        #print 'Y[begin_s:end_s:4,:]', Y[begin_s:end_s:4,:]
        Y[begin_s+1:end_s+1:4,:] = conv2(
            X[xe[begin_a:end_a:2],:], hbo, 'valid')
        #print 'Y[begin_s+1:end_s+1:4,:]', Y[begin_s+1:end_s+1:4,:]
        Y[begin_s+2:end_s+2:4,:] = conv2(
            X[xe[begin_b:end_b:2],:], hae, 'valid')
        #print 'Y[begin_s+2:end_s+2:4,:]', Y[begin_s+2:end_s+2:4,:]
        Y[begin_s+3:end_s+3:4,:] = conv2(
            X[xe[begin_a:end_a:2],:], hbe, 'valid')
        #print 'Y[begin_s+3:end_s+3:4,:]', Y[begin_s+3:end_s+3:4,:]

    return Y

def get_column_d_filtered(X, ha, hb):

    (n, p) = X.shape

    if n % 4 > 0:
        raise ValueError(
            'No. of rows in X must be a multiple of 4!')

    m = ha.shape[0]

    if not m == hb.shape[0]:
        raise ValueError(
            'Lengths of ha and hb must be the same!')

    if m % 2 > 0:
        raise ValueError(
            'Lengths of ha and hb must be even!')

    m2 = int(m/2)

    xe = dtcwtu.reflect(np.arange(1-m,n+m+1), 0.5, n+0.5)

    hao = ha[0:m:2]
    hae = ha[1:m:2]
    hbo = hb[0:m:2]
    hbe = hb[1:m:2]
    begin_t = 5
    end_t = n + 2*m - 2

    n2 = n/2
    Y = np.zeros((n2, p))
    begin2 = 0
    end2 = n2
    begin1 = 1
    end1 = n2+1

    if np.sum(ha * hb) > 0:
        begin2 = 1
        end2 = n2+1
        begin1 = 0
        end1 = n2

    """
    print 'Y[begin1:end1:2,:]', Y[begin1:end1:2,:]
    print 'xe[begin_t-1:end_t-1:4]', xe[begin_t-1:end_t-1:4]
    print 'xe[begin_t-3:end_t-3:4]', xe[begin_t-3:end_t-3:4]
    print 'X[xe[begin_t-1:end_t-1:4],:]', X[xe[begin_t-1:end_t-1:4],:]
    print 'X[xe[begin_t-3:end_t-3:4],:]', X[xe[begin_t-3:end_t-3:4],:]
    """
    Y[begin1:end1:2,:] = conv2(X[xe[begin_t-1:end_t-1:4],:], hao, 'valid') + \
            conv2(X[xe[begin_t-3:end_t-3:4],:], hae, 'valid')
    Y[begin2:end2:2,:] = conv2(X[xe[begin_t:end_t:4],:], hbo, 'valid') + \
            conv2(X[xe[begin_t-2:end_t-2:4],:], hbe, 'valid')

    return Y
