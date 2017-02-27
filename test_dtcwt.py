import click

import wavelets.dtcwt as wdtcwt
import numpy as np

@click.command()
@click.option('--oned', default=True)
@click.option('--twod', default=False)
@click.option('--lenna-path', 
    default='/home/oskar/Data/DTCWTSample/lennaX.csv')
def run_it_all_day(
    oned, twod, lenna_path):

    oned = bool(oned)
    twod = bool(twod)

    near_sym_b = wdtcwt.utils.get_wavelet_basis(
        'near_sym_b')
    qshift_b = wdtcwt.utils.get_wavelet_basis(
        'qshift_b')

    if oned:
        X = np.random.randn(512,1)

        with open('data.txt', 'w') as f:
            X_str = '\n'.join([
                str(x) for x in X.T[0].tolist()])
            f.write(X_str)

        (Yl, Yh, Y_scale) = wdtcwt.oned.dtwavexfm(
            X, 5, near_sym_b, qshift_b)
        Z = wdtcwt.oned.dtwaveifm(
            Yl, Yh, near_sym_b, qshift_b)
        error = np.abs(Z-X).max()

        if error < 10**(-12):
            print 'Ur gud, dood'
        else:
            print error

    if twod:
        line_list = []

        with open(lenna_path) as f:
            for l in f:
                vals = l.strip().split(',')

                line_list.append(
                    [float(v) for v in vals])

        X = np.array(line_list)
        (Yl, Yh, Y_scale) = wdtcwt.twod.dtwavexfm2(
            X, 4, near_sym_b, qshift_b)

if __name__=='__main__':
    run_it_all_day()
