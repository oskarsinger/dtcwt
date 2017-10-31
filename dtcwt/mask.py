import h5py
import os

import numpy as np

from drrobert.file_io import get_timestamped as get_ts
from dtcwt.utils import get_padded_wavelets as get_pw
from dtcwt.utils import get_wavelet_basis as get_wb
from dtcwt.oned import get_partial_reconstructions as get_pr
from dtcwt.oned import dtwavexfm
from theline.utils.misc import get_array_mod
from math import log

# TODO: implement CSV loading
class DTCWTMask:

    def __init__(self,
        ds,
        save_load_path,
        period=3600,
        max_freqs=7,
        pr=False,
        magnitude=False,
        serve_one_period=False,
        overlap=False,
        load=False,
        save=False,
        csv=False):

        # For now, assume server is batch-style
        self.ds = ds
        self.overlap = overlap
        self.period = period
        self.hertz = self.ds.get_status()['data_loader'].get_status()['hertz']
        self.max_freqs = max_freqs
        self.save_load_path = save_load_path
        self.pr = pr
        self.magnitude = magnitude
        self.serve_one_period = serve_one_period
        self.load = load
        self.save = save
        self.csv = csv

        self.window = int(self.period * self.hertz)
        self.w_window = int(self.window / 2)
        self.num_freqs = min([
            int(log(self.window, 2)) - 1,
            self.max_freqs])
        self.num_rounds = 0
        self.biorthogonal = get_wb('near_sym_b')
        self.qshift = get_wb('qshift_b')

        # Probably put all this is a separate func
        data = None
        repo = None
        dl = self.ds.get_status()['data_loader']
        name = '_'.join([
            's',
            dl.name(),
            dl.subject,
            'f',
            str(self.hertz),
            'p',
            str(self.period),
            'mf',
            str(self.max_freqs)])

        if not self.csv:
            name += '.hdf5'

        self.save_load_path = os.path.join(
            save_load_path, name)

        if self.load:

            repo = None

            if self.csv:
                os.mkdir(self.save_load_path)
            else:
                repo = h5py.File(
                    self.save_load_path, 'r')

            data = None
            num_batches = len(repo)
        elif self.save:
            # If saving, then data has obvi not been transformed yet
            data = self.ds.get_data()
            num_batches = int(float(data.shape[0]) / self.window)
            data = np.reshape(
                get_array_mod(data, self.window),
                (num_batches, self.window))
            repo = None

            if self.csv:
                os.mkdir(self.save_load_path)
            else:
                repo = h5py.File(
                    self.save_load_path, 'w')

        self.num_batches = num_batches
        self.data = data
        self.repo = repo
        self.current_w = None

    def get_data(self):

        wavelets = None

        if self.serve_one_period:
            new_w = self._get_one_period(self.num_rounds)

            if self.overlap:
                if self.num_rounds == 1:
                    wavelets = new_w[:self.w_window,:]
                elif self.num_rounds < self.num_batches:
                    w1 = self.current_w[self.w_window:,:]
                    w2 = new_w[:self.w_window,:]
                    wavelets = w1 + w2 / 2
                else:
                    wavelets = new_w

                self.current_w = new_w
            else:
                wavelets = new_w

        else:
            wavelets = [self._get_one_period(i)
                        for i in range(self.num_batches)]

            if self.overlap:
                averaged = [
                    wavelets[0][:self.w_window,:]]
                pairs = zip(
                    [None] + wavelets,
                    wavelets + [None])[1:-1]

                for (w1, w2) in pairs:
                    averaged.append(
                        w1[self.w_window:,:] + w2[:self.w_window,:] / 2)
                    
                averaged.append(
                    wavelets[-1][self.w_window:,:])

                wavelets = averaged

        return wavelets

    def _get_one_period(self, i):

        (Yh, Yl) = [None] * 2

        if self.load:
            (Yh, Yl) = self._load_wavelets(i)
        else:
            (Yh, Yl) = self._get_new_wavelets(i)

        if self.pr:
            (Yh, Yl) = get_pr(
                Yh, 
                Yl, 
                self.biorthogonal, 
                self.qshift)

        wavelets = get_pw(Yh, Yl)

        if self.magnitude:
            wavelets = np.absolute(wavelets)

        if self.save and not self.load:
            self._save(i, wavelets)

        self.num_rounds += 1

        return wavelets

    def _get_new_wavelets(self, i):

        data = self.data[i,:][:,np.newaxis]

        if self.overlap and i < self.num_batches - 1:
            data = np.vstack([
                data,
                self.data[i+1,:][:,np.newaxis]])

        (Yl, Yh, _) = dtwavexfm(
            data,
            self.num_freqs - 1,
            self.biorthogonal,
            self.qshift)

        return (Yh, Yl)

    def _save(self, i, wavelets):

        key = str(i)
        
        if self.csv:
            file_path = os.path.join(
                self.save_load_path,
                str(i) + 'wavelets' + '.csv')

            np.savetxt(
                file_path,   
                wavelets,
                delimiter=',')
        else:
            self.repo.create_dataset(
                key, data=wavelets)

    def _load_wavelets(self, i):

        # TODO: implement loading csv; not high priority
        # TODO: also, reimplement hdf5 for new saving format
        group = self.repo[str(i)]
        num_Yh = len(group) - 1
        Yh = [np.array(group['Yh_' + str(j)]) 
              for j in range(num_Yh)]
        Yl = np.array(group['Yl'])

        return (Yh, Yl)

    def cols(self):

        return self.num_freqs

    def rows(self):

        return self.ds.rows() / 2

    def refresh(self):

        self.ds.refresh()

        self.num_rounds = 0
        self.current_w = None

    def get_status(self):
        
        new_status = {
            'ds': self.ds,
            'period': self.period,
            'hertz': self.hertz,
            'max_freqs': self.max_freqs,
            'save_load_path': self.save_load_path,
            'pr': self.pr,
            'load': self.load,
            'save': self.save,
            'window': self.window,
            'repo': self.repo}

        for (k, v) in self.ds.get_status().items():
            if k not in new_status:
                new_status[k] = v

        return new_status
