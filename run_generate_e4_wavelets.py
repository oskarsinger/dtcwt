import click
import os

from dtcwt.mask import DTCWTMask as DTCWTM
from whitehorses.servers.masks import Interp1DMask as I1DM
from whitehorses.servers.batch import BatchServer as BS
from drrobert.file_io import get_timestamped as get_ts

import whitehorses.loaders.shortcuts as dlsh

@click.command()
@click.option('--data-path')
@click.option('--save-dir')
@click.option('--num-subperiods', default=288)
@click.option('--interpolate', default=False)
@click.option('--max-freqs', default=7)
@click.option('--max-hertz', default=4)
@click.option('--pr', default=False)
@click.option('--overlap', default=True)
@click.option('--csv', default=True)
def run_it_all_day_bb(
    data_path,
    save_dir,
    num_subperiods,
    interpolate,
    max_freqs,
    max_hertz,
    pr,
    overlap,
    csv):

    loaders = dlsh.get_hr_and_acc_all_subjects(
        data_path, max_hertz=max_hertz)
    servers = {s : [BS(dl) for dl in dls]
               for (s, dls) in loaders.items()}
    save_dir = os.path.join(
        save_dir, get_ts('DTCWT'))

    os.mkdir(save_dir) 

    if interpolate:
        servers = {s : [I1DM(ds) for ds in dss]
                   for (s, dss) in servers.items()}

    servers = {s : [DTCWTM(
                        ds, 
                        save_dir, 
                        magnitude=True,
                        pr=pr,
                        period=int(24*3600 / num_subperiods),
                        max_freqs=max_freqs,
                        overlap=overlap,
                        save=True,
                        csv=csv)
                    for ds in dss]
               for (s, dss) in servers.items()}

    for s in servers:
        for server in s:
            server.get_data()

if __name__=='__main__':
    run_it_all_day_bb()
