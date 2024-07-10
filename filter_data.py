import xarray as xr
from helpers.collection_of_experiments import *
from helpers.computational_tools import gaussian_remesh
from dask.diagnostics import ProgressBar
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--RR', type=str, default='R8')
    args = parser.parse_args()

    RR = args.RR

    print(args, RR)

    ds = CollectionOfExperiments.init_folder('/home/ctrsp-2024/pp2681/experiments/Feb2022/bare', additional_subfolder='output')

    with ProgressBar():
        ds.remesh('R64', RR, operator=gaussian_remesh, FGR=np.sqrt(6), exp='filtered-coarsegrained', compute=True)
    exp = ds['filtered-coarsegrained']

    dataset = xr.Dataset()
    for key in ['u', 'v', 'e', 'h', 'RV']:
        dataset[key] = exp.__getattribute__(key)
    dataset.to_netcdf(f'/home/ctrsp-2024/pp2681/notebooks/filtered-data/R64_{RR}_FGR-sqrt6-part1.nc')

    dataset = xr.Dataset()
    for key in ['ua', 'va', 'ea', 'ha']:
        dataset[key] = exp.__getattribute__(key)
    dataset.to_netcdf(f'/home/ctrsp-2024/pp2681/notebooks/filtered-data/R64_{RR}_FGR-sqrt6-part2.nc')

    dataset = xr.Dataset()
    sgs = exp.compute_subfilter_forcing()
    for key in sgs.keys():
        with ProgressBar():
            dataset[key] = sgs[key].compute()
    dataset.to_netcdf(f'/home/ctrsp-2024/pp2681/notebooks/filtered-data/R64_{RR}_FGR-sqrt6-part3.nc')