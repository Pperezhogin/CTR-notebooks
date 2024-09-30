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

    ds = CollectionOfExperiments.init_folder('/scratch/pp2681/mom6/Feb2022/bare', additional_subfolder='output')
    ds += CollectionOfExperiments.init_folder('/scratch/pp2681/Yellowstone/experiments/zelong-model-FGR', additional_subfolder='output')

    with ProgressBar():
        ds.remesh('R64', f'{RR}-FGR-sqrt12', operator=gaussian_remesh, FGR=np.sqrt(12), exp='filtered-coarsegrained', compute=True)
    exp = ds['filtered-coarsegrained']

    # As compared to existing FGR=sqrt(12) data this one is obtained by refining the grid, but not just increasing FGR
    dataset = xr.Dataset()
    for key in ['u', 'v', 'e', 'h', 'RV']:
        dataset[key] = exp.__getattribute__(key)
    dataset.astype('float32').to_netcdf(f'/scratch/pp2681/Yellowstone/filtered-data/R64_{RR}_FGR-sqrt12-grid-part1.nc')

    dataset = xr.Dataset()
    for key in ['ua', 'va', 'ea', 'ha']:
        dataset[key] = exp.__getattribute__(key)
    dataset.astype('float32').to_netcdf(f'/scratch/pp2681/Yellowstone/filtered-data/R64_{RR}_FGR-sqrt12-grid-part2.nc')

    dataset = xr.Dataset()
    sgs = exp.compute_subfilter_forcing()
    for key in sgs.keys():
        with ProgressBar():
            dataset[key] = sgs[key].compute()
    dataset.astype('float32').to_netcdf(f'/scratch/pp2681/Yellowstone/filtered-data/R64_{RR}_FGR-sqrt12-grid-part3.nc')