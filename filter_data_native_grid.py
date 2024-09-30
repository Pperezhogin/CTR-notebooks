import xarray as xr
from helpers.collection_of_experiments import *
from helpers.computational_tools import gaussian_remesh
from dask.diagnostics import ProgressBar
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--RR', type=str, default=8)
    args = parser.parse_args()

    RR = args.RR

    print(args, RR)

    ds = CollectionOfExperiments.init_folder('/scratch/pp2681/mom6/Feb2022/bare', additional_subfolder='output')

    with ProgressBar():
        ds.remesh('R64', 'R64', operator=gaussian_remesh, FGR=np.sqrt(6) * 64./RR, exp='filtered', compute=True)
    exp = ds['filtered']

    dataset = xr.Dataset()
    for key in ['u', 'v', 'e', 'h', 'RV']:
        dataset[key] = exp.__getattribute__(key)
    dataset.astype('float32').to_netcdf(f'/scratch/pp2681/Yellowstone/filtered-data/R64_R{RR}_FGR-sqrt6-native-part1.nc')

    dataset = xr.Dataset()
    for key in ['ua', 'va', 'ea', 'ha']:
        dataset[key] = exp.__getattribute__(key)
    dataset.astype('float32').to_netcdf(f'/scratch/pp2681/Yellowstone/filtered-data/R64_R{RR}_FGR-sqrt6-native-part2.nc')

    dataset = xr.Dataset()
    sgs = exp.compute_subfilter_forcing()
    for key in sgs.keys():
        with ProgressBar():
            dataset[key] = sgs[key].compute()
    dataset.astype('float32').to_netcdf(f'/scratch/pp2681/Yellowstone/filtered-data/R64_R{RR}_FGR-sqrt6-native-part3.nc')