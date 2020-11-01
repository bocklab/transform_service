import os
import json

from fastapi import HTTPException
import tensorstore as ts

from . import config

open_n5_mip = {}

def get_datasource_info(dataset_name):
    if dataset_name not in config.DATASOURCES:
        raise HTTPException(status_code=400, detail="Dataset {} not found".format(dataset_name))
    return config.DATASOURCES[dataset_name]


def get_datastore(dataset_name, mip):
    # Attempt to open & store handle to n5 groups
    key = (dataset_name, mip)
    if key in open_n5_mip:
        return open_n5_mip[key]

    if dataset_name not in config.DATASOURCES:
        raise HTTPException(status_code=400, detail="Dataset {} not found".format(dataset_name))
    
    datainfo = config.DATASOURCES[dataset_name]
        
    if mip not in datainfo['scales']:
        raise HTTPException(status_code=400, detail="Scale {} not found".format(mip))

    # Read main settings from config
    tsinfo = datainfo['tsinfo']
    
    # Set rest of settings for all datasources

    tsinfo['recheck_cached_metadata'] = 'open'
    tsinfo['recheck_cached_data'] = 'open'
    tsinfo['context'] = { 'cache_pool' : { 'total_bytes_limit': 200_000_000 }}

    if datainfo['type'] == 'neuroglancer_precomputed':
        tsinfo['scale_index'] = mip
    elif datainfo['type'] in ["zarr", "zarr-nested"]:
        # Zarr files have mipmaps stored in "s7" under root
        tsinfo['kvstore']['path'] = "%s/s%d" % (tsinfo['kvstore']['path'], mip)

        # Read the voxel_offset as well, to set the transform
        # TODO: Does tensorstore support Zarr attributes?
        # TODO: Figure out dimensionality (4 vs 3) rather than assume 4 for Zarr
        zattr_file = os.path.join(tsinfo['kvstore']['path'], ".zattrs")
        if os.path.exists(zattr_file):
            with open(zattr_file) as f:
                zattr = json.load(f)
                print(zattr)
                if 'voxel_offset' in zattr:

                    outputmaps = []
                    for dim in range(len(zattr['voxel_offset'])):
                        outputmaps.append(ts.OutputIndexMap(offset= - zattr['voxel_offset'][dim], input_dimension=dim))

                    if datainfo['width'] > 1:
                        outputmaps.append(ts.OutputIndexMap(offset=0, input_dimension=len(outputmaps)))

                    x = ts.IndexTransform(input_rank=len(outputmaps), output=outputmaps)
                    tsinfo['transform'] = x.to_json()


    else:
        raise HTTPException(status_code=400, detail="Datasource type '{}' not found".format(datainfo['type'] ))

    s = ts.open(tsinfo).result()
    print(s)
    open_n5_mip[key] = s
    return s


def get_datastore_downsample(dataset_name, mip):
    # Get the downsample for a given mip level
    info = get_datasource_info(dataset_name)
    if 'downsample_factor' in info:
        return info['downsample_factor'][mip]
    else:
        return [2**mip, 2**mip, 1.0]