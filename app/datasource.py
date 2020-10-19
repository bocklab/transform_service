import zarr
from fastapi import HTTPException
import tensorstore as ts

from . import config

open_n5_mip = {}

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
    tsinfo['context'] = { 'cache_pool' : { 'total_bytes_limit': 100_000_000 }}

    if datainfo['type'] == 'neuroglancer_precomputed':
        tsinfo['scale_index'] = mip
    elif datainfo['type'] in ["zarr", "zarr-nested"]:
        # Zarr files have mipmaps stored in "s7" under root
        tsinfo['kvstore']['path'] = "%s/s%d" % (tsinfo['kvstore']['path'], mip)
    else:
        raise HTTPException(status_code=400, detail="Datasource type '{}' not found".format(datainfo['type'] ))

    s = ts.open(tsinfo).result()
    open_n5_mip[key] = s
    return s


def get_datasource_info(dataset_name):
    if dataset_name not in config.DATASOURCES:
        raise HTTPException(status_code=400, detail="Dataset {} not found".format(dataset_name))
    return config.DATASOURCES[dataset_name]