import zarr
from fastapi import HTTPException
from cloudvolume import CloudVolume

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

    if datainfo['type'] == 'cloudvolume':
        s = CloudVolume(datainfo['url'], mip=mip, bounded=False, fill_missing=True, cache=False)
    else:
        if datainfo['type'] == 'n5':
            zroot = zarr.open(datainfo['path'], mode='r')
        elif datainfo['type'] == 'zarr':
            zroot = zarr.open(datainfo['path'], mode='r')
        elif datainfo['type'] == 'zarr-nested':
            store = zarr.NestedDirectoryStore(datainfo['path'])
            zroot = zarr.group(store=store)
        else:
            raise HTTPException(status_code=400, detail="Datasource type '{}' not found".format(datainfo['type'] ))
            
        s = zroot["s%d" % mip]
    open_n5_mip[key] = s
    return s


def get_datasource_info(dataset_name):
    if dataset_name not in config.DATASOURCES:
        raise HTTPException(status_code=400, detail="Dataset {} not found".format(dataset_name))
    return config.DATASOURCES[dataset_name]