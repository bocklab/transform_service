#!/usr/bin/env python3
import logging
import traceback

import zarr
import numpy as np
import orjson
import msgpack
import uvicorn

from typing import Optional, List, Any, Tuple

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse
from msgpack_asgi import MessagePackMiddleware
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from . import config
from . import process

# Use orjson for NaN -> null and numpy support
# Source: https://github.com/tiangolo/fastapi/issues/459
class ORJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content)

app = FastAPI(default_response_class=ORJSONResponse, debug=True)
app.add_middleware(MessagePackMiddleware)

open_n5_mip = {}

def get_datasource(dataset_name, mip):
    # Attempt to open & store handle to n5 groups
    key = (dataset_name, mip)
    if key in open_n5_mip:
        return open_n5_mip[key]

    if dataset_name not in config.DATASOURCES:
        raise HTTPException(status_code=400, detail="Dataset {} not found".format(dataset_name))
    
    datainfo = config.DATASOURCES[dataset_name]
        
    if mip not in datainfo['scales']:
        raise HTTPException(status_code=400, detail="Scale {} not found".format(mip))

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

#
# Main page of the service
#
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return """<html>
<head><title>Transformation Service</title></head>
<body>
<h1>Transformation Service</h1>
See the <a href="docs/">API docs</a>.
</body>
</html>"""

@app.get('/info/')
async def dataset_info():
    """Retrieve a list of available datasources."""
    cleaned_datsets = {}
    for k, info in config.DATASOURCES.items():
        cleaned_datsets[k] = {}
        for field in ('scales', 'voxel_size', 'description'):
            cleaned_datsets[k][field] = info.get(field, None)
        
    return cleaned_datsets


#
# Single point vector field query
#
@app.get('/dataset/{dataset}/s/{scale}/z/{z}/x/{x}/y/{y}/')
def view_link(dataset: str, scale: int, z: int, x: int, y: int):
    """Query a single point."""
    n5 = get_datasource(dataset, scale)

    voxel_offset = n5.attrs['voxel_offset']
    query_point = ((x // 2**scale) - voxel_offset[0], (y // 2**scale) - voxel_offset[1], (z - voxel_offset[2]))
    (dx, dy) = np.flip(np.float32(n5[query_point[0],query_point[1],query_point[2],:] / 4.0))
    
    # float32 -> float for JSON
    dx = float(dx)
    dy = float(dy)
    
    result = {
         'z':z, 'x':x+dx, 'y':y+dy,
         'dx' : dx, 'dy' : dy,
     }

    return result


class PointList(BaseModel):
    locations : List[Tuple[float, float, float]]


@app.post('/dataset/{dataset}/s/{scale}/values')
def values(dataset: str, scale: int, data : PointList):
    """Return segment IDs at given locations."""

    locs = np.array(data.locations).astype(np.float32)

    if locs.shape[0] > config.MaxLocations:
        raise HTTPException(status_code=400,
            detail="Max number of locations ({}) exceeded".format(config.MaxLocations))

    # scale & adjust locations
    transformed = map_points(dataset, scale, locs)

    # Apply results
    results = []
    for i in range(transformed.shape[0]):
        row = transformed[i]
        results.append({
            'x': float(row['x']),
            'y': float(row['y']),
            'z': float(row['z']),
            'dx': float(row['dx']),
            'dy': float(row['dy'])
        })

    return results


class ColumnPointList(BaseModel):
    x: List[float]
    y: List[float]
    z: List[float]

@app.post('/dataset/{dataset}/s/{scale}/values_array')
def values_array(dataset: str, scale: int, locs : ColumnPointList):
    """Return segment IDs at given locations."""

    # Get a Nx3 array of points
    locs = np.array([locs.x, locs.y, locs.z]).astype(np.float32).swapaxes(0,1)

    if locs.shape[0] > config.MaxLocations:
        raise HTTPException(status_code=400,
            detail="Max number of locations ({}) exceeded".format(config.MaxLocations))

    # scale & adjust locations
    transformed = map_points(dataset, scale, locs)

    # Set results
    results = {
            'x': transformed['x'].tolist(),
            'y': transformed['y'].tolist(),
            'z': transformed['z'].tolist(),
            'dx': transformed['dx'].tolist(),
            'dy': transformed['dy'].tolist()
        }

    return results


def map_points(dataset, scale, locs):
    """Do the work for mapping data.
       Input:  [n,3] numpy array representing n (x,y,z) points
       Output: [n,5] numpy array representing n (new_x, new_y, new_z, new_dx, new_dy)
    """
    n5 = get_datasource(dataset, scale)
    blocksize = np.asarray(n5.shape[:3]) * 2
    voxel_offset = n5.attrs['voxel_offset']

    query_points = np.empty_like(locs)
    query_points[:,0] = (locs[:,0] // 2**scale) - voxel_offset[0]
    query_points[:,1] = (locs[:,1] // 2**scale) - voxel_offset[1]
    query_points[:,2] = (locs[:,2] - voxel_offset[2])

    bad_points = ((query_points <= [0,0,0]) | (query_points >= np.array(n5.shape)[0:3])).all(axis=1)
    query_points[bad_points] = np.NaN
    if bad_points.all():
        # No valid points. The binning code will otherwise fail.
        # Make a fake field of NaN
        field = np.full((query_points.shape[0], 2), np.NaN)
    else:
        field = process.get_multiple_ids(query_points, n5,
                                        max_workers=config.MaxWorkers,
                                        blocksize=blocksize)

    results = np.zeros(locs.shape[0], dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('dx', '<f4'), ('dy', '<f4')])

    # (dx, dy) = np.flip(field[i] / 4.0)
    results['dx'] = field[:,1] / 4.0
    results['dy'] = field[:,0] / 4.0
    results['x'] = locs[:,0] + results['dx']
    results['y'] = locs[:,1] + results['dy']
    results['z'] = locs[:,2]

    return results


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
