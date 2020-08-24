#!/usr/bin/env python3
import logging
import traceback
from enum import Enum

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

api_description = """
This service will take a set of points and will transform them using the specified field.

Query units should be in *pixels* at full resolution (e.g. mip=0), which generally maps to the pixel values shown in CATMAID or Neuroglancer.

The return values are the transformed {x,y,z} along with the {dx,dy} values from the field.

The selection of scale (mip) selects the granularity of the field being used, but does not change the units.

Error values are returned as `null`, not `NaN` as done with the previous iteration of this service. The most likely cause of an error is being out-of-bounds of the underlying array.

_Note on using [msgpack](https://msgpack.org/)_: Use `Content-type: application/x-msgpack` and `Accept: application/x-msgpack` to use msgpack instead of JSON. There is currently data size limit of *64KB* when using msgpack. I am currently looking for a workaround.

_Wnat a binary endpoint?_ I am considering either npz or raw C-style arrays.

Questions? Feel free to bug Eric on FAFB or FlyWire slack.
"""

# Use orjson for NaN -> null and numpy support
# Source: https://github.com/tiangolo/fastapi/issues/459
class ORJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content)

app = FastAPI(default_response_class=ORJSONResponse,
                title="Transformation Service",
                description=api_description,
                debug=True)

# MessagePackMiddleware does not currently support large request (`more_body`) so we'll do our own...
# app.add_middleware(MessagePackMiddleware)





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
Please see the <a href="docs/">API documentation</a> for usage info.
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


# Datasets to be displayed in UI if they are part of an enum...
# This is a hack to populate the values.
DataSetName = Enum("DataSetName", dict(zip(config.DATASOURCES.keys(), config.DATASOURCES.keys())))

#
# Single point vector field query
#
class PointResponse(BaseModel):
    x: float
    y: float
    z: float
    dx: float
    dy: float

@app.get('/dataset/{dataset}/s/{scale}/z/{z}/x/{x}/y/{y}/', response_model=PointResponse)
def point_value(dataset: DataSetName, scale: int, z: int, x: float, y: float):
    """Query a single point."""

    locs = np.asarray([[x,y,z]])

    transformed = map_points(dataset.value, scale, locs)

    result = {
         'x': transformed['x'].tolist()[0],
         'y': transformed['y'].tolist()[0],
         'z': transformed['z'].tolist()[0],
         'dx': transformed['dx'].tolist()[0],
         'dy': transformed['dy'].tolist()[0]
     }

    return result


class PointList(BaseModel):
    locations : List[Tuple[float, float, float]]

@app.post('/dataset/{dataset}/s/{scale}/values', response_model=List[PointResponse])
def values(dataset: DataSetName, scale: int, data : PointList):
    """Return segment IDs at given locations."""

    locs = np.array(data.locations).astype(np.float32)

    if locs.shape[0] > config.MaxLocations:
        raise HTTPException(status_code=400,
            detail="Max number of locations ({}) exceeded".format(config.MaxLocations))

    # scale & adjust locations
    transformed = map_points(dataset.value, scale, locs)

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

class ColumnPointListResponse(BaseModel):
    x: List[float]
    y: List[float]
    z: List[float]
    dx: List[float]
    dy: List[float]

@app.post('/dataset/{dataset}/s/{scale}/values_array', response_model=ColumnPointListResponse)
def values_array(dataset: DataSetName, scale: int, locs : ColumnPointList):
    """Return segment IDs at given locations."""

    # Get a Nx3 array of points
    locs = np.array([locs.x, locs.y, locs.z]).astype(np.float32).swapaxes(0,1)

    if locs.shape[0] > config.MaxLocations:
        raise HTTPException(status_code=400,
            detail="Max number of locations ({}) exceeded".format(config.MaxLocations))

    # scale & adjust locations
    transformed = map_points(dataset.value, scale, locs)

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
    blocksize = np.asarray(n5.chunks[:3]) * 2
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

    # From Tommy Macrina:
    #   We store the vectors as fixed-point int16 with two bits for the decimal.
    #   Even if a vector is stored at a different MIP level (e.g. these are stored at MIP2),
    #   the vectors represent MIP0 displacements, so there's no further scaling required.

    results['dx'] = field[:,1] / 4.0
    results['dy'] = field[:,0] / 4.0
    results['x'] = locs[:,0] + results['dx']
    results['y'] = locs[:,1] + results['dy']
    results['z'] = locs[:,2]

    return results


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
