#!/usr/bin/env python3
import logging
import traceback
from enum import Enum
import asyncio

import zarr
import numpy as np
import orjson
import msgpack
import uvicorn

from typing import Optional, List, Any, Tuple

from fastapi import FastAPI, HTTPException, Body, Response, Request
from fastapi.responses import HTMLResponse, ORJSONResponse
from msgpack_asgi import MessagePackMiddleware
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from . import config
from . import process
from .query import map_points, query_points

api_description = """
This service will take a set of points and will transform them using the specified field.

Query units should be in *pixels* at full resolution (e.g. mip=0), which generally maps to the pixel values shown in CATMAID or Neuroglancer.

The return values are the transformed {x,y,z} along with the {dx,dy} values from the field.

The selection of scale (mip) selects the granularity of the field being used, but does not change the units.

Error values are returned as `null`, not `NaN` as done with the previous iteration of this service. The most likely cause of an error is being out-of-bounds of the underlying array.

_Note on using [msgpack](https://msgpack.org/)_: Use `Content-type: application/x-msgpack` and `Accept: application/x-msgpack` to use msgpack instead of JSON. There is currently data size limit of *64KB* when using msgpack. I am currently looking for a workaround.

Questions? Feel free to bug Eric on FAFB or FlyWire slack.
"""

app = FastAPI(default_response_class=ORJSONResponse,
                title="Transformation Service",
                description=api_description,
                debug=True)

# MessagePackMiddleware does not currently support large request (`more_body`) so we'll do our own...
app.add_middleware(MessagePackMiddleware)

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
    """Return dx, dy and new coordinates for an input set of locations."""

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
    """Return dx, dy and new coordinates for an input set of locations."""

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

class QueryColumnPointListResponse(BaseModel):
    values: List[List[float]]

@app.post('/query/dataset/{dataset}/s/{scale}/values_array', response_model=QueryColumnPointListResponse)
def query_values_array(dataset: DataSetName, scale: int, locs : ColumnPointList):
    """Return segment IDs at given locations.
       One 
    """

    # Get a Nx3 array of points
    locs = np.array([locs.x, locs.y, locs.z]).astype(np.float32).swapaxes(0,1)

    if locs.shape[0] > config.MaxLocations:
        raise HTTPException(status_code=400,
            detail="Max number of locations ({}) exceeded".format(config.MaxLocations))

    data = query_points(dataset.value, scale, locs)
    # Nx1 to 1xN
    data = data.swapaxes(0,1)

    # Set results
    results = {
        'values' : data.tolist()
    }

    return results   

class ColumnPointListStringResponse(BaseModel):
    values: List[List[str]]

@app.post('/query/dataset/{dataset}/s/{scale}/values_array_string_response', response_model=ColumnPointListStringResponse)
def query_values_array_string(dataset: DataSetName, scale: int, locs : ColumnPointList):
    """Return segment IDs at given locations.
       Like *query_values_array*, but result array contains strings for easier parsing in R.
    """

    results = query_values_array(dataset, scale, locs)

    results = {
        'values' : [[str(j) for j in i] for i in results['values']]
    }

    return results

class BinaryFormats(str, Enum):
    array_3xN = "array_float_3xN"
    array_Nx3 = "array_float_Nx3"

@app.post('/dataset/{dataset}/s/{scale}/values_binary/format/{format}',
            response_model=None,
            responses={ 200: {"content": {"application/octet-stream": {}},
                        "description": "Binary encoding of output array."}}
            )
def values_binary(dataset: DataSetName, scale: int, format: BinaryFormats, request: Request):
    """Raw binary version of the API. Data will consist of 1 uint 32.
       Currently acceptable formats consist of a single uint32 with the number of records, 
       All values must be little-endian floating point nubers.

       The response will _only_ contain `dx` and `dy`, stored as either 2xN or Nx2 (depending on format chosen)
    """

    body = asyncio.run(request.body())
    points = len(body) // (3 * 4)  # 3 x float
    if format == BinaryFormats.array_3xN:
        locs = np.frombuffer(body, dtype=np.float32).reshape(3,points).swapaxes(0,1)
    elif format == BinaryFormats.array_Nx3:
        locs = np.frombuffer(body, dtype=np.float32).reshape(points,3)
    else:
        raise Exception("Unexpected format: {}".format(format))

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

    data = np.zeros(dtype=np.float32, shape=(2,points), order="C")
    data[0,:] = transformed['dx']
    data[1,:] = transformed['dy']
    if format == BinaryFormats.array_Nx3:
        data = data.swapaxes(0,1)

    return Response(content=data.tobytes(), media_type="application/octet-stream")



@app.post('/query/dataset/{dataset}/s/{scale}/values_binary/format/{format}',
            response_model=None,
            responses={ 200: {"content": {"application/octet-stream": {}},
                        "description": "Binary encoding of output array."}}
            )
def query_values_binary(dataset: DataSetName, scale: int, format: BinaryFormats, request: Request):
    """Query a dataset for values at a point(s)

       The response will _only_ contain the value(s) at the coordinates requested.
       The datatype returned will be of the type referenced in */info/*.
    """

    body = asyncio.run(request.body())
    points = len(body) // (3 * 4)  # 3 x float
    if format == BinaryFormats.array_3xN:
        locs = np.frombuffer(body, dtype=np.float32).reshape(3,points).swapaxes(0,1)
    elif format == BinaryFormats.array_Nx3:
        locs = np.frombuffer(body, dtype=np.float32).reshape(points,3)
    else:
        raise Exception("Unexpected format: {}".format(format))

    if locs.shape[0] > config.MaxLocations:
        raise HTTPException(status_code=400,
            detail="Max number of locations ({}) exceeded".format(config.MaxLocations))

    data = query_points(dataset.value, scale, locs)

    if format == BinaryFormats.array_Nx3:
        data = data.swapaxes(0,1)

    return Response(content=data.tobytes(), media_type="application/octet-stream")

