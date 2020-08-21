#!/usr/bin/env python3
import logging
import traceback

import zarr
import numpy as np
import config
import process
import msgpack


from flask import Flask, request, jsonify, Response, make_response
app = Flask(__name__)

from flask.json import JSONEncoder

logging.basicConfig(level=logging.DEBUG)

# TODO: Move this to a config file.
# TODO: Figure out how to incorporate render transformations as well

open_n5_mip = {}

def get_n5(n5dataset, mip):
    # Attempt to open & store handle to n5 groups
    key = (n5dataset, mip)
    if key in open_n5_mip:
        return open_n5_mip[key]
    try:
        datainfo = config.DATASOURCES[n5dataset]
        if datainfo['type'] not in ['n5', 'zarr']:
            raise Exception("Datasource type is not supported")
        if mip not in datainfo['scales']:
            raise Exception("Scale not found")

        if datainfo['type'] == 'n5':
            zroot = zarr.open(datainfo['path'], mode='r')
        else:
                store = zarr.NestedDirectoryStore(datainfo['path'])
                zroot = zarr.group(store=store)
        s = zroot["s%d" % mip]
        open_n5_mip[key] = s
        return s
    except:
        raise

#
# Create a new werkzeug routing rule to support both integers and floating point numbers.
# For output, generate float only if a decimal is required.
#
from werkzeug.routing import FloatConverter as BaseFloatConverter
class SpecialNumberConverter(BaseFloatConverter):
    regex = r'-?\d+(\.\d+)?'
    #num_convert = lambda self, x: int(x) if float(x).is_integer() else float(x)

    def num_convert(self, x):
        x = round(float(x), 2)
        if x.is_integer():
            return int(x)
        else:
            return x

app.url_map.converters['number'] = SpecialNumberConverter

#
# Single point vector field query
#
@app.route('/dataset/<string:dataset>/s/<int:scale>/z/<number:z>/x/<number:x>/y/<number:y>/')
def view_link(dataset, scale, z, x, y):
    n5 = get_n5(dataset, scale)

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

    return jsonify(result)


#
# Implement the bulk point interface.
# This has been "stolen" from Philipp's CloudVolumeServer
#
@app.route("/dataset/<string:dataset>/s/<int:scale>/values", methods=['POST'])
def values(dataset, scale):
    """Return segment IDs at given locations."""

    # Parse values
    try:
        locs = None
        if request.mimetype == 'application/json':
            # Handle JSON
            locs = request.json['locations']
        elif request.mimetype in ["application/msgpack", "application/x-msgpack"]:
            locs = msgpack.unpackb(request.get_data())['locations']
        else:
            raise Exception("Unexpected mimetype: {}".format(request.mimetype))

        if not locs:
            return make_response(jsonify({'error': 'No locations provided'}), 400)

        locs = np.array(locs).astype(np.float32)

        if locs.shape[0] > config.MaxLocations:
            err = {'error': 'Max number of locations ({}) exceeded'.format(config.MaxLocations)}
            return make_response(jsonify(err), 400)

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

        if request.mimetype in ["application/msgpack", "application/x-msgpack"]:
            return(msgpack.packb(results))
        else:
            return jsonify(results)

    except BaseException as e:
        app.logger.error('Error: {}'.format(e))
        err = {'error': str(e)}
        return make_response(jsonify(err), 400)



# Implement the bulk point interface with input arrays (for Greg)
@app.route("/dataset/<string:dataset>/s/<int:scale>/values_array", methods=['POST'])
def values_array(dataset, scale):
    """Return segment IDs at given locations."""

    # Parse values
    try:
        locs = None
        if request.mimetype == 'application/json':
            # Handle JSON
            locs = request.json
        elif request.mimetype in ["application/msgpack", "application/x-msgpack"]:
            locs = msgpack.unpackb(request.get_data())
        else:
            raise Exception("Unexpected mimetype: {}".format(request.mimetype))

        if not locs:
            return make_response(jsonify({'error': 'No locations provided'}), 400)

        # Get a Nx3 array of points
        locs = np.array([locs['x'], locs['y'], locs['z']]).astype(np.float32).swapaxes(0,1)

        if locs.shape[0] > config.MaxLocations:
            err = {'error': 'Max number of locations ({}) exceeded'.format(config.MaxLocations)}
            return make_response(jsonify(err), 400)

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

        if request.mimetype in ["application/msgpack", "application/x-msgpack"]:
            return(msgpack.packb(results))
        else:
            return jsonify(results)

    except BaseException as e:
        app.logger.error('Error: {}'.format(e))
        err = {'error': str(e)}
        return make_response(jsonify(err), 400)



def map_points(dataset, scale, locs):
    """Do the work for mapping data.
       Input:  [n,3] numpy array representing n (x,y,z) points
       Output: [n,5] numpy array representing n (new_x, new_y, new_z, new_dx, new_dy)
    """
    n5 = get_n5(dataset, scale)
    voxel_offset = n5.attrs['voxel_offset']

    query_points = np.empty_like(locs)
    query_points[:,0] = (locs[:,0] // 2**scale) - voxel_offset[0]
    query_points[:,1] = (locs[:,1] // 2**scale) - voxel_offset[1]
    query_points[:,2] = (locs[:,2] - voxel_offset[2])

    bad_points = ((query_points <= [0,0,0]) | (query_points >= np.array(n5.shape)[0:3])).all(axis=1)
    query_points[bad_points] = np.NaN

    try:
        if bad_points.all():
            # No valid points. The binning code will otherwise fail.
            # Make a fake field of NaN
            field = np.full((query_points.shape[0], 2), np.NaN)
        else:
            field = process.get_multiple_ids(query_points, n5,
                                            max_workers=config.MaxWorkers)
    except BaseException:
        tb = traceback.format_exc()
        app.logger.error('Error: {}'.format(tb))
        return make_response(jsonify({'error': str(tb)}), 500)

    results = np.zeros(locs.shape[0], dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('dx', '<f4'), ('dy', '<f4')])

    # Apply results

    # (dx, dy) = np.flip(field[i] / 4.0)
    results['dx'] = field[:,1] / 4.0
    results['dy'] = field[:,0] / 4.0
    results['x'] = locs[:,0] + results['dx']
    results['y'] = locs[:,1] + results['dy']
    results['z'] = locs[:,2]

    return results

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True)
