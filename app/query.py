import numpy as np
import zarr

from . import config
from . import process
from . import datasource

def query_points(dataset, scale, locs):
    """Query a dataset for the points.
       Input:  [n,3] numpy array representing n (x,y,z) points
       Output: [n,5] numpy array representing n (new_x, new_y, new_z, new_dx, new_dy)
    """
    info = datasource.get_datasource_info(dataset)
    n5 = datasource.get_datastore(dataset, scale)

    shape = (n5.domain[0].inclusive_max, n5.domain[1].inclusive_max, n5.domain[2].inclusive_max)


    if info['type'] == 'neuroglancer_precomputed':
        blocksize = np.asarray(n5.spec().to_json()['scale_metadata']['chunk_size']) * config.CHUNK_MULTIPLIER  
    elif info['type'] in ['zarr', 'zarr-nested']:
        blocksize = np.array(n5.spec().to_json()['metadata']['chunks'])[0:3] * config.CHUNK_MULTIPLIER

    query_points = np.empty_like(locs)
    query_points[:,0] = locs[:,0] // 2**scale
    query_points[:,1] = locs[:,1] // 2**scale
    query_points[:,2] = locs[:,2]

    bad_points = ((query_points < [0,0,0]) | (query_points >= np.array(shape)[0:3])).any(axis=1)
    query_points[bad_points] = np.NaN
    if bad_points.all():
        # No valid points. The binning code will otherwise fail.
        field = np.full((query_points.shape[0], info["width"]), np.NaN, dtype=info["dtype"])
    else:
        field = process.get_multiple_ids(query_points, n5,
                                        max_workers=config.MaxWorkers,
                                        blocksize=blocksize)

    return field.astype(info["dtype"])

def map_points(dataset, scale, locs):
    """Do the work for mapping data.
       Input:  [n,3] numpy array representing n (x,y,z) points
       Output: [n,5] numpy array representing n (new_x, new_y, new_z, new_dx, new_dy)
    """

    field = query_points(dataset, scale, locs)
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
