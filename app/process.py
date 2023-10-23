from pathos.threading import ThreadPool

import numpy as np
import pandas as pd

#
# Code based on https://github.com/flyconnectome/CloudVolumeServer/blob/master/process.py
#


def _get_ids(vol, co):
    """Fetch block and extract IDs.

    Parameters
    ----------
    vol :       tensorstore
                Volume to query.
    co :        numpy array
                x/y/z coordinates WITHIN block
                of segment IDs to fetch.

    """
    # Use integer coordinates (for now...)
    co = co.astype(int)

    co_id = vol[co[:, 0], co[:, 1], co[:, 2]].read().result()

    # Return a 2D array, even if the second dimension is one item
    if co_id.ndim == 1:
        co_id = np.expand_dims(co_id, 1)

    return co_id


def get_multiple_ids(x, vol, dtype=None, max_workers=4, blocksize=np.array([512, 512, 32]), error_value=np.nan):
    """Return multiple segment IDs using cloudvolume.

    Parameters
    ----------
    x :       numpy array
              Array with x/y/z coordinates to fetch
              segmentation IDs for.
    vol :     cloudvolume.CloudVolume
    dtype :   dtype to use for return field (if None, return same as stored)
    """
    # Make sure x is array
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    if not max_workers:
        max_workers = 1

    # Make bins to fit with blocksize
    xbins = np.arange(0, np.nanmax(x) + blocksize[0] + 1, blocksize[0]).astype(int)
    ybins = np.arange(0, np.nanmax(x) + blocksize[1] + 1, blocksize[1]).astype(int)
    zbins = np.arange(0, np.nanmax(x) + blocksize[2] + 1, blocksize[2]).astype(int)

    # Sort data into bins
    cbin = pd.DataFrame(x)
    cbin['x_bin'] = pd.cut(cbin[0], xbins, include_lowest=True, right=False)
    cbin['y_bin'] = pd.cut(cbin[1], ybins, include_lowest=True, right=False)
    cbin['z_bin'] = pd.cut(cbin[2], zbins, include_lowest=True, right=False)

    # Throw out NaNs
    cbin = cbin.loc[~np.any(cbin.isnull(), axis=1)]

    # This is now a dictionary of bin -> indices of coordinates
    blocked = cbin.groupby(['x_bin', 'y_bin', 'z_bin']).indices

    # Map filtered indices back to non-filtered indices
    blocked = {k: cbin.index[v] for k,v in blocked.items()}

    # Start process pool (do not use max cpu count -> appears to be a bottle neck)

    with ThreadPool(nodes=max_workers) as pool:
        seg_ix = []
        cos = []

        # Iterate over all blocks
        for _, co_ix in blocked.items():
            co = x[co_ix]

            # Keep track of the indices of the coordinates we are querying
            # in this iteration
            seg_ix.append(co_ix)

            #  Add to list of coordinates
            cos.append(co)

        result = pool.map(_get_ids, [vol] * len(seg_ix), cos)

        seg_ids = np.vstack(result)

        #pool.clear()

    # Turn list of list of indices into a flat array
    seg_ix = np.hstack(seg_ix)

    # Generate placeholder of NaNs. Get data width from the returned data.
    ordered = np.full((x.shape[0], seg_ids.shape[1]), error_value, dtype=dtype)

    # Populate with segment IDs
    ordered[seg_ix] = seg_ids

    return ordered
