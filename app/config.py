
import os

# Number of cores used to parallel fetching of locations
MaxWorkers = 32

# Max number of locations per query
MaxLocations = 10e9

# Number of chunks for each worker to read
# Each chunk dimension is multiplied by this.
# e.g. 4 will lead to 64 (4*4*4) chunks per worker
CHUNK_MULTIPLIER = 2

DATASOURCES = {
    'test' : {
        'description' : 'Test volume',
        'type' : 'zarr',
        'scales' : [7],
        'voxel_size' : [4,4,40],
        'services' : ['transform'],
        'dtype' : 'float32',
        'width' : 2,
        'tsinfo' : {
            'driver' : 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': 'test.zarr',
            },
        }   
    },
    'flywire_v1' : {
        'description' : 'Mapping from FlyWire (v14.1) to FAFBv14',
        'type' : 'zarr-nested',
        'scales' : [2, 3, 4, 5, 6, 7],
        'voxel_size' : [4,4,40],
        'services' : ['transform'],
        'dtype' : 'float32',
        'width' : 2,
        'tsinfo' : {
            'driver' : 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': '/data/fields/flywire_v1/field.zarr'
            },
            'key_encoding': '/'
        }  
    },
    'flywire_v1_inverse' : {
        'description' : 'Mapping from FAFBv14 to FlyWire (v14.1)',
        'type' : 'zarr-nested',
        'scales' : [4],
        'voxel_size' : [4,4,40],
        'services' : ['transform'],
        'dtype' : 'float32',
        'width' : 2,
        'tsinfo' : {
            'driver' : 'zarr',
            'kvstore': {
                'driver': 'file',
                'path' : '/data/fields/v0_inverse/field.zarr',
            },
            'key_encoding': '/'
        }  
    },
    'fanc_v4_to_v3' : {
        'description' : 'Mapping from FANCv4 to FANCv3',
        'type' : 'zarr-nested',
        'scales' : [2],
        'voxel_size' : [4.3,4.3,45],
        'services' : ['transform'],
        'dtype' : 'float32',
        'width' : 2,
        'tsinfo' : {
            'driver' : 'zarr',
            'kvstore': {
                'driver': 'file',
                'path' : '/data/fields/fanc/field.zarr',
            },
            'key_encoding': '/'
        }  
    },
    'fafb-ffn1-20200412' : {
        'description' : 'fafb-ffn1-20200412 segmentation',
        'type' : 'neuroglancer_precomputed',
        'scales' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'voxel_size' : [4,4,40],
        'services' : ['query'],
        'dtype' : 'uint64',
        'width' : 1,
        'tsinfo' : {
            'driver' : 'neuroglancer_precomputed',
            'kvstore': {
                'driver': 'file',
                'path': '/data/fields/fafb-ffn1-20200412',
            },
            'path': 'segmentation',
        }  
        
    },
    'fafb-ffn1-20200412-gcs' : {
        'description' : 'fafb-ffn1-20200412 segmentation (access via GCS)',
        'type' : 'neuroglancer_precomputed',
        'scales' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'voxel_size' : [4,4,40],
        'services' : ['query'],
        'dtype' : 'uint64',
        'width' : 1,
        'tsinfo' : {
            'driver' : 'neuroglancer_precomputed',
            'kvstore': {
                'driver': 'gcs',
                'bucket': 'fafb-ffn1-20200412',
            },
            'path': 'segmentation',
        }       
    }
}
