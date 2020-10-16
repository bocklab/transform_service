
import os

# Number of cores used to parallel fetching of locations
MaxWorkers = os.cpu_count() // 2

# Max number of locations per query
MaxLocations = 10e9

DATASOURCES = {
    'test' : {
        'description' : 'Test volume',
        'type' : 'zarr',
        'path' : 'test.zarr',
        'scales' : [7],
        'voxel_size' : [4,4,40],
        'services' : ['transform'],
        'dtype' : 'float32',
        'width' : 2
    },
    'flywire_v1' : {
        'description' : 'Mapping from FlyWire (v14.1) to FAFBv14',
        'type' : 'zarr-nested',
        'path' : '/data/fields/flywire_v1/field.zarr',
        'scales' : [2, 3, 4, 5, 6, 7],
        'voxel_size' : [4,4,40],
        'services' : ['transform'],
        'dtype' : 'float32',
        'width' : 2
    },
    'flywire_v1_inverse' : {
        'description' : 'Mapping from FAFBv14 to FlyWire (v14.1)',
        'type' : 'zarr-nested',
        'path' : '/data/fields/v0_inverse/field.zarr',
        'scales' : [4],
        'voxel_size' : [4,4,40],
        'services' : ['transform'],
        'dtype' : 'float32',
        'width' : 2
    },
    'fanc_v4_to_v3' : {
        'description' : 'Mapping from FANCv4 to FANCv3',
        'type' : 'zarr-nested',
        'path' : '/data/fields/fanc/field.zarr',
        'scales' : [2],
        'voxel_size' : [4.3,4.3,45],
        'services' : ['transform'],
        'dtype' : 'float32',
        'width' : 2
    },
    'fafb-ffn1-20200412' : {
        'description' : 'fafb-ffn1-20200412 segmentation',
        'url' : 'precomputed://file:///data/fields/fafb-ffn1-20200412/segmentation',
        'type' : 'cloudvolume',
        'scales' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'voxel_size' : [4,4,40],
        'services' : ['query'],
        'dtype' : 'uint64',
        'width' : 1
    }
}
