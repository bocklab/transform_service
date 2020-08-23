
import os

# Number of cores used to parallel fetching of locations
MaxWorkers = os.cpu_count() // 2

# Max number of locations per query
MaxLocations = 10e9

DATASOURCES = {
    'flywire_v1' :
    {
        'description' : 'Mapping from FAFBv14 to FlyWire (v14.1)',
        'type' : 'zarr',
        'path' : '/data/fields/flywire_v1/field.zarr',
        'scales' : [2, 3, 4, 5, 6, 7],
        'voxel_size' : [4,4,40]
    },
    'flywire_v1_inverse' :
    {
        'description' : 'Mapping from FlyWire (v14.1) to FAFBv14',
        'type' : 'zarr',
        'path' : '/data/fields/v0_inverse/field.zarr',
        'scales' : [4],
        'voxel_size' : [4,4,40]
    },
    'flywire_v1_test':
    {
        'type' : 'zarr',
        'path' : 'test.zarr',
        'scales' : [7],
        'voxel_size' : [4,4,40]
    },
    'flywire_v1_test_n5':
    {
        'type' : 'n5',
        'path' : 'test.n5',
        'scales' : [7],
        'voxel_size' : [4,4,40]
    },
    'fanc_v3_to_v4':
    {
        'description' : 'Mapping from FANCv3 to FANCv4',
        'type' : 'zarr',
        'path' : '/data/fields/fanc/field.zarr',
        'scales' : [2],
        'voxel_size' : [4.3,4.3,45]
    }
}
