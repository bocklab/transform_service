
import os

# Number of cores used to parallel fetching of locations
MaxWorkers = os.cpu_count() // 2

# Max number of locations per query
MaxLocations = 10e9

DATASOURCES = {
    'flywire_v1' :
    {
        'type' : 'zarr',
        'path' : '/data/fields/flywire_v1/field.zarr',
        'scales' : [2, 3, 4, 5, 6, 7]
    },
    'flywire_v1_inverse' :
    {
        'type' : 'zarr',
        'path' : '/data/fields/v0_inverse/field.zarr',
        'scales' : [4]
    },
    'flywire_v1_test':
    {
        'type' : 'zarr',
        'path' : 'test.zarr',
        'scales' : [7]
    },
    'flywire_v1_test_n5':
    {
        'type' : 'n5',
        'path' : 'test.n5',
        'scales' : [7]
    },
}
