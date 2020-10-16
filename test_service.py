import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.config import DATASOURCES

client = TestClient(app)

def test_info():
    response = client.get("/info/")
    assert response.status_code == 200


def test_point():
    response = client.get("/dataset/test/s/7/z/5/x/15/y/23")
    assert response.status_code == 200
    assert response.json() == {'z': 5, 'x': -1.0, 'y': 7.25, 'dx': -16.0, 'dy': -15.75}


def test_binary_requests():
    # Make sure array_float_Nx3 and array_float_3xN get the same values
    q1 = np.array([[-1,1,1], [196,1,35], [48,188,9], [-79,23,21]], dtype=np.single, order="C", copy=True)
    r1 = np.frombuffer(client.post("/dataset/test/s/7/values_binary/format/array_float_Nx3", data=q1.tobytes()).content, dtype=np.float32).reshape(q1.shape[0],2)

    q2 = q1.swapaxes(0,1).copy(order='C')
    r2 = np.frombuffer(client.post("/dataset/test/s/7/values_binary/format/array_float_3xN", data=q2.tobytes()).content, dtype=np.float32).reshape(2,q2.shape[1])

    np.testing.assert_equal(r1, r2.swapaxes(0,1))


def test_out_of_range():
    q1 = np.array([[-1,1,1], [5000,5000,5000]], dtype=np.single, order="C", copy=True)
    r1 = np.frombuffer(client.post("/dataset/test/s/7/values_binary/format/array_float_Nx3", data=q1.tobytes()).content, dtype=np.float32).reshape(q1.shape[0],2)
    
    np.testing.assert_equal(r1, np.full((2,2), np.NaN))


def test_query_ffn1_binary():
    # TODO: Create a Google datasource with a fixture
    DATASOURCES['fafb-ffn1-20200412']['url'] = 'https://storage.googleapis.com/fafb-ffn1-20200412/segmentation'

    q1 = np.array([[87110, 63790, 5436], [0,0,0], [106110, 66106, 1968]], dtype=np.single, order="C", copy=True)
    response = client.post("/query/dataset/fafb-ffn1-20200412/s/0/values_binary/format/array_float_Nx3", data=q1.tobytes())
    assert response.status_code == 200

    print(len(response.content))
    r1 = np.frombuffer(response.content, dtype=np.uint64).reshape(q1.shape[0],1)
    # 8678640431
    np.testing.assert_equal(r1, np.asarray([[8678640431], [0], [2938732695]]))


def test_query_ffn1_values_string():
    # TODO: Create a Google datasource with a fixture
    DATASOURCES['fafb-ffn1-20200412']['url'] = 'https://storage.googleapis.com/fafb-ffn1-20200412/segmentation'
    
    q1 = {
            'x' : [87110, 0, 106110],
            'y' : [63790, 0, 66106],
            'z' : [5436, 0, 1968]
        }

    response = client.post("/query/dataset/fafb-ffn1-20200412/s/0/values_array_string_response", json=q1)
    assert response.status_code == 200

    assert response.json() == {'values' : [["8678640431", "0", "2938732695"]]}
