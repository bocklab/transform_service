import numpy as np
from fastapi.testclient import TestClient

from app.main import app

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



def test_query_ffn1():
    q1 = np.array([[87110, 63790, 5436]], dtype=np.single, order="C", copy=True)
    print(client.post("/query/dataset/fafb-ffn1-20200412/s/0/values_binary/format/array_float_Nx3", data=q1.tobytes()))
    r1 = np.frombuffer(client.post("/query/fafb-ffn1-20200412/test/s/0/values_binary/format/array_float_Nx3", data=q1.tobytes()).content, dtype=np.uint64) #.reshape(q1.shape[0],1)
    # 8678640431
    np.testing.assert_equal(r1, np.full((2,2), np.NaN)) 