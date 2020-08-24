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
