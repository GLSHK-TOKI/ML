def test_request_example(client):
    response = client.get("/")
    assert "Welcome to python backend template" in response.data.decode("utf-8")
