import pytest
from app.main import app


@pytest.fixture(scope="session")
def input_value():
    input = 39
    return input


@pytest.fixture(scope="session")
def wsgi():
    app.config.update(
        {
            "TESTING": True,
        }
    )
    yield app


@pytest.fixture(scope="session")
def client(wsgi):
    return wsgi.test_client()
