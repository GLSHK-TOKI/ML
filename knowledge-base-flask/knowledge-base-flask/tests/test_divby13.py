import pytest


def test_divisible_by_13(input_value):
    assert input_value % 13 == 0


@pytest.mark.skip
def test_less_than_200(input_value):
    assert input_value <= 200
