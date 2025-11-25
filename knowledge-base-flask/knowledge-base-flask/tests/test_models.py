from app.models.User import User


def test_new_user():
    user = User(13, "FlaskIsAwesome", "patkennedy79@gmail.com")
    assert user.email == "patkennedy79@gmail.com"
    assert user.username == "FlaskIsAwesome"
    assert user.id == 13
