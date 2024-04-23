from contextlib import contextmanager

import pytest
from ae_rlhf.app import database

database.init_db()


@contextmanager
@pytest.fixture(scope="function")
def session():
    """Context manager for a database session that rolls back after every test."""
    with database.Session() as db:
        yield db
        db.rollback()


@pytest.fixture(scope="function")
def db(session):
    """Create a session but rollback all changes after the test."""
    with session as db:
        yield db
