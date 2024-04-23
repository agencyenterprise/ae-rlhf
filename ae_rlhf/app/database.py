import logging
from contextlib import contextmanager

from sqlmodel import Session as SQLModelSession
from sqlmodel import SQLModel, create_engine

from ae_rlhf.app.models import Model, Pair, Run, Segment  # noqa: F401
from ae_rlhf.config import settings

logger = logging.getLogger(__name__)


def new_engine():
    """Create a new database engine"""
    return create_engine(
        settings.DATABASE_URL,
        echo=settings.DEBUG,
        connect_args={"check_same_thread": False},
    )


engine = new_engine()


def init_db():
    """Initialize the database engine"""
    SQLModel.metadata.create_all(engine)


@contextmanager
def Session():
    """Context manager for a database session"""
    session = SQLModelSession(engine)
    try:
        yield session
        session.commit()
    except Exception as e:
        logger.error(f"SQL Error {str(e)}")
        logger.debug("Trace", exc_info=True)
        session.rollback()
        logger.error("Rollback complete")
        raise
    finally:
        session.close()


def get_session():
    """Get a database session.

    This is used by the FastAPI dependency injection system.
    """
    with Session() as session:
        yield session
