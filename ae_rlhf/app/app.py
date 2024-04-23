from contextlib import asynccontextmanager

from fastapi import FastAPI

from ae_rlhf.app.database import init_db
from ae_rlhf.app.routes import api
from ae_rlhf.app.ui import ui


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


def create_app():
    app = FastAPI(lifespan=lifespan)
    app.include_router(ui)
    app.include_router(api, prefix="/api")
    return app
