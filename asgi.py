import uvicorn

from ae_rlhf.app.app import create_app
from ae_rlhf.config import settings

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "asgi:app", host=settings.HOST, port=settings.PORT, reload=settings.RELOAD
    )
