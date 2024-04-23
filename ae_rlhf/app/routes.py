from fastapi import APIRouter, Depends, Response

from ae_rlhf.app import crud
from ae_rlhf.app.database import get_session
from ae_rlhf.app.models import Pair

api = APIRouter()


@api.get("/health")
def health() -> Response:
    return Response(status_code=200, content="OK")


@api.get("/pair/{id}")
def get_pair(id: int, db=Depends(get_session)) -> Pair | None:
    return crud.read_pair(db, id=id)


@api.post("/pair/{id}")
def label_pair(id: int, label: int, db=Depends(get_session)) -> Response:
    crud.update_label(db, id=id, label=label)
    return Response(status_code=200, content="OK", headers={"HX-Trigger": "next-pair"})
