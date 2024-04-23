from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session

from ae_rlhf.app import crud
from ae_rlhf.app.database import get_session
from ae_rlhf.config import CONSTANTS

ui = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


SELECTION_OPTIONS = [
    # order is import, it determines how they are rendered in the UI
    {"id": CONSTANTS.left.id, "name": CONSTANTS.left.name},
    {"id": CONSTANTS.tie.id, "name": CONSTANTS.tie.name},
    {"id": CONSTANTS.right.id, "name": CONSTANTS.right.name},
    {"id": CONSTANTS.unknown.id, "name": CONSTANTS.unknown.name},
]


@ui.get("/", response_class=HTMLResponse)
async def index(request: Request, db: Session = Depends(get_session)):
    runs = crud.read_runs(db)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "runs": [{"name": run.name, "env_id": run.env_id} for run in runs],
        },
    )


@ui.get("/feedback/{run_name}", response_class=HTMLResponse)
async def feedback(request: Request, run_name: str, db: Session = Depends(get_session)):
    """The feedback page for the run.

    This is called when the user first nagivates to feedback/{run_name}.
    """
    context = next_unlabeled_pair_context(db, run_name=run_name)
    return templates.TemplateResponse(
        request=request,
        name="feedback.html",
        context=context,
    )


@ui.get("/feedback/{run_name}/main", response_class=HTMLResponse)
async def feedback_main(
    request: Request, run_name: str, db: Session = Depends(get_session)
):
    """The main content of the feedback page for a given run.

    This used to populate AJAX request for swapping out the main component
    of the feedback page during labeling.
    """
    context = next_unlabeled_pair_context(db, run_name=run_name)
    return templates.TemplateResponse(
        request=request,
        name="feedback_main.html",
        context=context,
    )


def next_unlabeled_pair_context(db: Session, run_name: str) -> dict:
    """Get the next unlabeled pair for the run."""
    pair = crud.next_unlabeled_pair(db, run_name=run_name)
    unlabeled_pairs = [
        p for p in crud.read_pairs(db, run_name=run_name) if p.label is None
    ]
    if pair is None or pair.segments is None:
        video: list[dict] = []
        run = crud.read_run(db, name=run_name)
    else:
        video = []
        for seg in pair.segments:
            if seg is not None and seg.video_uri is not None:
                video.append(
                    {
                        "id": seg.id,
                        "src": f"/video/{seg.id}",
                        "format": uri_to_format(seg.video_uri),
                    }
                )

        if len(video) != 2:
            raise ValueError(f"Expected 2 videos, got {len(video)}")
        run = pair.run

    return {
        "run": run,
        "pair": pair,
        "video": video,
        "selection_options": SELECTION_OPTIONS,
        "remaining_pairs": len(unlabeled_pairs),
    }


@ui.get("/video/{id}")
async def video(id: int, db: Session = Depends(get_session)):
    """Get the video with the given id.


    IMPORTANT: This function assumes that the video is stored locally and
    can be read from the filesystem.  If you want to use cloud storage then
    replace the `with open` block with the appropriate code to read the video
    and return its bytes as a response


    Args:
        id (int): The id of the segement associated with the video
        db (Session): The database session
    """
    segment = crud.read_segment(db=db, id=id)
    if segment is None:
        raise HTTPException(status_code=404, detail="Segment not found")
    if segment.video_uri is None:
        raise HTTPException(status_code=404, detail="Video not found")
    with open(segment.video_uri, "rb") as f:
        return Response(
            f.read(), media_type=f"video/{uri_to_format(segment.video_uri)}"
        )


def uri_to_format(uri: str) -> str:
    """Get the format of the video from the uri."""
    return uri.split(".")[-1]
