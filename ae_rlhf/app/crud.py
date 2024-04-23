from typing import Literal, Sequence

from sqlmodel import Session, select

from ae_rlhf.app.models import Model, Pair, Run, Segment


def create_run(db: Session, run: Run) -> Run:
    """Create a run in the database.

    Args:
        db (Session): Database session.
        run (Run): Run to create.

    Returns:
        Run: Created run.
    """
    db.add(run)
    return run


def read_run(
    db: Session, *, id: int | None = None, name: str | None = None
) -> Run | None:
    """Read a run from the database.

    One of the run id or the run name must be provided.

    Args:
        db (Session): Database session.
        id (int, optional): Run ID. Defaults to None.
        name (str, optional): Run name. Defaults to None.

    Returns:
        Run|None: The run if it exists else None.
    """
    if id is None and name is None:
        raise ValueError("Either the run id or the run name must be provided.")
    if id is not None:
        run = db.get(Run, id)
    else:
        stmt = select(Run).where(Run.name == name)
        run = db.exec(stmt).first()

    return run


def read_runs(db: Session, env_id: str | None = None) -> Sequence[Run]:
    """Read runs from the database.

    Args:
        db (Session): Database session.
        env_id (str, optional): Environment ID. Defaults to None.

    Returns:
        Sequence[Run]: Runs.
    """
    stmt = select(Run)
    if env_id is not None:
        stmt = stmt.where(Run.env_id == env_id)

    return db.exec(stmt).all()


def update_run(db: Session, run: Run) -> Run:
    """Update a run in the database.

    Args:
        db (Session): Database session.
        run (Run): Run to update.
    """
    db.add(run)
    return run


def delete_run(db: Session, *, id: int | None = None, name: str | None = None):
    """Delete a run from the database.

    One of the run id or the run name must be provided.

    Args:
        db (Session): Database session.
        id (int, optional): Run ID. Defaults to None.
        name (str, optional): Run name. Defaults to None.
    """
    if id is None and name is None:
        raise ValueError("Either the run id or the run name must be provided.")
    if id is not None:
        run = db.get(Run, id)
    else:
        stmt = select(Run).where(Run.name == name)
        run = db.exec(stmt).first()
    db.delete(run)


def read_unique_runs(db: Session) -> Sequence[Run]:
    """Read unique runs from the database.

    Args:
        db (Session): Database session.

    Returns:
        list[Run]: Unique runs.
    """
    stmt = select(Run)
    runs = db.exec(stmt).all()
    return runs


def create_pair(db: Session, pair: Pair) -> Pair:
    """Create a pair in the database.

    Args:
        db (Session): Database session.
        pair (Pair): Pair to create.

    Returns:
        Pair: Created pair.
    """
    db.add(pair)
    return pair


def read_pair(db: Session, *, id: int) -> Pair | None:
    """Read a pair from the database.

    Args:
        db (Session): Database session.
        id (int, optional): Pair ID. Defaults to None.

    Returns:
        Pair | None: The pair if it exists else None.
    """
    pair = db.get(Pair, id)
    return pair


def read_pairs(
    db: Session,
    *,
    run_name: str | None = None,
    run_id: int | None = None,
    iteration: int | None = None,
    limit: int | None = None,
    order_by: str | None = None,
    sort: Literal["asc", "desc"] = "asc",
) -> Sequence[Pair]:
    """Read pairs from the database.

    Args:
        db (Session): Database session.
        run_name (str, optional): Run name. Defaults to None.
        run_id (int, optional): Run ID. Defaults to None.

    Returns:
        list[Pair]: Pairs.
    """
    if run_id is not None:
        stmt = select(Pair).join(Run).where(Run.id == run_id)
    elif run_name is not None:
        stmt = select(Pair).join(Run).where(Run.name == run_name)
    else:
        stmt = select(Pair)

    if iteration is not None:
        stmt = stmt.where(Pair.iteration == iteration)

    if order_by is not None:
        order = getattr(Pair, order_by)
        if sort == "asc":
            stmt = stmt.order_by(order)
        elif sort == "desc":
            stmt = stmt.order_by(order.desc())
        else:
            raise ValueError(f"Invalid sort order {sort}.")

    if limit is not None:
        stmt = stmt.limit(limit)

    pairs = db.exec(stmt).all()
    return pairs


def next_unlabeled_pair(db: Session, *, run_name: str) -> Pair | None:
    """Get the next unlabeled pair for the run."""
    stmt = (
        select(Pair)
        .join(Run)
        .where(Pair.label.is_(None), Run.name == run_name)  # type: ignore
        .limit(1)
    )
    pair = db.exec(stmt).first()
    return pair


def update_pair(db: Session, pair: Pair) -> Pair:
    """Update a pair in the database.

    Args:
        db (Session): Database session.
        pair (Pair): Pair to update.

    Returns:
        Pair: Updated pair.
    """
    db.add(pair)
    return pair


def update_label(db: Session, *, id: int, label: int) -> Pair:
    """Add a label to a segment.

    A helper function to update a pair based on its id.

    Args:
        db (Session): Database session.
        id (int): Segment ID.
        label (str): Label.
    """
    pair = db.get(Pair, id)
    if pair is None:
        raise ValueError(f"Pair with id {id} does not exist, cannot update label.")
    pair.label = label
    return update_pair(db, pair)


def delete_pair(db: Session, *, id: int):
    """Delete a pair from the database.

    Args:
        db (Session): Database session.
        id (int, optional): Pair ID. Defaults to None.
    """
    pair = db.get(Pair, id)
    db.delete(pair)


def create_segment(db: Session, segment: Segment) -> Segment:
    """Create a segment in the database.

    Args:
        db (Session): Database session.
        segment (Segment): Segment to create.

    Returns:
        Segment: Created segment.
    """
    db.add(segment)
    return segment


def read_segment(db: Session, *, id: int) -> Segment | None:
    """Read a segment from the database.

    Args:
        db (Session): Database session.
        id (int, optional): Segment ID. Defaults to None.

    Returns:
        Segment|None: The segment if it exists else None.
    """
    segment = db.get(Segment, id)
    return segment


def update_segment(db: Session, segment: Segment) -> Segment:
    """Update a segment in the database.

    Args:
        db (Session): Database session.
        segment (Segment): Segment to update.

    Returns:
        Segment: Updated segment.
    """
    db.add(segment)
    return segment


def delete_segment(db: Session, *, id: int):
    """Delete a segment from the database.

    Args:
        db (Session): Database session.
        id (int, optional): Segment ID. Defaults to None.
    """
    segment = db.get(Segment, id)
    db.delete(segment)


def create_model(db: Session, model: Model) -> Model:
    """Create a model in the database.

    Args:
        db (Session): Database session.
        model (Model): Model to create.

    Returns:
        Model: Created model.
    """
    db.add(model)
    return model


def read_model(db: Session, *, id: int) -> Model | None:
    """Read a model from the database.

    Args:
        db (Session): Database session.
        id (int, optional): Model ID. Defaults to None.

    Returns:
        Model|None: The model if it exists else None.
    """
    model = db.get(Model, id)
    return model


def latest_model(db: Session, *, run_id: int, name: str) -> Model | None:
    """Gets the latest model for a given run."""
    stmt = (
        select(Model)
        .join(Run)
        .where(
            Model.name == name, Model.active.is_(True), Run.id == run_id  # type: ignore
        )
    )
    stmt = stmt.order_by(Model.id.desc()).limit(1)  # type: ignore
    return db.exec(stmt).first()


def update_model(db: Session, model: Model) -> Model:
    """Update a model in the database.

    Args:
        db (Session): Database session.
        model (Model): Model to update.

    Returns:
        Model: Updated model.
    """
    db.add(model)
    return model


def archive_model(db: Session, id: int):
    """Archive a model in the database.

    Args:
        db (Session): Database session.
        id (int): Model ID.
    """
    model = db.get(Model, id)
    if model is None:
        raise ValueError(f"Model with id {id} does not exist, cannot archive.")
    model.active = False
    return update_model(db, model)


def delete_model(db: Session, *, id: int):
    """Delete a model from the database.

    Args:
        db (Session): Database session.
        id (int, optional): Model ID. Defaults to None.
    """
    model = db.get(Model, id)
    db.delete(model)
