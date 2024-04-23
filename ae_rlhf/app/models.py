from pydantic import BaseModel
from sqlmodel import Field, Relationship, SQLModel


class CrudModel(SQLModel):
    @classmethod
    def create(cls, db):
        obj = cls()
        db.add(obj)
        db.commit()
        db.refresh(obj)
        return obj

    @classmethod
    def read(cls, db, *, id):
        return db.get(cls, id)

    def update(self, db):
        db.add(self)
        db.commit()
        db.refresh(self)
        return self

    def delete(self, db):
        db.delete(self)
        db.commit()


class Run(SQLModel, table=True):
    """A table for tracking runs.

    A run is associated with training data (pairs) and models that were trained on
    that data (models).  The name of the run is used as a unique identified and
    must be unique.  The env_id is optional and is used to track which environment
    the run was trained on.
    """

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(sa_column_kwargs={"unique": True})
    env_id: str | None = None
    pairs: list["Pair"] | None = Relationship(back_populates="run")
    models: list["Model"] | None = Relationship(back_populates="run")


class RunRead(BaseModel):
    """Model for a fully filled in run.

    Useful for typechecking so we don't need to check if each feild is None throughout
    the code.
    """

    id: int
    name: str
    env_id: str | None
    pairs: list["Pair"]


class Pair(SQLModel, table=True):
    """A table for pair of trajectory segments for human feedback.

    For preference learning on atari we use pairs of segments sampled from the agent
    and then we choose which pair is "better".  This table tracks these pairs and their
    preference labels.

    id: Primary key for the pair
    label: The label for the pair.  This is the label that the human gave the pair.
    iteration: The iteration that the pair was sampled from use to track how pairs
        are changing through training.  By convention pretraining is iteration 0
    segments: The segments that make up the pair.  This is a link to the Segment table.
    """

    id: int | None = Field(default=None, primary_key=True)
    iteration: int | None = None
    label: int | None = None
    segments: list["Segment"] | None = Relationship(back_populates="pair")
    run_id: int | None = Field(default=None, foreign_key="run.id")
    run: Run | None = Relationship(back_populates="pairs")


#
# class LabeledPair(BaseModel):
#    """Model for a fully filled in pair.
#
#    Useful for typechecking so we don't need to check if each feild is None throughout
#    the code.
#    """
#
#    id: int
#    label: int
#    segments: list["SegmentWithUri"]
#    run_id: int
#    run: Run


class Segment(SQLModel, table=True):
    """A tabel for segment of a trajectory to be compared (with another segment).

    For each segment we track a file uri which could be something like a file name on
    disk or a could resource.  Each segment will have the video of the agent acting
    in full resolution to make it easier for the human to compare as well as the the
    observations that come from the environment which could be preprocessed, e.g. by
    framestacking and downsampling.

    id: Primary key for the segment
    pair_id: Foreign key to the pair that this segment is a part of.
    pair: The pair that this segment is a part of.
    obs_uri: The uri for the observations for this segment, used for training the
        reward model.
    video_uri: The uri for the video for this segment, used for human feedback.
    """

    id: int | None = Field(default=None, primary_key=True)
    pair_id: int | None = Field(default=None, foreign_key="pair.id")
    pair: Pair = Relationship(back_populates="segments")
    obs_uri: str | None = None
    video_uri: str | None = None


class SegmentWithUri(BaseModel):
    """Model for a segment with the uri filled in.

    Useful for typechecking so we don't need to check if each feild is None throughout
    the code.
    """

    id: int
    pair_id: int
    obs_uri: str
    video_uri: str


class Model(SQLModel, table=True):
    """A table for storing model file locations

    id: Primary key for the model
    run_id: Foreign key to the run that this model was trained on.
    run: The run that this model was trained on.
    name: The name of the model. E.g. "ppo_policy" or "reward_model", etc.
    uri: The uri for the model file.  This could be a file name on disk or a cloud
        resource.
    iteration: The iteration that this model was trained on.  Used to track how models
        are changing through training.  By convention pretraining is iteration 0
    active: Whether or not this model is one that should be considered for future
        training or evaluation.  Setting active to False is a way to archive the model.
    """

    id: int | None = Field(default=None, primary_key=True)
    run_id: int | None = Field(default=None, foreign_key="run.id")
    run: Run | None = Relationship(back_populates="models")
    name: str
    uri: str | None = None
    iteration: int
    active: bool = True
