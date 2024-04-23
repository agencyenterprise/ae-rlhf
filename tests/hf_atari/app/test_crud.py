import sqlalchemy
from ae_rlhf.app import crud
from ae_rlhf.app.models import Pair, Run

# testing setup


def test_database_tables_are_created(db):
    """Ensures that the tables are created up initialization."""
    tables = sqlalchemy.inspect(db.get_bind()).get_table_names()
    assert sorted(tables) == ["model", "pair", "run", "segment"]


# testing crud


def test_create_pair(db):
    """Tests that a pair can be created"""
    pair = Pair(label=0)
    created_pair = crud.create_pair(db, pair)
    db.flush()
    assert created_pair.id is not None
    assert created_pair.label == pair.label


def test_create_pair_with_run(db):
    """Tests that a pair can be created with a run"""
    run = Run(name="test-run")
    pair = Pair(run=run)
    created_pair = crud.create_pair(db, pair)
    db.flush()
    assert created_pair.id is not None
    assert created_pair.run_id == pair.run_id
    assert created_pair.run.name == "test-run"


def test_read_pair(db):
    """Tests that a pair can be read"""
    pair = Pair(label=0)
    created_pair = crud.create_pair(db, pair)
    db.flush()
    read_pair = crud.read_pair(db, id=created_pair.id)
    assert read_pair.id == created_pair.id
    assert read_pair.label == created_pair.label


def test_read_pairs_with_limit(db):
    """Tests that pairs can be read with a limit"""
    for _ in range(10):
        pair = Pair()
        crud.create_pair(db, pair)

    pairs = crud.read_pairs(db, limit=5)
    assert len(pairs) == 5


def test_read_pairs_with_run_filter(db):
    """Tests that pairs can be read with an run filter"""
    run_0 = Run(name="test-0")
    run_1 = Run(name="test-1")
    for _ in range(10):
        pair = Pair(run=run_0)
        crud.create_pair(db, pair)

    for _ in range(5):
        pair = Pair(label=0, run=run_1)
        crud.create_pair(db, pair)

    pairs = crud.read_pairs(db, run_name="test-1")
    assert all(pair.run.name == "test-1" for pair in pairs)


def test_read_pairs_with_order_by(db):
    """Tests that pairs can be read with an order_by"""
    for _ in range(10):
        pair = Pair()
        crud.create_pair(db, pair)

    pairs = crud.read_pairs(db, order_by="id", sort="desc")
    assert all(pairs[i].id > pairs[i + 1].id for i in range(len(pairs) - 1))


def test_update_pair(db):
    """Tests that a pair can be updated"""
    pair = Pair()
    created_pair = crud.create_pair(db, pair)
    created_pair.label = 1
    updated_pair = crud.update_pair(db, created_pair)
    assert updated_pair.id == created_pair.id
    assert updated_pair.label == 1


def test_delete_pair(db):
    """Tests that a pair can be deleted"""
    pair = Pair()
    created_pair = crud.create_pair(db, pair)
    db.flush()
    crud.delete_pair(db, id=created_pair.id)


def test_unique_runs_returns_only_unique_run_names(db):
    """Tests that unique runs are returned"""
    run_0 = Run(name="test-0")
    run_1 = Run(name="test-1")
    crud.create_run(db, run_0)
    crud.create_run(db, run_1)

    runs = crud.read_unique_runs(db)
    assert len(runs) == 2
    assert runs[0].name == "test-0"
    assert runs[1].name == "test-1"
