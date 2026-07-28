"""◉ LoRA Canvas — remembered card positions (canvas_node_position).

Where a run card sits on the board is a display preference. These tests hold the
two properties that make it safe to store one:

  * it survives a reload (that is the whole point of slice 2), and
  * it can never stop a dataset from being deleted.

The second is not hypothetical. A child table wired to face_dataset WITHOUT a
mapper-level relationship has no ordering dependency in SQLAlchemy's unit of
work: the parent DELETE is emitted first and a legacy database whose foreign key
predates ON DELETE CASCADE answers HTTP 500. That bug has already shipped once in
this project, so the deletion tests below run with PRAGMA foreign_keys=OFF —
exactly the old-database shape where a missing flush order is fatal.
"""
from sqlalchemy import text


def _dataset(name='Ada', trigger='ada'):
    from app.services import face_dataset_service as svc
    return svc.create_dataset('local', name, trigger)


def _positions(dataset_id):
    from app.models import CanvasNodePosition
    return CanvasNodePosition.query.filter_by(dataset_id=dataset_id).count()


# ---- persistence -----------------------------------------------------------

def test_a_moved_card_survives_a_reload(client, app):
    """PUT then GET: the board reads back exactly what the drag wrote."""
    with app.app_context():
        ds_id = _dataset().id

    r = client.put(f'/api/dataset/{ds_id}/canvas/positions', json={
        'positions': [{'record_id': 7, 'x': 120.5, 'y': 340.0},
                      {'record_id': 8, 'x': 0, 'y': 0}]})
    assert r.status_code == 200
    assert r.get_json()['saved'] == 2

    lane = client.get('/api/train/canvas/positions').get_json()['positions'][str(ds_id)]
    assert lane == [{'record_id': 7, 'x': 120.5, 'y': 340.0},
                    {'record_id': 8, 'x': 0.0, 'y': 0.0}]


def test_moving_the_same_card_twice_updates_it_instead_of_duplicating(client, app):
    """The canvas re-sends a whole lane on every drop and re-pins it whenever a
    run arrives — a second write of the same card must be an update."""
    with app.app_context():
        ds_id = _dataset().id
    for x in (10, 20, 30):
        client.put(f'/api/dataset/{ds_id}/canvas/positions',
                   json={'positions': [{'record_id': 5, 'x': x, 'y': 1}]})
    with app.app_context():
        assert _positions(ds_id) == 1
    lane = client.get('/api/train/canvas/positions').get_json()['positions'][str(ds_id)]
    assert lane == [{'record_id': 5, 'x': 30.0, 'y': 1.0}]


def test_unusable_coordinates_are_refused_not_stored(client, app):
    """One NaN stored here would put a card somewhere no future load can find,
    and there is no UI to repair it. Rows that cannot be trusted are dropped."""
    with app.app_context():
        ds_id = _dataset().id
    r = client.put(f'/api/dataset/{ds_id}/canvas/positions', json={'positions': [
        {'record_id': 1, 'x': float('nan'), 'y': 3},
        {'record_id': 2, 'x': float('inf'), 'y': 3},
        {'record_id': 3, 'x': 'over there', 'y': 3},
        {'record_id': None, 'x': 1, 'y': 2},
        {'record_id': 4, 'x': 12, 'y': 34},
    ]})
    assert r.status_code == 200
    assert r.get_json()['saved'] == 1
    with app.app_context():
        assert _positions(ds_id) == 1


def test_tidy_up_clears_the_lane(client, app):
    """✦ Tidy up is the escape hatch: every remembered position of the lane goes
    and the automatic tree takes over again."""
    with app.app_context():
        ds_id = _dataset().id
    client.put(f'/api/dataset/{ds_id}/canvas/positions', json={
        'positions': [{'record_id': i, 'x': i, 'y': i} for i in range(1, 6)]})
    r = client.delete(f'/api/dataset/{ds_id}/canvas/positions')
    assert r.status_code == 200
    assert r.get_json()['cleared'] == 5
    assert client.get('/api/train/canvas/positions').get_json()['positions'] == {}


def test_tidy_up_on_an_untouched_lane_is_a_no_op_not_an_error(client, app):
    with app.app_context():
        ds_id = _dataset().id
    r = client.delete(f'/api/dataset/{ds_id}/canvas/positions')
    assert r.status_code == 200
    assert r.get_json()['cleared'] == 0


def test_an_unknown_dataset_is_404_on_both_write_paths(client, app):
    assert client.put('/api/dataset/9999/canvas/positions',
                      json={'positions': []}).status_code == 404
    assert client.delete('/api/dataset/9999/canvas/positions').status_code == 404


def test_positions_of_another_user_are_not_on_my_board(client, app):
    """The index is scoped to the caller's datasets — a row whose dataset is not
    mine never reaches the board."""
    from app.extensions import db
    from app.models import CanvasNodePosition
    from app.services import face_dataset_service as svc
    with app.app_context():
        mine = _dataset('Mine', 'mine').id
        theirs = svc.create_dataset('someone-else', 'Theirs', 'theirs').id
        db.session.add(CanvasNodePosition(dataset_id=mine, record_id=1, x=1, y=1))
        db.session.add(CanvasNodePosition(dataset_id=theirs, record_id=2, x=2, y=2))
        db.session.commit()
    assert set(client.get('/api/train/canvas/positions')
               .get_json()['positions']) == {str(mine)}


# ---- the delete-500 guard --------------------------------------------------

def test_deleting_a_dataset_that_has_canvas_positions_does_not_500(client, app):
    """The regression this table was most likely to reintroduce.

    foreign_keys=OFF reproduces a database created before ON DELETE CASCADE: if
    the parent DELETE were emitted first the children would be left dangling and
    an enforcing database would raise. The dataset must delete cleanly and take
    its position rows with it."""
    from app.extensions import db
    from app.models import CanvasNodePosition
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds_id = _dataset().id
        for i in range(1, 4):
            db.session.add(CanvasNodePosition(dataset_id=ds_id, record_id=i, x=i, y=i))
        db.session.commit()
        assert _positions(ds_id) == 3

        db.session.execute(text('PRAGMA foreign_keys=OFF'))
        assert db.session.execute(text('PRAGMA foreign_keys')).scalar_one() == 0
        db.session.commit()

        assert svc.delete_dataset('local', ds_id) is True
        assert _positions(ds_id) == 0
        assert svc.get_dataset('local', ds_id) is None


def test_deleting_a_dataset_through_the_api_still_answers_200(client, app):
    """The user-visible half of the same guard: the HTTP path, not the service."""
    from app.extensions import db
    from app.models import CanvasNodePosition
    with app.app_context():
        ds_id = _dataset().id
        db.session.add(CanvasNodePosition(dataset_id=ds_id, record_id=42, x=9, y=9))
        db.session.commit()
    assert client.post(f'/api/dataset/{ds_id}/delete').status_code == 200
    with app.app_context():
        assert _positions(ds_id) == 0


def test_the_position_model_declares_its_relationship_to_face_dataset():
    """A contract, not a style check. Without this mapper-level relationship the
    unit of work has no reason to delete the children first, and the explicit
    cleanup in delete_dataset becomes the ONLY thing standing between a user and
    an HTTP 500 — a single refactor away from the bug coming back."""
    from app.models import CanvasNodePosition, FaceDataset
    from sqlalchemy import inspect
    rels = inspect(CanvasNodePosition).relationships
    assert 'dataset' in rels, 'CanvasNodePosition must declare relationship() to face_dataset'
    assert rels['dataset'].mapper.class_ is FaceDataset
