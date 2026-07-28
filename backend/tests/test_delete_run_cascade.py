"""Deleting a run must take EVERYTHING that only existed for it — and nothing
that belongs to someone else.

Five tables carry a `record_id`. Before this suite, three of them survived a
deletion: checkpoint previews, the card's canvas position, and the image→run
provenance link. The run vanished from the graph and kept haunting the canvas
and the checkpoint gallery.

The tests below hold the four contracts of that cleanup:

  * every child table of the run is cleared (and none of another run's rows);
  * generated images are UNLINKED, never destroyed — they are real pictures the
    user also sees in the Test Studio, and removing a run is a tidying of
    lineage, not an order to delete images;
  * an archived source blob is freed only when this run was its LAST referrer.
    The archive is content-addressed and shared on purpose (that is why a whole
    history fits in a fraction of a gigabyte), so a naive delete would blank
    another run's comparison — the sharing test is the load-bearing one;
  * the deletion-impact preview counts exactly what the dialog announces, and
    degrades to zeros on an install where none of this data exists.

The deletions run with PRAGMA foreign_keys=OFF where a legacy-shaped database
matters, mirroring test_canvas_positions: a missing flush order is fatal there
and silent everywhere else."""
import json


def _dataset(name='Ada', trigger='ada'):
    from app.services import face_dataset_service as svc
    return svc.create_dataset('local', name, trigger)


def _rec(dataset_id=1, steps=1000, version=1, parent=None, snapshot=None):
    from app.models import TrainingRunRecord
    from app.extensions import db
    r = TrainingRunRecord(
        dataset_id=dataset_id, family='zimage', source='local', base_model='',
        variant='turbo', steps=steps, version=version, fingerprint='fp',
        manifest='[]', parent_record_id=parent)
    if snapshot is not None:
        r.snapshot = json.dumps(snapshot)
    db.session.add(r)
    db.session.commit()
    return r


def _snapshot_with(sigs, reference=None):
    """A minimal run snapshot that references `sigs` as its training images."""
    snap = {'v': 1, 'images': {str(i): {'c': s} for i, s in enumerate(sigs, 1)}}
    if reference:
        snap['dataset'] = {'reference': {'filename': 'ref.png', 'c': reference}}
    return snap


def _blob(app, sig, ext='.png', data=b'IMG'):
    """Write a blob into the archive at the address `sig` resolves to."""
    from app.services import run_archive
    p = run_archive.archive_root() / sig[:2] / f'{sig}{ext}'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


# ---- every child table goes ------------------------------------------------

def test_delete_clears_previews_canvas_and_unlinks_images(client, app):
    """The three tables that used to be left behind: preview links and the canvas
    position are deleted, the generated images survive with a NULL provenance."""
    from app.models import (CheckpointPreview, CanvasNodePosition, LoraTestImage,
                            TrainingRunRecord)
    from app.extensions import db
    with app.app_context():
        ds = _dataset()
        rec = _rec(dataset_id=ds.id, steps=2000)
        rid, dsid = rec.id, ds.id
        db.session.add(CheckpointPreview(record_id=rid, step=1000, dataset_id=dsid,
                                         prompt='p', seed=1))
        db.session.add(CheckpointPreview(record_id=rid, step=1000, dataset_id=dsid,
                                         prompt='p', seed=2))   # previews accumulate
        db.session.add(CanvasNodePosition(dataset_id=dsid, record_id=rid, x=1, y=2))
        db.session.add(LoraTestImage(dataset_id=dsid, checkpoint='a.safetensors',
                                     strength=1.0, filename='a.png',
                                     record_id=rid, step=1000))
        db.session.commit()

    assert client.delete(f'/api/dataset/train/runs/{rid}').status_code == 200

    with app.app_context():
        assert db.session.get(TrainingRunRecord, rid) is None
        assert CheckpointPreview.query.filter_by(record_id=rid).count() == 0
        assert CanvasNodePosition.query.filter_by(record_id=rid).count() == 0
        # The picture is still there — only its provenance was cut.
        img = LoraTestImage.query.filter_by(dataset_id=dsid).one()
        assert img.filename == 'a.png'
        assert img.record_id is None and img.step is None


def test_delete_leaves_another_runs_rows_alone(client, app):
    """A neighbour run keeps its preview, its canvas position and its linked
    images — the cleanup is scoped to the deleted record, not the dataset."""
    from app.models import CheckpointPreview, CanvasNodePosition, LoraTestImage
    from app.extensions import db
    with app.app_context():
        ds = _dataset()
        doomed, keeper = _rec(dataset_id=ds.id), _rec(dataset_id=ds.id, version=2)
        rid, kid, dsid = doomed.id, keeper.id, ds.id
        for r in (rid, kid):
            db.session.add(CheckpointPreview(record_id=r, step=1000,
                                             dataset_id=dsid, prompt='p'))
            db.session.add(CanvasNodePosition(dataset_id=dsid, record_id=r, x=1, y=2))
            db.session.add(LoraTestImage(dataset_id=dsid, checkpoint='a.safetensors',
                                         strength=1.0, record_id=r, step=1000))
        db.session.commit()

    assert client.delete(f'/api/dataset/train/runs/{rid}').status_code == 200

    with app.app_context():
        assert CheckpointPreview.query.filter_by(record_id=kid).count() == 1
        assert CanvasNodePosition.query.filter_by(record_id=kid).count() == 1
        assert LoraTestImage.query.filter_by(record_id=kid).count() == 1


def test_delete_on_a_run_with_nothing_attached_still_succeeds(client, app):
    """A never-previewed, never-tested run on a fresh install: no table has a
    row for it and the deletion is a plain 200 (no install-shape assumption)."""
    with app.app_context():
        rid = _rec().id
    assert client.delete(f'/api/dataset/train/runs/{rid}').status_code == 200


# ---- archived blobs: the sharing guard -------------------------------------

def test_archived_blob_shared_with_another_run_is_NOT_deleted(client, app):
    """THE load-bearing one. Two runs trained on the same image; deleting the
    first must leave the blob the second still shows in its comparison."""
    from app.services import run_archive
    with app.app_context():
        shared, only_mine = 'aaaa1111', 'bbbb2222'
        doomed = _rec(snapshot=_snapshot_with([shared, only_mine]))
        _rec(version=2, snapshot=_snapshot_with([shared]))
        rid = doomed.id
        _blob(app, shared)
        _blob(app, only_mine)

    assert client.delete(f'/api/dataset/train/runs/{rid}').status_code == 200

    with app.app_context():
        assert run_archive.path_for('aaaa1111') is not None   # still referenced
        assert run_archive.path_for('bbbb2222') is None       # last referrer gone


def test_reference_photo_blob_counts_as_a_reference(client, app):
    """A dataset's reference photo is an input of the run like any image — a run
    that only names it in `dataset.reference` still protects its blob."""
    from app.services import run_archive
    with app.app_context():
        sig = 'cccc3333'
        doomed = _rec(snapshot=_snapshot_with([sig]))
        _rec(version=2, snapshot=_snapshot_with([], reference=sig))
        rid = doomed.id
        _blob(app, sig)

    assert client.delete(f'/api/dataset/train/runs/{rid}').status_code == 200

    with app.app_context():
        assert run_archive.path_for(sig) is not None


def test_legacy_run_without_snapshot_releases_nothing(client, app):
    """A run that predates snapshots cannot prove which blobs are its own, so
    NOTHING is released — extra bytes beat a hole in the archive."""
    from app.services import run_archive
    with app.app_context():
        rid = _rec().id                       # snapshot stays NULL
        _blob(app, 'dddd4444')

    assert client.delete(f'/api/dataset/train/runs/{rid}').status_code == 200

    with app.app_context():
        assert run_archive.path_for('dddd4444') is not None


def test_delete_survives_a_missing_archive_folder(client, app):
    """`run_images/` never created (archiving disabled, or nothing trained yet):
    the release step is a no-op, not a 500."""
    with app.app_context():
        rid = _rec(snapshot=_snapshot_with(['eeee5555'])).id
    assert client.delete(f'/api/dataset/train/runs/{rid}').status_code == 200


# ---- the confirmation payload ----------------------------------------------

def test_deletion_impact_counts_what_the_dialog_announces(client, app):
    """The preview endpoint reports each figure the confirmation prints."""
    from app.models import CheckpointNote, CheckpointPreview, CanvasNodePosition, LoraTestImage
    from app.extensions import db
    with app.app_context():
        ds = _dataset()
        rec = _rec(dataset_id=ds.id, snapshot=_snapshot_with(['ffff6666']))
        rid, dsid = rec.id, ds.id
        _rec(dataset_id=dsid, version=2, parent=rid)          # a child to detach
        db.session.add(CheckpointNote(record_id=rid, step=1000, note='n'))
        db.session.add(CheckpointPreview(record_id=rid, step=1000,
                                         dataset_id=dsid, prompt='p'))
        db.session.add(CanvasNodePosition(dataset_id=dsid, record_id=rid, x=0, y=0))
        for _ in range(3):
            db.session.add(LoraTestImage(dataset_id=dsid, checkpoint='a.safetensors',
                                         strength=1.0, record_id=rid, step=1000))
        db.session.commit()
        _blob(app, 'ffff6666')

    body = client.get(f'/api/dataset/train/runs/{rid}/deletion-impact').get_json()
    assert body['notes'] == 1
    assert body['previews'] == 1
    assert body['canvas_positions'] == 1
    assert body['images_unlinked'] == 3
    assert body['children_detached'] == 1
    assert body['archived_images_released'] == 1
    assert body['has_saves'] is False


def test_deletion_impact_is_all_zeros_on_a_bare_run(client, app):
    """Nothing attached anywhere → every count is 0, no key missing."""
    with app.app_context():
        rid = _rec().id
    body = client.get(f'/api/dataset/train/runs/{rid}/deletion-impact').get_json()
    for key in ('notes', 'previews', 'canvas_positions', 'images_unlinked',
                'children_detached', 'archived_images_released'):
        assert body[key] == 0, key


def test_deletion_impact_unknown_run_is_404(client, app):
    assert client.get('/api/dataset/train/runs/999999/deletion-impact').status_code == 404
