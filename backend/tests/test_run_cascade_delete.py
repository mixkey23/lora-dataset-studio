"""🗑 Deleting a run AND everything it produced — the explicit cascade.

Removing a run used to mean ONE thing: tidying a ghost out of the graph. A run
whose checkpoints were still on disk was refused (409, "delete those first") and
generated images were only unlinked. That is still the default and this suite
guards it — but it is no longer the only mode: `?cascade=1` is the "delete
everything" a user asks for from the run panel.

The contracts held here are the ones that would hurt if they broke:

  * the cascade REALLY takes the weights (the assertion that fails on the old
    code: same run, conservative delete = 409 and files intact, cascade = 200 and
    the run dir empty);
  * the CONSERVATIVE mode is byte-for-byte what it was — no caller starts
    shredding files because a keyword gained a default. This is the most
    important anti-regression test of the pass;
  * children that resumed from the run SURVIVE, keep their own checkpoints and
    stay usable (detached, re-rooted — never deleted with their parent);
  * a dataset that is training right now is REFUSED before anything is touched;
  * a 👍 image and a LoRA already deployed into ComfyUI are KEPT — destroying
    something the user explicitly kept is worse than leaving an orphan;
  * a file that cannot be moved yields a PARTIAL (409) that keeps the run row, so
    the remaining weights stay reachable from the card that owns them;
  * the FK children are flushed BEFORE the parent row (this repo's "delete 500"
    trap), asserted with foreign_keys ON so a missing flush order is fatal here
    instead of silent.

Nothing touches a real install: every path is a pytest tmp dir and the
checkpoint scan is pointed at it."""
import os

import pytest


# ---- fixtures over a temporary run dir --------------------------------------

def _dataset(name='Nova', trigger='nova'):
    from app.services import face_dataset_service as svc
    return svc.create_dataset('local', name, trigger)


def _rec(dataset_id, steps=1000, version=1, parent=None):
    from app.models import TrainingRunRecord
    from app.extensions import db
    r = TrainingRunRecord(dataset_id=dataset_id, family='zimage', source='local',
                          base_model='', variant='turbo', steps=steps,
                          version=version, fingerprint='fp', manifest='[]',
                          parent_record_id=parent)
    db.session.add(r)
    db.session.commit()
    return r


def _fake_run_dir(monkeypatch, tmp_path, by_record):
    """Point the checkpoint scan at a tmp folder and hand each file to a record.

    `by_record` is {record_id: [filenames]}. Patching `list_checkpoints` and
    `_run_dir` on the lora_training module is enough for the WHOLE real path —
    the graph's "still on disk?" probe, the cascade's file enumeration and
    `delete_checkpoint`'s anti path-traversal whitelist all read these two."""
    from app.services import lora_training as lt
    run_dir = tmp_path / 'run'
    run_dir.mkdir(exist_ok=True)
    for rid, names in by_record.items():
        for n in names:
            (run_dir / n).write_bytes(b'W' * 128)

    def _list(user_id, dataset_id, base_model=None, family=None, variant=None):
        out = []
        for rid, names in by_record.items():
            for n in names:
                if (run_dir / n).exists():
                    out.append({'filename': n, 'step': 500, 'final': False,
                                'run_source': 'local', 'run_id': rid})
        return out

    monkeypatch.setattr(lt, 'list_checkpoints', _list)
    monkeypatch.setattr(lt, '_run_dir', lambda *a, **k: str(run_dir))
    return run_dir


@pytest.fixture()
def no_deploy_probe(monkeypatch):
    """The deployment join reads the ComfyUI pool; in tests there is none. Stub it
    to "nothing deployed" so the impact preview is deterministic."""
    from app.services import cloud_training as ct
    monkeypatch.setattr(ct, 'annotate_deployed_checkpoints',
                        lambda *a, **k: list(a[2] if len(a) > 2 else []))
    return True


# ---- THE central assertion: conservative refuses, cascade takes -------------

def test_conservative_delete_refuses_a_run_with_checkpoints_on_disk(
        client, app, tmp_path, monkeypatch, no_deploy_probe):
    """Unchanged behaviour, asserted on the very run the cascade will take: 409,
    the run row still there and NOT one byte removed."""
    from app.models import TrainingRunRecord
    with app.app_context():
        ds = _dataset()
        rec = _rec(ds.id)
        rid = rec.id
        run_dir = _fake_run_dir(monkeypatch, tmp_path, {rid: ['a-000500.safetensors']})

    r = client.delete(f'/api/dataset/train/runs/{rid}')
    assert r.status_code == 409
    assert (run_dir / 'a-000500.safetensors').exists()
    with app.app_context():
        from app.extensions import db
        assert db.session.get(TrainingRunRecord, rid) is not None


def test_cascade_takes_the_checkpoints_and_the_run(
        client, app, tmp_path, monkeypatch, no_deploy_probe):
    """The new mode: the same run, `?cascade=1`, and the weights are gone with it."""
    from app.models import TrainingRunRecord
    with app.app_context():
        ds = _dataset()
        rid = _rec(ds.id).id
        run_dir = _fake_run_dir(monkeypatch, tmp_path,
                                {rid: ['a-000500.safetensors', 'a-001000.safetensors']})

    r = client.delete(f'/api/dataset/train/runs/{rid}?cascade=1')
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body['checkpoints_deleted'] == 2
    assert body['checkpoints_failed'] == 0
    assert not (run_dir / 'a-000500.safetensors').exists()
    assert not (run_dir / 'a-001000.safetensors').exists()
    with app.app_context():
        from app.extensions import db
        assert db.session.get(TrainingRunRecord, rid) is None


def test_cascade_deletes_the_generated_images(
        client, app, tmp_path, monkeypatch, no_deploy_probe):
    """The images the run produced go too — row AND file — unlike the
    conservative mode, which only unlinks them."""
    from app.models import LoraTestImage
    from app.services import face_dataset_service as fds
    from app.extensions import db
    with app.app_context():
        ds = _dataset()
        rid, dsid = _rec(ds.id).id, ds.id
        _fake_run_dir(monkeypatch, tmp_path, {rid: ['a-000500.safetensors']})
        folder = fds._dataset_path(dsid)
        os.makedirs(folder, exist_ok=True)
        open(os.path.join(folder, 'img.png'), 'wb').write(b'PNG')
        db.session.add(LoraTestImage(dataset_id=dsid, checkpoint='a.safetensors',
                                     strength=1.0, status='done', filename='img.png',
                                     record_id=rid, step=500))
        db.session.commit()

    r = client.delete(f'/api/dataset/train/runs/{rid}?cascade=1')
    assert r.status_code == 200, r.get_json()
    assert r.get_json()['images_deleted'] == 1
    with app.app_context():
        assert LoraTestImage.query.filter_by(dataset_id=dsid).count() == 0


def test_cascade_keeps_a_liked_image_and_unlinks_it(
        client, app, tmp_path, monkeypatch, no_deploy_probe):
    """👍 is the one explicit "I am keeping this" the app records. The picture
    survives; only its provenance is cut."""
    from app.models import LoraTestImage
    from app.extensions import db
    with app.app_context():
        ds = _dataset()
        rid, dsid = _rec(ds.id).id, ds.id
        _fake_run_dir(monkeypatch, tmp_path, {rid: []})
        db.session.add(LoraTestImage(dataset_id=dsid, checkpoint='a.safetensors',
                                     strength=1.0, status='done', filename='keep.png',
                                     rating=1, record_id=rid, step=500))
        db.session.add(LoraTestImage(dataset_id=dsid, checkpoint='a.safetensors',
                                     strength=1.0, status='done', filename='miss.png',
                                     record_id=rid, step=500))
        db.session.commit()

    r = client.delete(f'/api/dataset/train/runs/{rid}?cascade=1')
    assert r.status_code == 200, r.get_json()
    assert r.get_json()['images_kept'] == 1
    with app.app_context():
        rows = LoraTestImage.query.filter_by(dataset_id=dsid).all()
        assert [x.filename for x in rows] == ['keep.png']
        assert rows[0].record_id is None and rows[0].step is None


# ---- children survive and stay usable ---------------------------------------

def test_children_survive_the_cascade_and_keep_their_checkpoints(
        client, app, tmp_path, monkeypatch, no_deploy_probe):
    """A run that resumed from the deleted one keeps its row, its own weights and
    its notes — it is re-rooted, not destroyed."""
    from app.models import CheckpointNote, TrainingRunRecord
    from app.extensions import db
    with app.app_context():
        ds = _dataset()
        parent = _rec(ds.id)
        child = _rec(ds.id, version=2, parent=parent.id)
        pid, cid = parent.id, child.id
        db.session.add(CheckpointNote(record_id=cid, step=500, note='child note'))
        db.session.commit()
        run_dir = _fake_run_dir(monkeypatch, tmp_path,
                                {pid: ['p-000500.safetensors'],
                                 cid: ['c-000500.safetensors']})

    assert client.delete(f'/api/dataset/train/runs/{pid}?cascade=1').status_code == 200

    with app.app_context():
        kid = db.session.get(TrainingRunRecord, cid)
        assert kid is not None
        assert kid.parent_record_id is None           # detached, re-rooted
        assert CheckpointNote.query.filter_by(record_id=cid).count() == 1
    # Its OWN checkpoint is untouched — only the parent's went to the trash.
    assert (run_dir / 'c-000500.safetensors').exists()
    assert not (run_dir / 'p-000500.safetensors').exists()
    # And it is still usable: the graph answers for it.
    assert client.get(f'/api/dataset/train/runs/{cid}/lineage').status_code == 200


# ---- refusals ----------------------------------------------------------------

def test_cascade_is_refused_while_the_dataset_is_training(
        client, app, tmp_path, monkeypatch, no_deploy_probe):
    """Files ai-toolkit is about to rewrite are never touched: 409, and the
    checkpoint is still there."""
    from app.services import lora_training as lt
    with app.app_context():
        ds = _dataset()
        rid = _rec(ds.id).id
        run_dir = _fake_run_dir(monkeypatch, tmp_path, {rid: ['a-000500.safetensors']})
    monkeypatch.setattr(lt, '_local_training_active_for', lambda dsid: True)

    r = client.delete(f'/api/dataset/train/runs/{rid}?cascade=1')
    assert r.status_code == 409
    assert 'training right now' in r.get_json()['error']
    assert (run_dir / 'a-000500.safetensors').exists()


def test_cascade_on_an_unknown_run_is_404(client, app):
    assert client.delete('/api/dataset/train/runs/999999?cascade=1').status_code == 404


def test_a_file_that_cannot_be_moved_keeps_the_run(
        client, app, tmp_path, monkeypatch, no_deploy_probe):
    """PARTIAL is an error, not a 200 with a sad number: the row stays so the
    remaining weights are still reachable from the card that owns them."""
    from app.models import TrainingRunRecord
    from app.services import lora_training as lt
    from app.extensions import db
    with app.app_context():
        ds = _dataset()
        rid = _rec(ds.id).id
        _fake_run_dir(monkeypatch, tmp_path, {rid: ['a-000500.safetensors']})

    def _boom(*a, **k):
        raise OSError('file is locked')
    monkeypatch.setattr(lt, 'delete_checkpoint', _boom)

    r = client.delete(f'/api/dataset/train/runs/{rid}?cascade=1')
    assert r.status_code == 409
    assert r.get_json()['checkpoints_failed'] == 1
    with app.app_context():
        assert db.session.get(TrainingRunRecord, rid) is not None


# ---- the "delete 500" flush order -------------------------------------------

def test_child_rows_are_flushed_before_the_parent_with_fk_enforced(
        client, app, tmp_path, monkeypatch, no_deploy_probe):
    """The repo's known trap: these tables carry a `record_id` with no declared
    relationship(), so deleting the parent first raises IntegrityError. Run with
    foreign_keys ON so a wrong order is FATAL here instead of silent."""
    from app.models import (CanvasNodePosition, CheckpointNote, CheckpointPreview,
                            TrainingRunRecord)
    from app.extensions import db
    from sqlalchemy import event, text
    engine = None
    with app.app_context():
        engine = db.engine
        db.session.execute(text('PRAGMA foreign_keys=ON'))
        ds = _dataset()
        rid, dsid = _rec(ds.id).id, ds.id
        db.session.add(CheckpointNote(record_id=rid, step=500, note='n'))
        db.session.add(CheckpointPreview(record_id=rid, step=500, dataset_id=dsid,
                                         prompt='p'))
        db.session.add(CanvasNodePosition(dataset_id=dsid, record_id=rid, x=1, y=2))
        db.session.commit()
        _fake_run_dir(monkeypatch, tmp_path, {rid: ['a-000500.safetensors']})

    @event.listens_for(engine, 'connect')
    def _fk_on(dbapi_conn, _rec_):                     # pragma: no cover - trivial
        dbapi_conn.execute('PRAGMA foreign_keys=ON')

    try:
        r = client.delete(f'/api/dataset/train/runs/{rid}?cascade=1')
        assert r.status_code == 200, r.get_json()
        with app.app_context():
            assert db.session.get(TrainingRunRecord, rid) is None
            assert CheckpointNote.query.filter_by(record_id=rid).count() == 0
            assert CheckpointPreview.query.filter_by(record_id=rid).count() == 0
            assert CanvasNodePosition.query.filter_by(record_id=rid).count() == 0
    finally:
        event.remove(engine, 'connect', _fk_on)


# ---- the confirmation payload -----------------------------------------------

def test_impact_counts_what_the_cascade_would_take(
        client, app, tmp_path, monkeypatch, no_deploy_probe):
    """The dialog gets figures, not a generic "are you sure": checkpoints, their
    size on disk, the images that go and the ones that stay."""
    from app.models import LoraTestImage
    from app.extensions import db
    with app.app_context():
        ds = _dataset()
        rid, dsid = _rec(ds.id).id, ds.id
        _fake_run_dir(monkeypatch, tmp_path,
                      {rid: ['a-000500.safetensors', 'a-001000.safetensors']})
        for rating in (0, 0, 1):
            db.session.add(LoraTestImage(dataset_id=dsid, checkpoint='a.safetensors',
                                         strength=1.0, status='done', filename='i.png',
                                         rating=rating, record_id=rid, step=500))
        db.session.commit()

    body = client.get(f'/api/dataset/train/runs/{rid}/deletion-impact').get_json()
    cas = body['cascade']
    assert cas['checkpoints'] == 2
    assert cas['checkpoint_bytes'] == 256
    assert cas['images_deleted'] == 2
    assert cas['images_kept_rated'] == 1
    assert cas['training_active'] is None
    # The conservative half of the same payload is untouched.
    assert body['has_saves'] is True


def test_impact_cascade_block_is_zeroed_on_a_bare_run(client, app):
    """No dataset folder, no ai-toolkit, nothing trained: a zeroed block, never a
    missing key and never a 500."""
    with app.app_context():
        ds = _dataset()
        rid = _rec(ds.id).id
    body = client.get(f'/api/dataset/train/runs/{rid}/deletion-impact').get_json()
    for key in ('checkpoints', 'checkpoint_bytes', 'images_deleted',
                'images_kept_rated', 'deployed_kept'):
        assert body['cascade'][key] == 0, key
