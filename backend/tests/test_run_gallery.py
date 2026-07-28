"""🖼 The RUN-scoped gallery — everything one training run ever generated, in
one panel, grouped by the checkpoint that made it.

What is asserted here is what a second reader would have got wrong:

  * the GROUPING and its order (steps descending, the step-less group last);
  * the CAPS — a run with a dozen checkpoints must answer something bounded, and
    it must say that it cut rather than look complete;
  * that the delete is THE checkpoint delete with a wider scope, refusing ids
    from another run exactly as the narrow one refuses ids from another step;
  * that an image attributable to a RUN but not to a step stops being counted as
    "could not be traced back to a checkpoint" — it is traced, just not that far.
"""


def _create(client, name='Nova', trigger='nova'):
    return client.post('/api/dataset/create',
                       json={'name': name, 'trigger_word': trigger}).get_json()['id']


def _record(db, dataset_id, source='local', cloud_run_id=None, family='krea'):
    from app.models import TrainingRunRecord
    rec = TrainingRunRecord(dataset_id=dataset_id, family=family, source=source,
                            cloud_run_id=cloud_run_id, fingerprint='f', version=1)
    db.session.add(rec)
    db.session.commit()
    return rec


def _image(db, dataset_id, checkpoint='a.safetensors', **kw):
    from app.models import LoraTestImage
    img = LoraTestImage(dataset_id=dataset_id, checkpoint=checkpoint, strength=1.0,
                        status=kw.pop('status', 'done'),
                        filename=kw.pop('filename', 'x.png'), **kw)
    db.session.add(img)
    db.session.commit()
    return img


# --- grouping ----------------------------------------------------------------

def test_the_run_gallery_groups_by_step_most_trained_first(client, app):
    from app.extensions import db
    from app.services import cloud_training as ct
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        other = _record(db, ds)
        _image(db, ds, record_id=rec.id, step=500, filename='a.png')
        _image(db, ds, record_id=rec.id, step=2000, filename='b.png')
        newest_2000 = _image(db, ds, record_id=rec.id, step=2000, filename='c.png')
        _image(db, ds, record_id=rec.id, step=1000, filename='d.png')
        # Another run's image must never appear under this one.
        _image(db, ds, record_id=other.id, step=2000, filename='e.png')
        # A cell still generating is not an image yet.
        _image(db, ds, record_id=rec.id, step=500, status='pending', filename=None)
        db.session.commit()

        out = ct.run_gallery(rec.id)
        assert [g['step'] for g in out['groups']] == [2000, 1000, 500]
        assert [g['count'] for g in out['groups']] == [2, 1, 1]
        assert out['count'] == 4
        assert out['truncated'] is False
        # Newest first INSIDE a step, like every other gallery in the app.
        assert out['groups'][0]['images'][0]['id'] == newest_2000.id
        assert out['groups'][0]['images'][0]['step'] == 2000


def test_images_tied_to_the_run_but_to_no_step_get_their_own_group_LAST(client, app):
    from app.extensions import db
    from app.services import cloud_training as ct
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        _image(db, ds, record_id=rec.id, step=3000, filename='a.png')
        stepless = _image(db, ds, record_id=rec.id, step=None, filename='b.png')
        db.session.commit()

        out = ct.run_gallery(rec.id)
        assert [g['step'] for g in out['groups']] == [3000, None]
        assert out['groups'][-1]['images'][0]['id'] == stepless.id
        # …and it stays out of every checkpoint gallery: those filter on a step.
        assert ct.checkpoint_gallery(rec.id, 3000)['count'] == 1


def test_a_run_with_nothing_is_an_empty_answer_not_an_error(client, app):
    from app.extensions import db
    from app.services import cloud_training as ct
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        db.session.commit()
        out = ct.run_gallery(rec.id)
        assert (out['count'], out['groups'], out['truncated']) == (0, [], False)


def test_checkpoint_notes_ride_along_and_outlive_the_files(client, app):
    """A note is keyed by (run, step) and survives the save being deleted. Read
    from the pills, the panel of a cleaned-up run would show its images and
    silently lose everything written about them."""
    from app.extensions import db
    from app.models import CheckpointNote
    from app.services import cloud_training as ct
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        _image(db, ds, record_id=rec.id, step=2000, filename='a.png')
        db.session.add_all([
            CheckpointNote(record_id=rec.id, step=2000, note='soft eyes'),
            # A step that produced no image at all still has its note carried.
            CheckpointNote(record_id=rec.id, step=3500, note='shipping this one'),
            CheckpointNote(record_id=rec.id, step=500, note='   '),
            CheckpointNote(record_id=999, step=2000, note='another run'),
        ])
        db.session.commit()

        out = ct.run_gallery(rec.id)
        assert out['checkpoint_notes'] == [{'step': 3500, 'note': 'shipping this one'},
                                           {'step': 2000, 'note': 'soft eyes'}]
        assert out['groups'][0]['note'] == 'soft eyes'


# --- the volume --------------------------------------------------------------

def test_a_big_run_is_capped_per_step_and_overall_and_SAYS_so(client, app):
    from app.extensions import db
    from app.services import cloud_training as ct
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        for step in (1000, 2000):
            for i in range(6):
                _image(db, ds, record_id=rec.id, step=step, filename=f'{step}-{i}.png')
        db.session.commit()

        out = ct.run_gallery(rec.id, per_step=2)
        # Counts stay EXACT — only the images are cut.
        assert [g['count'] for g in out['groups']] == [6, 6]
        assert [len(g['images']) for g in out['groups']] == [2, 2]
        assert all(g['truncated'] for g in out['groups'])
        assert (out['count'], out['shown'], out['truncated']) == (12, 4, True)

        # The overall budget cuts the DEEPEST steps, never the freshest — the
        # panel opens on what the user is judging.
        tight = ct.run_gallery(rec.id, limit=3, per_step=2)
        assert [len(g['images']) for g in tight['groups']] == [2, 1]
        assert tight['shown'] == 3


# --- the routes --------------------------------------------------------------

def test_the_run_gallery_route_answers_for_a_run_that_does_not_exist(client):
    r = client.get('/api/train/run/424242/images')
    assert r.status_code == 200
    body = r.get_json()
    assert body.pop('delete_mode') in ('trash', 'app_trash')
    assert body['count'] == 0 and body['groups'] == []
    assert body['record_id'] == 424242


def test_the_run_delete_route_refuses_a_body_that_is_not_a_list(client):
    r = client.post('/api/train/run/1/images/delete', json={'image_ids': 'all'})
    assert r.status_code == 400


# --- the delete, widened -----------------------------------------------------

def test_the_run_delete_spans_every_step_and_refuses_another_run_s_images(client, app):
    from app.extensions import db
    from app.models import LoraTestImage
    from app.services import cloud_training as ct
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        other = _record(db, ds)
        a = _image(db, ds, record_id=rec.id, step=1000, filename='a.png')
        b = _image(db, ds, record_id=rec.id, step=2000, filename='b.png')
        c = _image(db, ds, record_id=rec.id, step=None, filename='c.png')
        foreign = _image(db, ds, record_id=other.id, step=2000, filename='d.png')
        db.session.commit()

        out = ct.delete_checkpoint_images(rec.id, None, [a.id, b.id, c.id, foreign.id])
        assert out['rows_removed'] == 3
        assert [s['reason'] for s in out['skipped']] == ['not_in_gallery']
        assert [s['id'] for s in out['skipped']] == [foreign.id]
        assert db.session.get(LoraTestImage, foreign.id) is not None
        assert ct.run_gallery(rec.id)['count'] == 0


def test_the_narrow_checkpoint_delete_is_unchanged_by_the_widening(client, app):
    """Non-regression: a step-scoped delete still refuses the run's OTHER steps."""
    from app.extensions import db
    from app.models import LoraTestImage
    from app.services import cloud_training as ct
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        mine = _image(db, ds, record_id=rec.id, step=1000, filename='a.png')
        sibling = _image(db, ds, record_id=rec.id, step=2000, filename='b.png')
        db.session.commit()

        out = ct.delete_checkpoint_images(rec.id, 1000, [mine.id, sibling.id])
        assert out['rows_removed'] == 1
        assert out['skipped'] == [{'id': sibling.id, 'reason': 'not_in_gallery'}]
        assert db.session.get(LoraTestImage, sibling.id) is not None


# --- the backfill pass that makes the step-less group possible ---------------

def test_a_step_less_final_save_now_names_its_RUN_instead_of_being_orphaned(client, app):
    from app.extensions import db
    from app.models import LoraTestImage
    from app.services import checkpoint_link_backfill as bf
    from app.services import cloud_training as ct
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds, source='local')
        # Tagged with the run, no step anywhere in the name — pass 2 gives up on
        # it, pass 3 keeps the half that IS known.
        final = _image(db, ds, f'krea\\lora_nova_Krea-2-Raw_rl{rec.id}_v2.safetensors',
                       filename='final.png')
        # Still unattributable: no run tag at all.
        orphan = _image(db, ds, 'krea\\lora_nova_Krea-2-Raw.safetensors',
                        filename='orphan.png')
        db.session.commit()

        out = bf.backfill_checkpoint_links()
        assert out['by_run'] == 1
        assert out['by_name'] == 0
        db.session.expire_all()
        row = db.session.get(LoraTestImage, final.id)
        assert (row.record_id, row.step) == (rec.id, None)
        assert db.session.get(LoraTestImage, orphan.id).record_id is None
        # The honest footnote shrinks by exactly what stopped being a mystery.
        assert bf.unlinked_count() == 1
        assert [g['step'] for g in ct.run_gallery(rec.id)['groups']] == [None]


def test_resolve_run_name_answers_where_resolve_checkpoint_name_gives_up(client, app):
    from app.extensions import db
    from app.services.checkpoint_link_backfill import (
        resolve_checkpoint_name, resolve_run_name)
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds, source='local')
        cloud = _record(db, ds, source='cloud', cloud_run_id=74)
        stepless = f'krea\\lora_nova_Krea-2-Raw_rl{rec.id}_v2.safetensors'
        assert resolve_checkpoint_name(stepless) is None
        assert resolve_run_name(stepless) == (rec.id, ds)
        assert resolve_run_name('krea\\lora_nova_Krea_rc74_v1.safetensors') == (cloud.id, ds)
        # No tag, or a tag pointing nowhere: still nothing.
        assert resolve_run_name('krea\\lora_nova_Krea-2-Raw.safetensors') is None
        assert resolve_run_name('krea\\lora_nova_Krea_rl9999_v1.safetensors') is None
