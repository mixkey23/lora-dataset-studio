"""Retrofit pass for images promoted from a 🗃️ Bank BEFORE the promotion carried
their framing (bug reported by axelf_ on Discord): they landed with framing NULL
and stayed invisible in the Composition tally. Fixing the promotion path only
helps the next promotion — this pass repairs the datasets already on disk.

These cases are deliberately about the AWKWARD databases, not the happy path: the
same code runs on installs nobody here will ever see. A value already present is
sacred (a user may have classified by hand), a verdict outside the four buckets
is left NULL (the classifier can still pick the row up), a dangling bank link is
a normal state, and a schema that predates the image bank must not break boot.
"""
from app.extensions import db
from app.models import (BankImage, FaceDataset, FaceDatasetImage, ImageBank,
                        SystemState)
from app.services import framing_backfill as fb


def _clear_flag():
    """create_app already ran the one-shot pass on the empty fixture DB (flag set,
    0 rows). Drop it so a test can exercise a not-yet-processed database."""
    row = db.session.get(SystemState, fb._STATE_KEY)
    if row is not None:
        db.session.delete(row)
        db.session.commit()


def _dataset():
    ds = FaceDataset(user_id='local', name='D', trigger_word='trg')
    db.session.add(ds)
    db.session.commit()
    return ds


def _bank():
    b = ImageBank(user_id='local', name='B', source_path='/nowhere')
    db.session.add(b)
    db.session.commit()
    return b


def _bank_image(bank, framing, relpath='a.png'):
    bi = BankImage(bank_id=bank.id, relpath=relpath, framing=framing)
    db.session.add(bi)
    db.session.commit()
    return bi


def _promoted(ds, bank_image_id, framing=None, filename='x.webp'):
    """A row as the OLD promotion path wrote it: linked to its bank image, no
    framing (unless the test is asserting an existing value is preserved)."""
    img = FaceDatasetImage(dataset_id=ds.id, source='import', status='keep',
                           filename=filename, framing=framing,
                           bank_image_id=bank_image_id)
    db.session.add(img)
    db.session.commit()
    return img


def _framing_of(img_id):
    db.session.expire_all()
    return db.session.get(FaceDatasetImage, img_id).framing


def test_fills_every_bucket_from_the_bank(app):
    """The nominal recovery: each of the four composition buckets is copied back
    onto the promoted row that lost it."""
    with app.app_context():
        ds, bank = _dataset(), _bank()
        ids = {}
        for n, bucket in enumerate(('face', 'bust', 'body', 'back')):
            bi = _bank_image(bank, bucket, relpath=f'{bucket}.png')
            ids[bucket] = _promoted(ds, bi.id, filename=f'{bucket}.webp').id

        assert fb.backfill_promoted_framings() == 4
        for bucket, img_id in ids.items():
            assert _framing_of(img_id) == bucket


def test_never_overwrites_an_existing_framing(app):
    """A framing already on the row wins, even when it disagrees with the bank:
    the user may have re-classified it by hand and we must not undo that."""
    with app.app_context():
        ds, bank = _dataset(), _bank()
        bi = _bank_image(bank, 'body')
        img = _promoted(ds, bi.id, framing='face')

        assert fb.backfill_promoted_framings() == 0
        assert _framing_of(img.id) == 'face'


def test_unknown_or_missing_bank_verdict_stays_null(app):
    """'unknown', NULL, or a verdict this build does not know are NOT guesses we
    are allowed to make — the row stays NULL so the dataset classifier can still
    take it."""
    with app.app_context():
        ds, bank = _dataset(), _bank()
        ids = []
        for n, verdict in enumerate(('unknown', None, '', 'FACE', 'closeup')):
            bi = _bank_image(bank, verdict, relpath=f'v{n}.png')
            ids.append(_promoted(ds, bi.id, filename=f'v{n}.webp').id)

        assert fb.backfill_promoted_framings() == 0
        assert [_framing_of(i) for i in ids] == [None] * len(ids)


def test_orphan_bank_link_is_ignored_not_an_error(app):
    """The bank (or just that image) was deleted after the promotion: the id
    dangles. Frequent, and not an anomaly — skip it silently and keep repairing
    the rows around it."""
    with app.app_context():
        ds, bank = _dataset(), _bank()
        good = _bank_image(bank, 'bust')
        recoverable = _promoted(ds, good.id, filename='ok.webp')
        orphan = _promoted(ds, 999_999, filename='orphan.webp')

        assert fb.backfill_promoted_framings() == 1
        assert _framing_of(recoverable.id) == 'bust'
        assert _framing_of(orphan.id) is None


def test_rows_that_never_came_from_a_bank_are_left_alone(app):
    """A plain import or a generated image has no bank link — there is nothing to
    recover from, and inventing a bucket is exactly what this pass must not do."""
    with app.app_context():
        ds = _dataset()
        imported = _promoted(ds, None, filename='plain.webp')
        generated = FaceDatasetImage(dataset_id=ds.id, source='generated',
                                     status='keep', filename='gen.webp')
        db.session.add(generated)
        db.session.commit()

        assert fb.backfill_promoted_framings() == 0
        assert _framing_of(imported.id) is None
        assert _framing_of(generated.id) is None


def test_second_run_changes_nothing(app):
    """Idempotent by construction: everything the pass could fill is no longer
    NULL, so re-running it — same boot or a later one — matches no row."""
    with app.app_context():
        ds, bank = _dataset(), _bank()
        bi = _bank_image(bank, 'back')
        img = _promoted(ds, bi.id)

        assert fb.backfill_promoted_framings() == 1
        assert fb.backfill_promoted_framings() == 0
        assert _framing_of(img.id) == 'back'


def test_empty_database_is_silent(app):
    """A fresh install: zero rows, zero work, zero noise."""
    with app.app_context():
        assert fb.backfill_promoted_framings() == 0


def test_schema_without_the_bank_table_does_not_raise(app):
    """An install that predates the image bank has no bank_image table to join.
    The pass must skip cleanly — a boot-time repair may never break startup."""
    with app.app_context():
        ds = _dataset()
        img = _promoted(ds, 42, filename='legacy.webp')
        db.session.commit()
        db.session.execute(db.text('DROP TABLE bank_image'))
        db.session.commit()

        assert fb._schema_supports_backfill() is False
        assert fb.backfill_promoted_framings() == 0
        assert _framing_of(img.id) is None


def test_run_if_needed_records_the_flag_and_short_circuits(app):
    """Guarded like the lineage backfill: the pass runs once, records what it did,
    and a restart re-reads the flag instead of re-scanning."""
    with app.app_context():
        _clear_flag()
        ds, bank = _dataset(), _bank()
        bi = _bank_image(bank, 'face')
        img = _promoted(ds, bi.id)

        first = fb.run_if_needed()
        assert first['framings'] == 1 and first['version'] == fb.BACKFILL_VERSION
        assert _framing_of(img.id) == 'face'

        # A later promotion that lost its framing is NOT re-scanned at this rule
        # version — the flag short-circuits, exactly like lineage_backfill.
        bi2 = _bank_image(bank, 'body', relpath='b2.png')
        later = _promoted(ds, bi2.id, filename='later.webp')
        again = fb.run_if_needed()
        assert again['ran_at'] == first['ran_at']
        assert _framing_of(later.id) is None
        assert fb.summary()['framings'] == 1


def test_boot_ran_the_pass_on_the_fixture_database(app):
    """create_app wires it: a freshly created database comes back with the flag
    already recorded at 0 repaired rows."""
    with app.app_context():
        assert fb.summary().get('version') == fb.BACKFILL_VERSION
        assert fb.summary().get('framings') == 0
