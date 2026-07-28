"""Let a checkpoint keep EVERY preview it was ever given, instead of one.

``checkpoint_preview`` shipped with ``UniqueConstraint('record_id', 'step')``:
one row per checkpoint, and regenerating a preview overwrote the pointer. The
image itself stayed on disk, but nothing referenced it any more — a second look
at the same epoch silently erased the first. The LoRA Canvas shows a gallery
under each checkpoint, so the previews have to accumulate.

SQLite cannot DROP a constraint. The table has to be recreated:
``CREATE new / INSERT SELECT / DROP old / RENAME``.

⚠️ This is the one step of the whole feature that can lose data WITHOUT AN
ERROR. An ``INSERT SELECT`` that forgets a column, or a WHERE that quietly
matches less than the table holds, destroys rows and returns success. So the
guard here is not "does the table exist afterwards" — it is a ROW COUNT compared
before and after, inside the transaction, with the copy abandoned (and the
original left untouched) the instant the two disagree.

Written for databases nobody here will ever open:
  * idempotent — the constraint is detected through ``PRAGMA index_list``, so a
    database already migrated (or freshly created by ``db.create_all()``, which
    now builds the table without the constraint) is left alone;
  * column-driven — the copied columns are the INTERSECTION of what the old
    table really has and what the model declares, in the model's order, so an
    install that predates a column, or one carrying a column a later build
    dropped, both survive;
  * additive in spirit — nothing but the constraint changes; every row, every
    value, every id is preserved (ids matter: ``lora_test_image_id`` pointers
    are addressed by them);
  * fail-safe — any failure rolls back and leaves the ORIGINAL table in place.
    A canvas gallery that still overwrites its preview is a far better outcome
    than a lost history, and either way the app must still boot.
"""
from __future__ import annotations

import logging

from sqlalchemy import MetaData, text
from sqlalchemy.schema import CreateTable

from ..extensions import db

logger = logging.getLogger(__name__)

_TABLE = 'checkpoint_preview'
_TEMP = 'checkpoint_preview__migrating'


def _table_exists(name: str) -> bool:
    row = db.session.execute(
        text("SELECT 1 FROM sqlite_master WHERE type='table' AND name=:n"),
        {'n': name}).first()
    return row is not None


def _existing_columns(name: str) -> list[str]:
    return [r[1] for r in db.session.execute(text(f'PRAGMA table_info({name})'))]


def has_unique_constraint() -> bool:
    """True while the legacy one-preview-per-checkpoint constraint is still in
    force. Read from ``PRAGMA index_list``, not from the CREATE TABLE text: the
    constraint may have been declared named (``uq_checkpoint_preview``) or
    anonymously depending on the build that created the database, and both show
    up here as a UNIQUE index of origin 'u'."""
    if not _table_exists(_TABLE):
        return False
    for row in db.session.execute(text(f'PRAGMA index_list({_TABLE})')):
        # (seq, name, unique, origin, partial)
        name, unique, origin = row[1], row[2], row[3]
        if not unique:
            continue
        if origin not in ('u', 'pk', 'c'):
            continue
        if origin == 'pk':
            continue
        cols = [r[2] for r in db.session.execute(text(f'PRAGMA index_info({name})'))]
        if cols == ['record_id', 'step']:
            return True
    return False


def _create_temp_table() -> None:
    """Build the temp table from the MODEL, so its shape can never drift from
    what the ORM expects to read back."""
    from ..models import CheckpointPreview
    tbl = CheckpointPreview.__table__.to_metadata(MetaData(), name=_TEMP)
    # Indexes are separate DDL; they are (re)created after the rename under
    # their real names, so nothing collides with the live table meanwhile.
    tbl.indexes.clear()
    db.session.execute(text(str(CreateTable(tbl).compile(db.engine))))


def _recreate_indexes() -> None:
    from ..models import CheckpointPreview
    for idx in CheckpointPreview.__table__.indexes:
        cols = ', '.join(c.name for c in idx.columns)
        db.session.execute(text(
            f'CREATE INDEX IF NOT EXISTS {idx.name} ON {_TABLE} ({cols})'))
    # `index=True` columns get SQLAlchemy's default name; create_all already made
    # them on a fresh database, and IF NOT EXISTS makes this a no-op there.
    for col in ('record_id', 'dataset_id'):
        db.session.execute(text(
            f'CREATE INDEX IF NOT EXISTS ix_{_TABLE}_{col} ON {_TABLE} ({col})'))


def lift_unique_constraint() -> dict:
    """Recreate ``checkpoint_preview`` without its unique constraint, preserving
    every row. Returns {'migrated': bool, 'rows': int, 'reason': str|None}.
    Never raises: the caller is boot."""
    from ..models import CheckpointPreview
    try:
        if not has_unique_constraint():
            return {'migrated': False, 'rows': 0, 'reason': 'already-lifted'}

        # A leftover temp table from a crash mid-migration: drop it, the source
        # of truth is still the untouched original.
        if _table_exists(_TEMP):
            db.session.execute(text(f'DROP TABLE {_TEMP}'))
            db.session.commit()

        model_cols = [c.name for c in CheckpointPreview.__table__.columns]
        have = set(_existing_columns(_TABLE))
        copied = [c for c in model_cols if c in have]
        dropped = sorted(have - set(model_cols))
        if 'id' not in copied:
            # Without the primary key the lora_test_image_id pointers could not be
            # matched back. Refuse rather than produce a plausible-looking copy.
            return {'migrated': False, 'rows': 0, 'reason': 'no-id-column'}
        if dropped:
            logger.warning('checkpoint_preview migration: column(s) %s exist in the '
                           'database but not in the model — not copied', dropped)

        before = db.session.execute(text(f'SELECT COUNT(*) FROM {_TABLE}')).scalar() or 0

        _create_temp_table()
        cols_sql = ', '.join(copied)
        db.session.execute(text(
            f'INSERT INTO {_TEMP} ({cols_sql}) SELECT {cols_sql} FROM {_TABLE}'))

        after = db.session.execute(text(f'SELECT COUNT(*) FROM {_TEMP}')).scalar() or 0
        if after != before:
            # THE guard. Roll back before anything is dropped: the original table
            # is still there, intact, and the app boots on the old behaviour.
            db.session.rollback()
            logger.error('checkpoint_preview migration ABORTED: copied %d row(s) of '
                         '%d — the original table is untouched', after, before)
            return {'migrated': False, 'rows': before, 'reason': 'row-count-mismatch'}

        db.session.execute(text(f'DROP TABLE {_TABLE}'))
        db.session.execute(text(f'ALTER TABLE {_TEMP} RENAME TO {_TABLE}'))
        _recreate_indexes()
        db.session.commit()

        final = db.session.execute(text(f'SELECT COUNT(*) FROM {_TABLE}')).scalar() or 0
        if final != before:  # pragma: no cover - belt and braces after the commit
            logger.error('checkpoint_preview migration: %d row(s) after the rename, '
                         'expected %d', final, before)
        logger.info('checkpoint_preview: unique constraint lifted, %d preview(s) kept',
                    final)
        return {'migrated': True, 'rows': final, 'reason': None}
    except Exception:
        logger.exception('checkpoint_preview migration failed (skipped, the table is '
                         'left as it was — boot continues)')
        try:
            db.session.rollback()
        except Exception:
            pass
        try:
            if _table_exists(_TEMP):
                db.session.execute(text(f'DROP TABLE {_TEMP}'))
                db.session.commit()
        except Exception:
            db.session.rollback()
        return {'migrated': False, 'rows': 0, 'reason': 'failed'}


def run_if_needed() -> dict:
    """Boot entry point. No version flag is needed: ``has_unique_constraint`` IS
    the flag, read from the schema itself — the one state that cannot go stale."""
    return lift_unique_constraint()
