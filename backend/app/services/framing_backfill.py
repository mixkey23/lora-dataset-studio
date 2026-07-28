"""Retrofit the framing of dataset images promoted from an image bank BEFORE the
promotion carried it.

The dataset Composition tally counts only rows that HAVE a framing
(``dataset_payload``), and the bank promotion used to drop the framing its own
classify pass had already written. Those rows landed with ``framing = NULL`` and
sat invisible in Composition. Fixing the promotion path only helps the NEXT
promotion — this pass repairs what is already on disk.

The evidence is the row itself: a promoted row records the ``bank_image_id`` it
came from, and that bank image carries the framing verdict. Copying it back is a
recovery, not a guess.

Conservative by design — it must behave on databases nobody here will ever see:
  * ONLY rows whose framing is still NULL are touched. A value already there is
    left alone, even a surprising one: the user may have classified by hand.
  * ONLY the four composition buckets are copied. A bank framing that is NULL,
    ``'unknown'`` or anything else leaves the row NULL, so the dataset classifier
    can still pick it up later. We do not invent a bucket.
  * A ``bank_image_id`` pointing at a bank image that no longer exists (bank
    deleted, rescanned) is skipped silently. That is a normal state, not an error.
  * A database whose schema predates the bank (no ``bank_image`` table, or no
    ``framing`` column on it) skips the pass and says so in the log.
  * ONE set-based UPDATE, never a Python row loop, so a large library costs a
    single short write transaction — boot must not hold the SQLite write lock.
  * Any failure is swallowed: a backfill must never keep the app from starting.

Idempotent: guarded by a persisted version flag like the lineage backfill, and
even without the flag a second run matches nothing (every row it could fill is
no longer NULL).
"""
from __future__ import annotations

import json
import logging
from ..utils.timestamps import utc_stamp

from sqlalchemy import inspect, text

from ..extensions import db
from ..models import SystemState

logger = logging.getLogger(__name__)

# Bump when the recovery RULE improves, so a smarter pass re-runs once on an
# already-processed database (it still only fills framings that are still NULL).
BACKFILL_VERSION = 1
_STATE_KEY = 'framing_backfill'

# Only the four composition buckets are whitelisted below. Anything else (NULL,
# 'unknown', a verdict a future build might add) must leave the row NULL rather
# than be copied blindly.
# One statement. The correlated subquery form works on every SQLite that ships
# with a supported Python (UPDATE ... FROM would need 3.33+ and buys nothing
# here). `bank_image_id` is indexed on the dataset side and the lookup lands on
# bank_image's primary key; the outer WHERE scans face_dataset_image once, which
# is a table counted in thousands, not millions. `framing` is deliberately NOT
# indexed and this pass does not add one — a single boot-time scan does not
# justify an index every write would then have to maintain.
_UPDATE_SQL = text(
    """
    UPDATE face_dataset_image
       SET framing = (SELECT b.framing FROM bank_image b
                       WHERE b.id = face_dataset_image.bank_image_id)
     WHERE face_dataset_image.framing IS NULL
       AND face_dataset_image.bank_image_id IS NOT NULL
       AND EXISTS (SELECT 1 FROM bank_image b
                    WHERE b.id = face_dataset_image.bank_image_id
                      AND b.framing IN ('face', 'bust', 'body', 'back'))
    """
)

_COUNT_SQL = text(
    """
    SELECT COUNT(*) FROM face_dataset_image
     WHERE face_dataset_image.framing IS NULL
       AND face_dataset_image.bank_image_id IS NOT NULL
       AND EXISTS (SELECT 1 FROM bank_image b
                    WHERE b.id = face_dataset_image.bank_image_id
                      AND b.framing IN ('face', 'bust', 'body', 'back'))
    """
)


def _load_state() -> dict:
    row = db.session.get(SystemState, _STATE_KEY)
    if row is None or not row.value:
        return {}
    try:
        return json.loads(row.value)
    except (TypeError, ValueError):
        return {}


def _save_state(state: dict) -> None:
    row = db.session.get(SystemState, _STATE_KEY)
    if row is None:
        row = SystemState(key=_STATE_KEY)
        db.session.add(row)
    row.value = json.dumps(state)
    db.session.commit()


def _schema_supports_backfill() -> bool:
    """True when both sides of the join exist. An install that never used the
    image bank (or predates it) simply has nothing to recover from."""
    insp = inspect(db.engine)
    names = set(insp.get_table_names())
    if 'bank_image' not in names or 'face_dataset_image' not in names:
        return False
    bank_cols = {c['name'] for c in insp.get_columns('bank_image')}
    ds_cols = {c['name'] for c in insp.get_columns('face_dataset_image')}
    return 'framing' in bank_cols and {'framing', 'bank_image_id'} <= ds_cols


def backfill_promoted_framings() -> int:
    """Copy the bank's framing onto every promoted row that is still missing one.
    Returns how many rows were repaired. Pure DB work, one statement, never raises."""
    try:
        if not _schema_supports_backfill():
            logger.info('framing backfill: schema predates the image bank (skipped)')
            return 0
        # Read first: on the overwhelming majority of installs there is nothing to
        # do, and this way boot never opens a write transaction at all.
        if not db.session.execute(_COUNT_SQL).scalar():
            return 0
        filled = db.session.execute(_UPDATE_SQL).rowcount or 0
        db.session.commit()
        return filled
    except Exception:
        logger.exception('framing backfill: failed (skipped, nothing changed)')
        try:
            db.session.rollback()
        except Exception:
            pass
        return 0


def run_if_needed() -> dict:
    """Boot entry point: repair once, guarded by a persisted version flag so a
    normal restart never re-scans. Never blocks startup."""
    try:
        state = _load_state()
        if int(state.get('version') or 0) >= BACKFILL_VERSION:
            return state                       # already done at this rule version
        filled = backfill_promoted_framings()
        summary_ = {'version': BACKFILL_VERSION, 'framings': filled,
                    'ran_at': utc_stamp()}
        _save_state(summary_)
        if filled:
            logger.info('framing backfill: recovered %d promoted framing(s)', filled)
        return summary_
    except Exception:
        logger.exception('framing backfill failed (non-fatal, boot continues)')
        try:
            db.session.rollback()
        except Exception:
            pass
        return {}


def summary() -> dict:
    """The last recorded backfill result, for the diagnostic payload. {} until the
    first pass has run."""
    try:
        return _load_state()
    except Exception:
        return {}
