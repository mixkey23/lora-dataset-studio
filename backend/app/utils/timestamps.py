"""One honest UTC timestamp string, for the "when did this pass last run" fields.

`datetime.utcnow()` is deprecated (it returns a NAIVE datetime that merely claims
to be UTC) and each backfill had its own copy of
``datetime.utcnow().isoformat(timespec='seconds') + 'Z'``.

Deliberately narrow. This does NOT touch the naive `datetime.utcnow` used for
database columns and for the age/duration arithmetic in cloud_training: those
values are compared against datetimes SQLite already holds as naive, and swapping
in an aware object there raises ``can't subtract offset-naive and offset-aware
datetimes`` on data that is already on disk in every install. That migration is a
real one and does not belong in a deprecation cleanup.

The output is byte-for-byte what the backfills produced before (``Z`` suffix, no
offset, second precision), because it is persisted in report payloads users can
already have.
"""
from datetime import datetime, timezone

_FORMAT = '%Y-%m-%dT%H:%M:%SZ'


def utc_stamp() -> str:
    """Current UTC instant as ``2026-07-27T09:15:04Z``."""
    return datetime.now(timezone.utc).strftime(_FORMAT)
