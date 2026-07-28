"""The backfill passes' "ran_at" stamp.

`datetime.utcnow()` is deprecated in 3.12+ and every backfill carried its own
copy of the same expression. The replacement has to be byte-for-byte compatible:
these strings are persisted in report payloads users already have on disk.
"""
import re
import warnings

from app.utils.timestamps import utc_stamp

_ISO_Z = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$')


def test_the_stamp_keeps_the_exact_shape_that_is_already_persisted():
    assert _ISO_Z.match(utc_stamp()), utc_stamp()


def test_no_backfill_still_calls_the_deprecated_utcnow():
    import pathlib
    services = pathlib.Path(__file__).resolve().parents[1] / 'app' / 'services'
    for name in ('lineage_backfill', 'framing_backfill', 'checkpoint_link_backfill'):
        src = (services / f'{name}.py').read_text(encoding='utf-8')
        assert 'utcnow' not in src, f'{name} still uses the deprecated utcnow()'
        assert 'utc_stamp' in src


def test_the_stamp_raises_no_deprecation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        utc_stamp()
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]
