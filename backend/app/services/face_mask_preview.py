"""Background runner + result store for the concept face-mask PREVIEW.

Why this is not a plain synchronous POST any more
-------------------------------------------------
The preview used to be one blocking request. That made two things impossible at
once, and they are the same bug seen from two sides:

* nothing could be shown WHILE it ran — the button said "Looking for faces…" for
  the whole InsightFace pass (tens of seconds before image 1 even starts) and a
  crash looked identical to a slow run;
* nothing could be REJOINED. The browser tab that issued the request is the only
  holder of its handle, so leaving the page threw the pass away — and coming back
  offered to start a second full pass over the same images, possibly next to the
  first one still burning CPU.

Streaming the response would have fixed only the first. A server-side job fixes
both, so that is what this is.

Why not the two job registries already here
-------------------------------------------
* `dataset_activity` is the dataset-wide BUSY flag: entering it blocks generation
  on that dataset and lights the global activity banner. A read-only preview that
  writes nothing must not seize the dataset — see the gotcha noted in the wave
  report; a unified registry is a refactor, not this fix.
* `bank_jobs` is keyed by bank id in a single module dict; reusing it for dataset
  ids would collide two id namespaces in one keyspace.

So: the same design CONTRACT as bank_jobs (in-memory only, thread-safe, one live
job per key, inline under TESTING), keyed by dataset.

Staleness
---------
A stored preview is only true of the exact kept set it was computed from, so it
is stored with a fingerprint of that set (id + filename + size + mtime of every
kept image). Reads recompute it and flag `stale`. A preview shown as fresh after
the images changed would be worse than no preview at all — the user would trust
boxes drawn from photos that are no longer in the run.

In-memory ONLY: a restart empties this, and the worst case is one recomputation.
"""
from __future__ import annotations
import hashlib
import threading
import time

_lock = threading.Lock()
_state: dict = {}          # dataset_id -> {'job': job|None, 'result': result|None}

# A running job untouched for this long has lost its thread (the subprocess has
# its own, much shorter, watchdog) — treat it as dead so the UI can start again
# instead of waiting on a ghost forever.
_STALE_TTL = 60 * 60


def fingerprint(entries) -> str:
    """Fingerprint of the kept set. ``entries`` = iterable of
    (image_id, filename, size, mtime_ns). Sorted before hashing so row order
    never changes the answer."""
    parts = sorted(f'{i}|{n}|{s}|{m}' for (i, n, s, m) in entries)
    return hashlib.sha1('\n'.join(parts).encode('utf-8')).hexdigest()[:16]


def _slot(dataset_id) -> dict:
    return _state.setdefault(int(dataset_id), {'job': None, 'result': None})


def _live(job) -> bool:
    return bool(job) and not job['finished'] and time.time() - job['_touched'] < _STALE_TTL


def public(job) -> dict | None:
    """The job as the UI sees it — no private bookkeeping."""
    if not job:
        return None
    return {k: job[k] for k in ('phase', 'done', 'total', 'error', 'finished')}


def get(dataset_id):
    """The live job for this dataset, or None."""
    with _lock:
        job = _slot(dataset_id)['job']
        return job if _live(job) else None


def snapshot(dataset_id, current_fp=None) -> dict:
    """What the panel needs on mount and on every poll: the running job (if any)
    and the last result, flagged `stale` when the kept set moved under it."""
    with _lock:
        slot = _slot(dataset_id)
        job, result = slot['job'], slot['result']
        out = {'job': public(job) if _live(job) or (job and job['finished']) else None}
        if result:
            out['result'] = {k: v for k, v in result.items() if k != 'fingerprint'}
            out['result']['stale'] = bool(
                current_fp is not None and result.get('fingerprint') != current_fp)
        else:
            out['result'] = None
        return out


def start(app, dataset_id, fn, total=0, fp=None):
    """Start ``fn(job)`` for this dataset, or JOIN the one already running.

    Returns ``(job, started)``. ``started`` False means a pass was already in
    flight and this call did nothing — the second click, or the return to the
    page, costs zero InsightFace passes.
    """
    now = time.time()
    with _lock:
        slot = _slot(dataset_id)
        if _live(slot['job']):
            return slot['job'], False
        job = {'phase': 'starting', 'done': 0, 'total': int(total or 0),
               'error': None, 'finished': False, 'started_at': now,
               '_touched': now, '_fp': fp}
        slot['job'] = job

    def _run():
        try:
            with app.app_context():
                fn(job)
        except Exception as e:  # noqa: BLE001 — a background crash must surface, not vanish
            with _lock:
                job['error'] = f'{type(e).__name__}: {e}'
        finally:
            with _lock:
                job['finished'] = True
                job['_touched'] = time.time()

    # Under TESTING the job runs INLINE, exactly like bank_jobs: the suite uses a
    # per-connection sqlite:///:memory: DB, so a real worker thread would open a
    # fresh, EMPTY database.
    if app.config.get('TESTING'):
        _run()
    else:
        threading.Thread(target=_run, daemon=True,
                         name=f'facemask-preview-{dataset_id}').start()
    return job, True


def progress(job, record):
    """Apply one {'phase', 'done'?, 'total'?} record from the stderr reader."""
    if not job or not record:
        return
    with _lock:
        for key in ('phase', 'done', 'total'):
            if record.get(key) is not None:
                job[key] = record[key]
        job['_touched'] = time.time()


def set_result(dataset_id, result, fp=None):
    """Publish the finished preview. Kept until superseded — a preview the user
    walked away from is exactly the one they come back to look at."""
    with _lock:
        payload = dict(result or {})
        payload['fingerprint'] = fp
        payload['at'] = time.time()
        _slot(dataset_id)['result'] = payload


def fail(job, message):
    with _lock:
        job['error'] = str(message)
        job['_touched'] = time.time()


def reset(dataset_id=None):
    """Test hook — drop one dataset's state, or all of it."""
    with _lock:
        if dataset_id is None:
            _state.clear()
        else:
            _state.pop(int(dataset_id), None)
