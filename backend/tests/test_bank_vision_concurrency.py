"""🗃️ Bank vision passes run their Ollama calls concurrently.

The watermark, framing and caption passes call Ollama once per image over banks
of tens of thousands of files, and most of each call is round-trip waiting
rather than GPU work — so the calls now overlap (see services/vision_pool.py).
Overlapping them is only safe if three things stay true, and this file pins all
three:

* **The database stays on one thread.** The SQLAlchemy session is not shareable,
  so worker threads may only do the network call. Asserted structurally: every
  commit is checked to come from the thread that owns the job, while the Ollama
  calls demonstrably come from others.
* **Stop still stops, and loses nothing.** A cancelled pass ends quickly, every
  answer already paid for is persisted, and every image never reached is left
  exactly as it was so a re-run finishes the job instead of redoing it.
* **The accounting survives.** A failing call is still one counted error, an
  empty answer is still one counted "not analysed", and the GPU-exclusive window
  is taken once for the pass rather than once per image.

No real Ollama here: the vision call is replaced by a fake that records which
thread called it and how many calls were in flight at once.
"""
import os
import threading
import time

import pytest
from PIL import Image


# --- fixtures ----------------------------------------------------------------
def _photo(size=64, value=120):
    return Image.new('RGB', (size, size), (value, value, value))


def _mkbank(client, tmp_path, count, name='CONC'):
    src = tmp_path / 'src'
    os.makedirs(src, exist_ok=True)
    for i in range(count):
        _photo(value=60 + i).save(str(src / f'{i:03d}.jpg'), 'JPEG', quality=90)
    r = client.post('/api/bank/create', json={'name': name, 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id'], src


class VisionSpy:
    """Stands in for describe_image_ollama, recording what a real one can't tell
    us: which thread ran the call, and how many ran at the same time."""

    def __init__(self, answer='{"watermark": false}', delay=0.01, on_call=None):
        self.answer = answer
        self.delay = delay
        self.on_call = on_call
        self.calls = 0
        self.peak = 0
        self._live = 0
        self.threads = set()
        self._lock = threading.Lock()

    def __call__(self, image_bytes, prompt, **kwargs):
        with self._lock:
            self.calls += 1
            index = self.calls
            self._live += 1
            self.peak = max(self.peak, self._live)
            self.threads.add(threading.current_thread().name)
        try:
            time.sleep(self.delay)
            if self.on_call:
                forced = self.on_call(index)
                if forced is not None:
                    return forced
            return self.answer(index) if callable(self.answer) else self.answer
        finally:
            with self._lock:
                self._live -= 1


@pytest.fixture()
def commit_threads(app, monkeypatch):
    """Every thread name that committed to the database during the test."""
    from app.extensions import db
    seen = []
    real = db.session.commit

    def spy():
        seen.append(threading.current_thread().name)
        return real()

    monkeypatch.setattr(db.session, 'commit', spy)
    return seen


def _patch_vision(monkeypatch, spy):
    """Both passes import describe_image_ollama from vision_ollama at call time,
    so patching the module attribute covers them."""
    import app.services.vision_ollama as vo
    monkeypatch.setattr(vo, 'describe_image_ollama', spy)
    monkeypatch.setattr(vo, 'unload_vision_model', lambda *a, **k: True)


def _allow_pass(monkeypatch):
    """Neutralise the launch preconditions (real Ollama probe, GPU busy check)."""
    import app.capabilities as caps
    import app.services.image_bank_service as svc
    monkeypatch.setattr(caps, 'probe_ollama_model', lambda *a, **k: {'ok': True})
    monkeypatch.setattr(svc, '_gpu_busy_reason', lambda *a, **k: None)


def _states(app, bank_id):
    from app.models import BankImage
    with app.app_context():
        return [(r.id, r.watermark_state) for r in
                BankImage.query.filter_by(bank_id=bank_id).order_by(BankImage.id.asc()).all()]


def _framings(app, bank_id):
    from app.models import BankImage
    with app.app_context():
        return [(r.id, r.framing) for r in
                BankImage.query.filter_by(bank_id=bank_id).order_by(BankImage.id.asc()).all()]


# --- the pass actually overlaps its calls -----------------------------------
def test_watermark_pass_overlaps_its_ollama_calls(client, app, tmp_path, monkeypatch):
    import app.services.image_bank_service as svc
    _allow_pass(monkeypatch)
    spy = VisionSpy()
    _patch_vision(monkeypatch, spy)
    bank_id, _src = _mkbank(client, tmp_path, 20)
    with app.app_context():
        job = svc.start_watermark(app, 'local', bank_id, rescan=True)
    assert job['error'] is None, job['error']
    assert spy.calls == 20
    assert spy.peak > 1, 'the pass ran strictly one call at a time'
    assert spy.peak <= 4, f'ran wider than the default concurrency: {spy.peak}'
    # Every image got its verdict, none left undecided.
    assert all(state == 'none' for _id, state in _states(app, bank_id))


def test_framing_pass_overlaps_its_ollama_calls(client, app, tmp_path, monkeypatch):
    import app.services.image_bank_service as svc
    _allow_pass(monkeypatch)
    spy = VisionSpy(answer='{"framing": "face", "label": "x"}')
    _patch_vision(monkeypatch, spy)
    bank_id, _src = _mkbank(client, tmp_path, 20)
    with app.app_context():
        job = svc.start_framing(app, 'local', bank_id, rescan=True)
    assert job['error'] is None, job['error']
    assert spy.calls == 20
    assert spy.peak > 1
    assert all(f for _id, f in _framings(app, bank_id))


def test_concurrency_of_one_restores_the_old_sequential_pass(client, app, tmp_path,
                                                             monkeypatch):
    """The escape hatch has to be real: set the knob to 1 and nothing overlaps."""
    import app.config as _cfg
    import app.services.image_bank_service as svc
    _allow_pass(monkeypatch)
    _cfg.save_config({'ollama': {'vision_concurrency': 1}})
    spy = VisionSpy()
    _patch_vision(monkeypatch, spy)
    bank_id, _src = _mkbank(client, tmp_path, 8)
    with app.app_context():
        job = svc.start_watermark(app, 'local', bank_id, rescan=True)
    assert job['error'] is None
    assert spy.calls == 8
    assert spy.peak == 1
    assert spy.threads == {threading.current_thread().name}


# --- the database stays on the owning thread --------------------------------
def test_no_worker_thread_ever_touches_the_database(client, app, tmp_path,
                                                    monkeypatch, commit_threads):
    """The structural invariant. Worker threads do the network call; the session
    is only ever used by the thread that owns the job."""
    import app.services.image_bank_service as svc
    _allow_pass(monkeypatch)
    spy = VisionSpy()
    _patch_vision(monkeypatch, spy)
    bank_id, _src = _mkbank(client, tmp_path, 20)
    owner = threading.current_thread().name
    with app.app_context():
        svc.start_watermark(app, 'local', bank_id, rescan=True)
    assert spy.threads - {owner}, 'the calls never left the owning thread — nothing proven'
    assert set(commit_threads) == {owner}, (
        f'a database commit ran on a worker thread: {set(commit_threads) - {owner}}')


# --- Stop -------------------------------------------------------------------
def test_stop_ends_the_pass_and_keeps_every_answer_paid_for(client, app, tmp_path,
                                                            monkeypatch):
    """THE cancellation test.

    A parallel pass must stop as promptly as a sequential one, and the images it
    did analyse must stay analysed — otherwise "Stop, then run it again" would
    redo work the user already waited for. Everything it never reached stays
    untouched (NULL), which is precisely what makes the re-run pick up where it
    left off rather than from zero.
    """
    from app.services import bank_jobs
    import app.services.image_bank_service as svc
    _allow_pass(monkeypatch)
    bank_id, _src = _mkbank(client, tmp_path, 120)

    def stop_after_six(index):
        if index >= 6:
            bank_jobs.cancel(bank_id)
        return None

    spy = VisionSpy(delay=0.02, on_call=stop_after_six)
    _patch_vision(monkeypatch, spy)
    started = time.perf_counter()
    with app.app_context():
        job = svc.start_watermark(app, 'local', bank_id, rescan=True)
    elapsed = time.perf_counter() - started

    assert job['error'] is None, job['error']
    assert job['cancelled'] is True
    # Stopped short — not "ran to the end and then noticed".
    assert spy.calls < 120, f'the cancel did not cut the pass short ({spy.calls} calls)'
    # And stopped PROMPTLY: at most the calls already in flight, never a long tail.
    assert spy.calls <= 6 + 4, f'too many calls after the stop: {spy.calls}'
    assert elapsed < 5, f'stop took {elapsed:.1f}s'

    decided = [s for _id, s in _states(app, bank_id) if s is not None]
    undecided = [s for _id, s in _states(app, bank_id) if s is None]
    # Nothing answered was thrown away: every completed call is persisted.
    assert len(decided) == spy.calls, (
        f'{spy.calls} images were analysed but only {len(decided)} were saved')
    # And everything never reached is untouched, so a re-run resumes.
    assert len(undecided) == 120 - spy.calls
    assert 'cancelled' in (job['detail'] or '')


def test_a_resumed_pass_only_pays_for_what_is_left(client, app, tmp_path, monkeypatch):
    """The other half of Stop: run again after cancelling and the pass picks up
    the untouched images only."""
    from app.services import bank_jobs
    import app.services.image_bank_service as svc
    _allow_pass(monkeypatch)
    bank_id, _src = _mkbank(client, tmp_path, 40)

    first = VisionSpy(delay=0.01, on_call=lambda i: bank_jobs.cancel(bank_id) and None
                      if i >= 5 else None)
    _patch_vision(monkeypatch, first)
    with app.app_context():
        svc.start_watermark(app, 'local', bank_id, rescan=True)
    done_first = len([s for _id, s in _states(app, bank_id) if s is not None])
    assert 0 < done_first < 40

    bank_jobs.reset()
    second = VisionSpy(delay=0)
    _patch_vision(monkeypatch, second)
    with app.app_context():
        job = svc.start_watermark(app, 'local', bank_id, rescan=False)
    assert job['error'] is None
    # Only the leftovers were paid for a second time.
    assert second.calls == 40 - done_first
    assert all(s is not None for _id, s in _states(app, bank_id))


# --- accounting survives -----------------------------------------------------
def test_an_empty_answer_is_still_counted_as_not_analysed(client, app, tmp_path,
                                                          monkeypatch):
    """Ollama returning nothing is NOT 'clean'. The row must stay unscanned and
    the report must say so — that honesty predates the concurrency and has to
    survive it."""
    import app.services.image_bank_service as svc
    _allow_pass(monkeypatch)
    spy = VisionSpy(answer='')
    _patch_vision(monkeypatch, spy)
    bank_id, _src = _mkbank(client, tmp_path, 10)
    with app.app_context():
        job = svc.start_watermark(app, 'local', bank_id, rescan=True)
    assert job['error'] is None
    assert '10 not analysed' in (job['detail'] or ''), job['detail']
    assert all(s is None for _id, s in _states(app, bank_id))


def test_one_failing_call_is_counted_and_the_pass_finishes(client, app, tmp_path,
                                                           monkeypatch):
    import app.services.image_bank_service as svc
    _allow_pass(monkeypatch)

    def blow_up_on_the_third(index):
        if index == 3:
            raise RuntimeError('ollama fell over')
        return None

    spy = VisionSpy(on_call=blow_up_on_the_third)
    _patch_vision(monkeypatch, spy)
    bank_id, _src = _mkbank(client, tmp_path, 12)
    with app.app_context():
        job = svc.start_watermark(app, 'local', bank_id, rescan=True)
    assert job['error'] is None, 'one bad call must not sink the pass'
    assert spy.calls == 12
    states = [s for _id, s in _states(app, bank_id)]
    assert states.count('error') == 1
    assert '1 unreadable' in (job['detail'] or ''), job['detail']


def test_the_gpu_window_is_taken_once_for_the_whole_pass(client, app, tmp_path,
                                                         monkeypatch):
    """The window unloads ComfyUI and blocks a training start; taking it per call
    would thrash the GPU. One pass, one window — whatever the concurrency."""
    import contextlib
    import app.gpu_window as gw
    import app.services.image_bank_service as svc
    _allow_pass(monkeypatch)
    entered = []

    @contextlib.contextmanager
    def counting_window(*a, **k):
        entered.append(1)
        yield

    monkeypatch.setattr(gw, 'gpu_exclusive_vision_window', counting_window)
    _patch_vision(monkeypatch, VisionSpy())
    bank_id, _src = _mkbank(client, tmp_path, 20)
    with app.app_context():
        svc.start_watermark(app, 'local', bank_id, rescan=True)
    assert entered == [1], f'the GPU window was taken {len(entered)} times'
