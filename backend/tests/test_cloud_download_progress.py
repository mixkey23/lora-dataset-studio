"""The two things a run card could not tell you about a pod that is costing money.

1. "Is it downloading or is it frozen?" — while a run pulls its base weights
   (26 GB) the card showed a single fixed sentence for as long as it took. The
   pod's log carried a byte counter the whole time and it was thrown away.
2. "Has it been quiet long enough to kill?" — the silence counter was measured
   from updated_at, which is re-stamped when the app re-adopts a run after a
   restart AND on every monitor poll. On a machine that restarts often the
   45-minute freeze watchdog could never be reached.

Everything here is offline: no pod, no network, no thread.
"""
from datetime import datetime, timedelta

import pytest

# Verbatim tail of a real pod log (run 121, 2026-07-28): the download had been
# frozen at 1.95G for 20 minutes while the card said 'fetching transformer
# weights'. tqdm keeps re-printing the bar with a bumped ELAPSED even though
# the byte counter has not moved — which is why the freeze fingerprint below
# must ignore the text and key on the counters.
RUN121_TAIL = (
    'raw.safetensors:   0%|          | 0.00/26.3G [00:00<?, ?B/s]\n'
    'raw.safetensors:   1%|          | 293M/26.3G [00:50<35:45, 12.1MB/s]\n'
    'raw.safetensors:   4%|▍         | 1.17G/26.3G [15:10<6:25:15, 1.09MB/s]\n'
    'raw.safetensors:   7%|▋         | 1.95G/26.3G [15:11<2:37:06, 2.58MB/s]\n'
    'raw.safetensors:   7%|▋         | 1.95G/26.3G [15:30<2:37:06, 2.58MB/s]\n'
)

STEP_TAIL = ('lora_t:  50%|█████     | 1750/3500 '
             '[1:06:27<1:11:05,  2.44s/it, lr: 1.0e-04 loss: 1.913e-01]')


@pytest.fixture()
def ct(app, monkeypatch):
    from app.services import cloud_training
    monkeypatch.setattr(cloud_training, '_start_monitor', lambda *a, **k: None)
    yield cloud_training


def _mkrun(ct, tmp_path, log=None, **kw):
    staging = tmp_path / f"run_{kw.get('job_name', 'j1')}"
    staging.mkdir(parents=True, exist_ok=True)
    if log is not None:
        (staging / 'training.log').write_text(log, encoding='utf-8')
    fields = dict(dataset_id=1, status='training', vast_instance_id='90001',
                  vast_label='lds-1', job_name='j1', price_per_hour=0.47,
                  staging_dir=str(staging))
    fields.update(kw)
    run = ct.CloudTrainingRun(**fields)
    ct.db.session.add(run)
    ct.db.session.commit()
    return run


# ------------------------------------------------- 1. the byte counter --

def test_parse_download_progress_reads_the_real_bar():
    from app.services.lora_training import parse_download_progress
    d = parse_download_progress(RUN121_TAIL)
    assert d['label'] == 'raw.safetensors'
    assert (d['done'], d['total']) == ('1.95G', '26.3G')
    assert d['percent'] == 7
    assert d['speed'] == '2.58MB/s'
    assert d['eta'] == '2:37:06'
    assert d['elapsed'] == '15:30'          # the LAST bar, not the first


def test_parse_download_progress_degrades_on_anything_else():
    """A third-party format must never be able to break the card. Anything that
    is not a complete byte bar yields None, and the caller keeps showing the
    phase sentence it always showed."""
    from app.services.lora_training import parse_download_progress
    for text in ('', 'Running 1 job\n{"type": "diffusion_trainer"}',
                 STEP_TAIL,                                  # a STEP bar
                 'raw.safetensors: 7% 1.95G/26.3G',          # no tqdm bar
                 'raw.safetensors:   7%|x| 1.95G/26.3G [15:30<2:37:06]',  # no rate
                 'weights: 42%|x| 12/30 [00:10<00:12, 1.4it/s]'):         # items
        assert parse_download_progress(text) is None, text


def test_a_byte_bar_is_no_longer_misread_as_a_training_step():
    """'0.00/26.3G' used to match the step regex as step 0 of 26 — a
    plausible-looking number that was pure noise."""
    from app.services.lora_training import _parse_training_log
    assert _parse_training_log(RUN121_TAIL)['step'] is None
    # ...and a real step bar still parses.
    assert _parse_training_log(STEP_TAIL)['step'] == 1750


def test_the_run_card_payload_carries_the_bytes(ct, app, tmp_path):
    """RED before this wave: the payload had no download field at all, so the
    card could only show 'fetching transformer weights'."""
    with app.app_context():
        run = _mkrun(ct, tmp_path, log=RUN121_TAIL)
        payload = ct._run_payload(run)
        assert payload['download']['done'] == '1.95G'
        assert payload['download']['total'] == '26.3G'
        assert payload['download']['speed'] == '2.58MB/s'


def test_a_finished_run_and_an_unparsable_log_carry_no_bytes(ct, app, tmp_path):
    with app.app_context():
        done = _mkrun(ct, tmp_path, log=RUN121_TAIL, status='done', job_name='j2')
        assert ct._run_payload(done)['download'] is None
        noisy = _mkrun(ct, tmp_path, log='nothing to see here', job_name='j3')
        assert ct._run_payload(noisy)['download'] is None
        nolog = _mkrun(ct, tmp_path, job_name='j4')
        assert ct._run_payload(nolog)['download'] is None


# --------------------------------------- 2. a silence that survives boot --

def test_silence_is_measured_from_progress_not_from_updated_at(ct, app, tmp_path):
    """The whole point: a monitor that keeps writing the SAME state every 10 s
    keeps updated_at fresh forever. Silence must be judged on the pod, not on
    the writer."""
    with app.app_context():
        run = _mkrun(ct, tmp_path, log=RUN121_TAIL,
                     updated_at=datetime.utcnow() - timedelta(hours=1))
        started = datetime.utcnow() - timedelta(hours=1)
        ct.note_progress(run, started)                 # first observation
        # ... an hour of the monitor re-writing the same sentence ...
        ct._set(run, phase_detail='running:   - fetching transformer weights')
        assert ct._idle_seconds(run) < 5               # updated_at looks alive
        ct.note_progress(run)
        assert ct._silent_seconds(run) > 55 * 60       # the pod is not


def test_a_restart_does_not_reset_the_silence_counter(ct, app, tmp_path):
    """Three restarts in one hour kept a dead pod under the 45-minute threshold.
    Re-adoption re-stamps updated_at and rewrites phase_detail; neither may
    count as progress."""
    with app.app_context():
        run = _mkrun(ct, tmp_path, log=RUN121_TAIL,
                     updated_at=datetime.utcnow() - timedelta(minutes=50))
        ct.note_progress(run, datetime.utcnow() - timedelta(minutes=50))

        # --- simulated restart: the app re-adopts the run ------------------
        ct._set(run, phase_detail='Resuming — reattaching to running job')
        ct.db.session.expire_all()
        readopted = ct.CloudTrainingRun.query.get(run.id)
        ct.note_progress(readopted)

        assert ct._silent_seconds(readopted) > 45 * 60
        assert ct._run_payload(readopted)['idle_seconds'] > 45 * 60


def test_real_progress_restarts_the_counter(ct, app, tmp_path):
    """The mirror image — never kill a run that is moving. Only the counters
    count: tqdm re-printing the same bar with a bumped elapsed does not."""
    with app.app_context():
        run = _mkrun(ct, tmp_path, log=RUN121_TAIL,
                     updated_at=datetime.utcnow() - timedelta(minutes=50))
        ct.note_progress(run, datetime.utcnow() - timedelta(minutes=50))

        same_text_new_elapsed = RUN121_TAIL + (
            'raw.safetensors:   7%|x| 1.95G/26.3G [40:02<2:37:06, 2.58MB/s]\n')
        (tmp_path / 'run_j1' / 'training.log').write_text(
            same_text_new_elapsed, encoding='utf-8')
        ct.note_progress(run)
        assert ct._silent_seconds(run) > 45 * 60       # still frozen

        moved = RUN121_TAIL + (
            'raw.safetensors:  12%|x| 3.20G/26.3G [40:02<2:10:00, 2.58MB/s]\n')
        (tmp_path / 'run_j1' / 'training.log').write_text(moved, encoding='utf-8')
        ct.note_progress(run)
        assert ct._silent_seconds(run) < 60


def test_the_freeze_watchdog_now_fires_across_a_restart(ct, app, tmp_path, monkeypatch):
    """End to end: the supervisor terminates a pod whose progress clock is old,
    even though updated_at was refreshed a second ago by a re-adoption."""
    destroyed = []
    monkeypatch.setattr(ct.vast_client, 'destroy_instance',
                        lambda iid: destroyed.append(str(iid)) or True)
    with app.app_context():
        run = _mkrun(ct, tmp_path, log=RUN121_TAIL,
                     updated_at=datetime.utcnow() - timedelta(minutes=90))
        ct.note_progress(run, datetime.utcnow() - timedelta(minutes=90))
        ct._set(run, phase_detail='Resuming — reattaching to running job')

        acted = ct.supervise_active_runs()

        assert destroyed == ['90001']
        assert [a['reason'] for a in acted] == ['freeze']
        ct.db.session.refresh(run)
        assert run.status == 'stopped'
        assert 'no progress' in (run.phase_detail or '').lower()


def test_a_run_with_no_recorded_clock_falls_back_to_updated_at(ct, app, tmp_path):
    """Upgrade path: runs that were already active when this shipped have no
    stored clock. They must be judged exactly as before, never as instantly
    frozen and never as eternally fresh."""
    with app.app_context():
        fresh = _mkrun(ct, tmp_path, updated_at=datetime.utcnow())
        assert ct._silent_seconds(fresh) < 5
        stale = _mkrun(ct, tmp_path, job_name='j5',
                       updated_at=datetime.utcnow() - timedelta(hours=3))
        assert ct._silent_seconds(stale) > 2 * 3600


# ------------------------------------------------- anti-regression: stop --

def test_monitor_responsiveness_still_reads_updated_at(ct, app, tmp_path):
    """_monitor_is_responsive answers a DIFFERENT question — 'can this thread
    be trusted to carry out a stop?' — and must keep using the monitor's own
    heartbeat. A frozen pod whose monitor is alive and writing is still
    stoppable gracefully."""
    import threading

    class _Alive(threading.Thread):
        def is_alive(self):
            return True

    with app.app_context():
        run = _mkrun(ct, tmp_path, log=RUN121_TAIL, updated_at=datetime.utcnow())
        ct.note_progress(run, datetime.utcnow() - timedelta(hours=2))
        ct._monitor_threads[int(run.id)] = _Alive()
        try:
            # silent for two hours, yet the monitor is writing -> graceful path
            assert ct._silent_seconds(run) > 2 * 3600
            assert ct._monitor_is_responsive(run) is True
            run.updated_at = datetime.utcnow() - timedelta(minutes=10)
            ct.db.session.commit()
            assert ct._monitor_is_responsive(run) is False
        finally:
            ct._monitor_threads.pop(int(run.id), None)
