import io
import json
import os

from PIL import Image


def _png(color=(255, 0, 0)):
    buf = io.BytesIO()
    Image.new('RGB', (64, 64), color).save(buf, 'PNG')
    return buf.getvalue()


def _scorer(stdout, returncode=0, stderr=''):
    """Stand-in for face_similarity._run_scorer, whose contract is
    (stdout, stderr_lines, returncode, timed_out). That helper is the seam now
    that the service drives the child with Popen instead of subprocess.run — it
    has to read the scorer's "[face] i/N" progress WHILE it runs, which run()
    only hands back once the child has exited."""
    lines = [ln for ln in (stderr or '').splitlines() if ln.strip()]
    return (stdout, lines, returncode, False)


def _dataset_with_ref_and_kept_image(svc, LOCAL_USER):
    """Real FaceDataset with a reference photo + one kept FaceDatasetImage, both
    backed by real files on disk (score_dataset_faces checks os.path.isfile)."""
    from app.models import FaceDatasetImage
    ds = svc.create_dataset(LOCAL_USER, 'FaceTest', 'facetest')
    d = svc._dataset_dir(ds.id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'ref.webp'), 'wb') as fh:
        fh.write(_png())
    ds.ref_filename = 'ref.webp'
    svc.db.session.commit()
    fn = 'kept.webp'
    with open(os.path.join(d, fn), 'wb') as fh:
        fh.write(_png((0, 255, 0)))
    img = FaceDatasetImage(dataset_id=ds.id, source='import', status='keep', filename=fn, framing='face')
    svc.db.session.add(img)
    svc.db.session.commit()
    return ds, img


# --- face_similarity.score_dataset_faces / analyze_faces --------------------

def test_analyze_faces_maps_state_and_sim_onto_rows(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import face_similarity as fsim
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage

    monkeypatch.setattr(fsim, 'is_available', lambda: True)
    with app.app_context():
        ds, img = _dataset_with_ref_and_kept_image(svc, LOCAL_USER)
        path = svc._img_path(img)
        contract = {"ref_ok": True, "results": {path: {"state": "scorable", "sim": 0.62,
                                                        "det": 0.9, "bbox_frac": 0.2, "yaw": 5.0}}}
        # Noise lines before the JSON line -- the parser must pick the LAST `{`-line.
        noisy_stdout = "some progress line\n[face] loading model\n" + json.dumps(contract)
        monkeypatch.setattr('app.services.face_similarity._run_scorer',
                            lambda *a, **k: _scorer(noisy_stdout))
        counts, _err = svc.analyze_faces(LOCAL_USER, ds.id)
        refreshed = svc.db.session.get(FaceDatasetImage, img.id)
        assert refreshed.face_state == 'scorable'
        assert refreshed.face_score == 0.62
        assert counts == {'scorable': 1}


def test_analyze_faces_ref_not_ok_returns_empty_and_no_row_change(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import face_similarity as fsim
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage

    monkeypatch.setattr(fsim, 'is_available', lambda: True)
    with app.app_context():
        ds, img = _dataset_with_ref_and_kept_image(svc, LOCAL_USER)
        contract = {"ref_ok": False, "results": {}, "error": "ref unusable"}
        monkeypatch.setattr('app.services.face_similarity._run_scorer',
                            lambda *a, **k: _scorer(json.dumps(contract)))
        counts, _err = svc.analyze_faces(LOCAL_USER, ds.id)
        refreshed = svc.db.session.get(FaceDatasetImage, img.id)
        assert counts == {}
        assert refreshed.face_state is None
        assert refreshed.face_score is None


def test_score_dataset_faces_unavailable_returns_empty_without_subprocess(app, monkeypatch):
    """is_available() False (capability probe) -> score_dataset_faces returns {}
    WITHOUT ever invoking subprocess.run (the never-raise/never-shell-out contract)."""
    from app.services import face_similarity as fsim
    from app.capabilities import probe_face_scoring

    monkeypatch.setattr(fsim, 'is_available', lambda: False)

    def _boom(*a, **k):
        raise AssertionError('no subprocess must be spawned when unavailable')

    monkeypatch.setattr('app.services.face_similarity._run_scorer', _boom)
    with app.app_context():
        # inputs missing on disk -> ({}, None) before the availability check
        assert fsim.score_dataset_faces('/does/not/matter', ['/also/not']) == ({}, None)


def test_is_available_delegates_to_capability_probe(app, monkeypatch):
    from app.services import face_similarity as fsim
    import app.capabilities as caps_mod

    monkeypatch.setattr(caps_mod, 'probe_face_scoring', lambda: {'ok': True, 'detail': 'x'})
    with app.app_context():
        assert fsim.is_available() is True
    monkeypatch.setattr(caps_mod, 'probe_face_scoring', lambda: {'ok': False, 'detail': 'x'})
    with app.app_context():
        assert fsim.is_available() is False


def test_analyze_faces_returns_cleanly_when_scorer_unavailable(app, monkeypatch):
    """analyze_faces must not raise or crash when the scorer is unavailable --
    score_dataset_faces short-circuits to {}, so no row gets a state/score and
    counts comes back empty."""
    from app.services import face_dataset_service as svc
    from app.services import face_similarity as fsim
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage

    monkeypatch.setattr(fsim, 'is_available', lambda: False)
    with app.app_context():
        ds, img = _dataset_with_ref_and_kept_image(svc, LOCAL_USER)
        counts, err = svc.analyze_faces(LOCAL_USER, ds.id)
        refreshed = svc.db.session.get(FaceDatasetImage, img.id)
        assert counts == {}
        assert err and err['kind'] == 'unavailable'
        assert refreshed.face_state is None


def test_score_dataset_faces_stdin_payload_includes_models_root(app, monkeypatch):
    from app.services import face_similarity as fsim
    from app.config import LOCAL_USER, save_config

    monkeypatch.setattr(fsim, 'is_available', lambda: True)
    captured = {}

    def _fake_run(*args, **kwargs):
        captured['input'] = args[1]
        return _scorer(json.dumps({"ref_ok": True, "results": {}}))

    monkeypatch.setattr('app.services.face_similarity._run_scorer', _fake_run)
    with app.app_context():
        save_config({'face_scoring': {'models_root': 'C:/models/insightface'}})
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ref = os.path.join(d, 'ref.png')
            img_path = os.path.join(d, 'img.png')
            with open(ref, 'wb') as fh:
                fh.write(_png())
            with open(img_path, 'wb') as fh:
                fh.write(_png())
            fsim.score_dataset_faces(ref, [img_path])
    payload = json.loads(captured['input'])
    assert payload['models_root'] == 'C:/models/insightface'


def test_score_dataset_faces_stdin_payload_models_root_none_when_unconfigured(app, monkeypatch):
    from app.services import face_similarity as fsim

    monkeypatch.setattr(fsim, 'is_available', lambda: True)
    captured = {}

    def _fake_run(*args, **kwargs):
        captured['input'] = args[1]
        return _scorer(json.dumps({"ref_ok": True, "results": {}}))

    monkeypatch.setattr('app.services.face_similarity._run_scorer', _fake_run)
    with app.app_context():
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ref = os.path.join(d, 'ref.png')
            img_path = os.path.join(d, 'img.png')
            with open(ref, 'wb') as fh:
                fh.write(_png())
            with open(img_path, 'wb') as fh:
                fh.write(_png())
            fsim.score_dataset_faces(ref, [img_path])
    payload = json.loads(captured['input'])
    assert payload['models_root'] is None


def test_score_dataset_faces_native_crash_returns_empty_not_exception(app, monkeypatch):
    """A crashed/killed subprocess (OSError) or a timeout must degrade to {},
    never raise up through analyze_faces to the route."""
    from app.services import face_dataset_service as svc
    from app.services import face_similarity as fsim
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage

    monkeypatch.setattr(fsim, 'is_available', lambda: True)

    def _crash(*a, **k):
        raise OSError('the interpreter died')

    monkeypatch.setattr('app.services.face_similarity._run_scorer', _crash)
    with app.app_context():
        ds, img = _dataset_with_ref_and_kept_image(svc, LOCAL_USER)
        counts, err = svc.analyze_faces(LOCAL_USER, ds.id)  # must not raise
        refreshed = svc.db.session.get(FaceDatasetImage, img.id)
        assert counts == {}
        assert err and err['kind'] == 'failed' and 'interpreter died' in err['detail']
        assert refreshed.face_state is None


# --- live progress ("[face] i/N" was printed and nobody read it) -------------

def test_score_dataset_faces_streams_progress_while_the_child_runs(app, monkeypatch, tmp_path):
    """The scorer prints "[face] i/N" for every image. Until now nobody read it,
    so the pass sat at 0/N for minutes and then jumped to N/N — indistinguishable
    from a hang. The bank's embedding pass has read "[embed] i/N" all along;
    this is the same mechanism."""
    import sys
    import tempfile
    from app.services import face_similarity as fsim

    script = tmp_path / 'stub_scorer.py'
    script.write_text(
        'import json, sys\n'
        'json.loads(sys.stdin.read())\n'
        'for i in (1, 2, 3):\n'
        '    print("[face] %d/3 scorable sim=0.5" % i, file=sys.stderr, flush=True)\n'
        'print(json.dumps({"ref_ok": True, "results": {}}))\n',
        encoding='utf-8')
    monkeypatch.setattr(fsim, 'is_available', lambda: True)
    monkeypatch.setattr(fsim, '_SCRIPT', str(script))
    monkeypatch.setattr(fsim, '_scoring_python', lambda: sys.executable)
    seen = []
    with app.app_context():
        with tempfile.TemporaryDirectory() as d:
            ref = os.path.join(d, 'ref.png')
            img_path = os.path.join(d, 'img.png')
            for f in (ref, img_path):
                with open(f, 'wb') as fh:
                    fh.write(_png())
            results, err = fsim.score_dataset_faces(
                ref, [img_path], on_progress=lambda done, total: seen.append((done, total)))
    assert err is None and results == {}
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_a_progress_callback_that_raises_never_takes_the_pass_down(app, monkeypatch, tmp_path):
    """Progress is a display concern; the scoring is the work. A UI counter that
    blows up must not lose a multi-minute pass."""
    import sys
    import tempfile
    from app.services import face_similarity as fsim

    script = tmp_path / 'stub_scorer.py'
    script.write_text(
        'import json, sys\n'
        'json.loads(sys.stdin.read())\n'
        'print("[face] 1/1 scorable sim=0.5", file=sys.stderr, flush=True)\n'
        'print(json.dumps({"ref_ok": True, "results": {}}))\n',
        encoding='utf-8')
    monkeypatch.setattr(fsim, 'is_available', lambda: True)
    monkeypatch.setattr(fsim, '_SCRIPT', str(script))
    monkeypatch.setattr(fsim, '_scoring_python', lambda: sys.executable)

    def _explode(_done, _total):
        raise RuntimeError('the indicator broke')

    with app.app_context():
        with tempfile.TemporaryDirectory() as d:
            ref = os.path.join(d, 'ref.png')
            img_path = os.path.join(d, 'img.png')
            for f in (ref, img_path):
                with open(f, 'wb') as fh:
                    fh.write(_png())
            results, err = fsim.score_dataset_faces(ref, [img_path], on_progress=_explode)
    assert err is None and results == {}


def test_analyze_faces_feeds_the_dataset_indicator_live(app, monkeypatch):
    """analyze_faces must hand the counter to the scorer instead of only filling
    it during the (fast) persist loop — and must NOT then count every image a
    second time, which would push the bar past its own total."""
    from app.services import face_dataset_service as svc
    from app.services import face_similarity as fsim
    from app.services import dataset_activity
    from app.config import LOCAL_USER

    monkeypatch.setattr(fsim, 'is_available', lambda: True)
    with app.app_context():
        ds, img = _dataset_with_ref_and_kept_image(svc, LOCAL_USER)
        path = svc._img_path(img)
        contract = {'ref_ok': True,
                    'results': {path: {'state': 'scorable', 'sim': 0.6, 'det': 0.9,
                                       'bbox_frac': 0.2, 'yaw': 1.0}}}
        live = []

        def _fake(_python, _payload, _timeout, on_progress):
            assert on_progress is not None, 'the counter must reach the scorer'
            on_progress(1, 1)
            live.append(dataset_activity.get(ds.id))
            return (json.dumps(contract), [], 0, False)

        monkeypatch.setattr(fsim, '_run_scorer', _fake)
        counts, err = svc.analyze_faces(LOCAL_USER, ds.id)
        assert err is None and counts == {'scorable': 1}
        # Seen from INSIDE the subprocess call: 1/1 before a single row was written.
        assert live and live[0]['done'] == 1 and live[0]['total'] == 1


# --- dataset_payload exposes face_thresholds --------------------------------

def test_dataset_payload_includes_face_thresholds_from_config(app):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER, save_config
    with app.app_context():
        save_config({'face_scoring': {'green': 0.55, 'orange': 0.42}})
        ds = svc.create_dataset(LOCAL_USER, 'Thresh', 'thresh')
        payload = svc.dataset_payload(LOCAL_USER, ds.id)
        assert payload['face_thresholds'] == {'green': 0.55, 'orange': 0.42}


def test_dataset_payload_face_thresholds_default(app):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Default', 'default')
        payload = svc.dataset_payload(LOCAL_USER, ds.id)
        assert payload['face_thresholds'] == {'green': 0.50, 'orange': 0.45}


def test_score_dataset_faces_crash_reports_stderr_tail(app, monkeypatch):
    """A subprocess that dies with a traceback and no JSON must surface the
    traceback's LAST LINE as the error detail — that line named the real
    problem (nested antelopev2 AssertionError) in the field."""
    import os, tempfile
    from app.services import face_similarity as fsim

    monkeypatch.setattr(fsim, 'is_available', lambda: True)
    _stderr = ('Traceback (most recent call last):\n'
               '  File "face_analysis.py", line 61\n'
               'AssertionError')
    monkeypatch.setattr(
        'app.services.face_similarity._run_scorer',
        lambda *a, **k: _scorer('', stderr=_stderr, returncode=1))
    with app.app_context():
        with tempfile.TemporaryDirectory() as d:
            ref = os.path.join(d, 'ref.png'); img_path = os.path.join(d, 'img.png')
            for f in (ref, img_path):
                with open(f, 'wb') as fh:
                    fh.write(_png())
            results, err = fsim.score_dataset_faces(ref, [img_path])
    assert results == {}
    assert err['kind'] == 'failed' and err['detail'] == 'AssertionError'


def test_score_faces_payload_carries_scoring_error(app, monkeypatch):
    """The Studio route payload must carry scoring_error so the toast can say
    WHY instead of a green « done — 0/N » (user-reported)."""
    import os
    from app.services import face_dataset_service as svc
    from app.services import lora_test_studio as studio
    from app.models import LoraTestImage
    from app.config import LOCAL_USER

    monkeypatch.setattr(
        'app.services.face_similarity.score_dataset_faces',
        lambda ref, paths, timeout=900: ({}, {'kind': 'failed', 'detail': 'AssertionError'}))
    with app.app_context():
        ds, img = _dataset_with_ref_and_kept_image(svc, LOCAL_USER)
        cell_path = os.path.join(svc._dataset_dir(ds.id), 'cell.webp')
        with open(cell_path, 'wb') as fh:
            fh.write(_png())
        svc.db.session.add(LoraTestImage(dataset_id=ds.id, status='done',
                                         filename='cell.webp',
                                         checkpoint='lora_x_000000250.safetensors',
                                         prompt='p', seed=1, strength=1.0))
        svc.db.session.commit()
        out = studio.score_faces(LOCAL_USER, ds.id)
    assert out['scored'] == 0 and out['total'] == 1
    assert out['scoring_error'] == {'kind': 'failed', 'detail': 'AssertionError'}


# --- scope: the pass must reach the images nobody can judge by eye ----------
# Generated variations land at status='pending' with a filename — the triage pile.
# The pass used to filter on status='keep' alone, so the ONE set whose identity
# cannot be eyeballed (a face with no distinctive marking, restaged from a bad
# party photo) was the only set it never scored. Worse, 🎯 Auto-triage has always
# selected on `status === 'pending' && scorable`: a set that could not exist.

def _row(svc, ds, filename, status, source='generated', colour=(0, 0, 255)):
    from app.models import FaceDatasetImage
    if filename:
        with open(os.path.join(svc._dataset_dir(ds.id), filename), 'wb') as fh:
            fh.write(_png(colour))
    img = FaceDatasetImage(dataset_id=ds.id, source=source, status=status,
                           filename=filename, framing='body')
    svc.db.session.add(img)
    svc.db.session.commit()
    return img


def test_generated_variations_awaiting_triage_are_scored(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import face_similarity as fsim
    from app.config import LOCAL_USER

    monkeypatch.setattr(fsim, 'is_available', lambda: True)
    with app.app_context():
        ds, kept = _dataset_with_ref_and_kept_image(svc, LOCAL_USER)
        variation = _row(svc, ds, 'gen1.webp', 'pending')
        rejected = _row(svc, ds, 'gen2.webp', 'reject')
        in_flight = _row(svc, ds, None, 'pending')          # job still running
        failed = _row(svc, ds, 'gen3.webp', 'failed')
        scored = {svc._img_path(i): {'state': 'scorable', 'sim': 0.61, 'det': 0.9,
                                     'bbox_frac': 0.2, 'yaw': 3.0}
                  for i in (kept, variation, rejected, failed)}
        seen = {}

        def _fake_run(*a, **k):
            seen['paths'] = set(json.loads(a[1])['images'])
            return _scorer(json.dumps({'ref_ok': True, 'results': scored}))

        monkeypatch.setattr('app.services.face_similarity._run_scorer', _fake_run)
        svc.analyze_faces(LOCAL_USER, ds.id)

        assert svc._img_path(variation) in seen['paths'], \
            'the triage pile is exactly the set the eye cannot judge'
        assert svc._img_path(kept) in seen['paths'], 'the kept set stays in scope'
        assert svc._img_path(rejected) not in seen['paths'], \
            're-scoring what the user threw away would re-arm auto-triage on it'
        assert svc._img_path(failed) not in seen['paths']
        svc.db.session.refresh(variation)
        svc.db.session.refresh(in_flight)
        assert variation.face_score == 0.61 and variation.face_state == 'scorable'
        assert in_flight.face_state is None, 'a row with no file has nothing to score'


def test_the_scope_counter_matches_what_the_pass_will_do(app):
    """The button's label promises a number; it must be the same set the pass
    walks, computed from the already-loaded rows (no extra query)."""
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER

    with app.app_context():
        ds, kept = _dataset_with_ref_and_kept_image(svc, LOCAL_USER)
        _row(svc, ds, 'a.webp', 'pending')
        _row(svc, ds, 'b.webp', 'pending')
        _row(svc, ds, 'c.webp', 'reject')
        _row(svc, ds, None, 'pending')
        kept.face_state = 'scorable'
        svc.db.session.commit()
        payload = svc.dataset_payload(LOCAL_USER, ds.id)
        rows = svc.face_scoring_rows(ds.id)

    assert payload['face_scoring_scope'] == {'total': 3, 'unscored': 2}
    assert len(rows) == 3, 'the counter and the query must not drift apart'


# --- the reference's SHAPE, published so the UI can warn about Krea framing --

def test_the_payload_publishes_the_reference_pixel_size(app):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER

    with app.app_context():
        ds, _img = _dataset_with_ref_and_kept_image(svc, LOCAL_USER)
        # _png() writes a 64x64 square — a shape Krea would reproduce verbatim.
        payload = svc.dataset_payload(LOCAL_USER, ds.id)
        assert (payload['ref_width'], payload['ref_height']) == (64, 64)

        # A reference the server cannot measure (truncated file, exotic codec,
        # Pillow missing) must cost a missing warning, never a broken workspace.
        with open(svc._ref_path(ds), 'wb') as fh:
            fh.write(b'not an image at all')
        payload = svc.dataset_payload(LOCAL_USER, ds.id)
        assert payload['ref_width'] is None and payload['ref_height'] is None

        os.remove(svc._ref_path(ds))
        payload = svc.dataset_payload(LOCAL_USER, ds.id)
        assert payload['ref_width'] is None

        assert svc.image_pixel_size(os.path.join(svc._dataset_dir(ds.id), 'nope.webp')) is None


def test_the_reference_shape_is_cached_but_never_stale(app, monkeypatch):
    """dataset_payload is POLLED, so measuring the reference on every call was a
    fresh disk open on a hot path forever. Cached on the file's identity — a
    re-crop rewrites the SAME filename, and a stale shape here would silence (or
    invent) the "your square reference will squeeze the body shots" warning."""
    import io as _io
    from PIL import Image as _Image
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER

    with app.app_context():
        ds, _img = _dataset_with_ref_and_kept_image(svc, LOCAL_USER)
        ref = svc._ref_path(ds)
        assert svc.image_pixel_size(ref) == (64, 64)

        opens = {'n': 0}
        real_open = _Image.open

        def _counting_open(*a, **k):
            opens['n'] += 1
            return real_open(*a, **k)

        monkeypatch.setattr('PIL.Image.open', _counting_open)
        for _ in range(5):
            assert svc.image_pixel_size(ref) == (64, 64)
        assert opens['n'] == 0, 'a polled payload must not re-open the file every time'

        # Re-crop: same filename, different content → the cache must follow.
        buf = _io.BytesIO()
        _Image.new('RGB', (100, 40), (1, 2, 3)).save(buf, 'PNG')
        with open(ref, 'wb') as fh:
            fh.write(buf.getvalue())
        assert svc.image_pixel_size(ref) == (100, 40)


def test_a_dataset_with_no_reference_reports_no_shape(app):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'NoRef', 'noref')
        payload = svc.dataset_payload(LOCAL_USER, ds.id)
        assert payload['ref_width'] is None and payload['ref_height'] is None


def test_the_timeout_grows_with_the_set_it_has_to_walk():
    """A timeout returns NO partial result, so a set big enough to blow the old
    flat 900 s budget lost the whole pass. Widening the scope made that reachable."""
    from app.services import face_similarity as fsim
    assert fsim.default_timeout(0) == 900
    assert fsim.default_timeout(100) == 900, 'small sets keep the generous floor'
    assert fsim.default_timeout(2000) > 900
    assert fsim.default_timeout(2000) > fsim.default_timeout(1000)
    assert fsim.default_timeout(None) == 900
