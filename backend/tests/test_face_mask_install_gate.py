"""Face masking is an OPTIONAL capability — so its absence must be a POSED
DECISION, never a silent downgrade, and never a wall.

Two halves:
  1. the pre-launch report says "this run will train unmasked" when the user
     asked for face masking and InsightFace is not installed — as a WARNING
     (one explicit confirm in the UI), because continuing unmasked is a
     legitimate choice on a machine that deliberately never installed it;
  2. the cases where masking is refused BY DESIGN (character/style dataset,
     slider mode) stay silent — installing InsightFace would not change a thing
     there, so warning about it would be noise. This half is the anti-regression
     guard-rail of this pass.

Also asserts the ML-venv isolation this feature rides on is not broken:
insightface must never land in the Flask venv's own requirements.
"""
import os

from PIL import Image

from app.extensions import db
from app.models import FaceDatasetImage
from app.services import face_dataset_service as svc
from app.config import LOCAL_USER, save_config

CONCEPT_DESC = 'balancing a spoon'      # names no body part -> no face conflict


def _dataset(tmp_path, kind='concept', n=14, trigger='fmi_act'):
    """Enough kept images to clear every family floor, so the only thing the
    preflight can complain about here is the face-mask capability."""
    save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
    kw = {'kind': kind}
    if kind == 'concept':
        kw['concept_desc'] = CONCEPT_DESC
    ds = svc.create_dataset(LOCAL_USER, 'FMI', trigger, **kw)
    img_dir = svc._dataset_dir(ds.id)
    for i in range(n):
        fn = f'k{i}.png'
        Image.new('RGB', (64, 64)).save(os.path.join(img_dir, fn))
        db.session.add(FaceDatasetImage(
            dataset_id=ds.id, status='keep', filename=fn, framing='body',
            caption=f'a person balancing a spoon on the nose, shot number {i}'))
    db.session.commit()
    return ds


def _mask_warning(report):
    return [w for w in report['warnings'] if 'InsightFace' in w]


def _mask_check(report):
    return [c for c in report.get('checks') or [] if c['id'] == 'face_mask']


# --- 1. the decision is posed -------------------------------------------------
def test_preflight_warns_when_masking_is_on_but_insightface_is_missing(app, tmp_path,
                                                                      monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_mask
    with app.app_context():
        ds = _dataset(tmp_path)
        lt.update_train_settings(LOCAL_USER, ds.id, {'mask_faces': True})
        monkeypatch.setattr(face_mask, 'is_available', lambda: False)
        r = lt.training_preflight(LOCAL_USER, ds.id)
    warn = _mask_warning(r)
    assert warn, 'a requested face mask that cannot be produced must be announced'
    assert 'unmasked' in warn[0]
    # A WARNING, not a blocker: the run stays possible for someone who declines
    # a 400 MB download — they just decide it knowingly.
    assert not any('InsightFace' in b for b in r['blockers'])
    row = _mask_check(r)
    assert row and row[0]['status'] == 'warn'


def test_preflight_is_silent_when_insightface_is_available(app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_mask
    with app.app_context():
        ds = _dataset(tmp_path)
        lt.update_train_settings(LOCAL_USER, ds.id, {'mask_faces': True})
        monkeypatch.setattr(face_mask, 'is_available', lambda: True)
        r = lt.training_preflight(LOCAL_USER, ds.id)
    assert not _mask_warning(r)
    row = _mask_check(r)
    assert row and row[0]['status'] == 'ok'


def test_preflight_is_silent_when_masking_was_never_asked_for(app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_mask
    with app.app_context():
        ds = _dataset(tmp_path)                       # no mask_faces opt-in
        monkeypatch.setattr(face_mask, 'is_available', lambda: False)
        r = lt.training_preflight(LOCAL_USER, ds.id)
    assert not _mask_warning(r)
    assert not _mask_check(r)      # nothing to report -> no row at all


# --- 2. refused BY DESIGN stays silent (anti-regression) ----------------------
def test_a_character_dataset_is_never_warned_about_face_masking(app, tmp_path,
                                                                monkeypatch):
    """face_masking_enabled() already refuses non-concept datasets on purpose —
    a Character wants its identity learned. Installing InsightFace would change
    nothing, so this must not produce a warning."""
    from app.services import lora_training as lt
    from app.services import face_mask
    with app.app_context():
        ds = _dataset(tmp_path, kind='character', trigger='fmi_char')
        lt.update_train_settings(LOCAL_USER, ds.id, {'mask_faces': True})
        monkeypatch.setattr(face_mask, 'is_available', lambda: False)
        r = lt.training_preflight(LOCAL_USER, ds.id)
    assert not _mask_warning(r)
    assert not _mask_check(r)


def test_slider_mode_is_never_warned_about_face_masking(app, tmp_path, monkeypatch):
    """Slider mode forces face masking OFF by design (the guided slider loss never
    reads batch.mask_tensor). Degradation is INTENDED there -> stay silent."""
    from app.services import lora_training as lt
    from app.services import face_mask
    with app.app_context():
        ds = _dataset(tmp_path, trigger='fmi_slider')
        lt.update_train_settings(LOCAL_USER, ds.id, {'mask_faces': True})
        monkeypatch.setattr(lt, 'slider_mode_enabled', lambda _ds: True)
        monkeypatch.setattr(face_mask, 'is_available', lambda: False)
        r = lt.training_preflight(LOCAL_USER, ds.id)
    assert not _mask_warning(r)
    assert not _mask_check(r)


def test_a_probe_failure_never_blocks_the_preflight(app, tmp_path, monkeypatch):
    """is_available() shells out to another interpreter; an exception there must
    degrade to "say nothing", never take the whole report down."""
    from app.services import lora_training as lt
    from app.services import face_mask

    def boom():
        raise OSError('no such interpreter')

    with app.app_context():
        ds = _dataset(tmp_path, trigger='fmi_boom')
        lt.update_train_settings(LOCAL_USER, ds.id, {'mask_faces': True})
        monkeypatch.setattr(face_mask, 'is_available', boom)
        r = lt.training_preflight(LOCAL_USER, ds.id)
    assert 'verdict' in r          # the report still came back whole


# --- 3. the ML-venv isolation this whole feature depends on ------------------
def test_insightface_never_enters_the_flask_venv_requirements():
    """InsightFace pins numpy<2, which the Flask venv must not inherit. It lives
    ONLY in requirements-ml.txt, installed into the interpreter face_scoring
    resolves (which may be a dedicated 3.10-3.12 env)."""
    from app import setup_installer
    from app import config as cfg

    base = (cfg.BACKEND_DIR / 'requirements.txt').read_text(encoding='utf-8').lower()
    assert 'insightface' not in base
    assert 'onnxruntime' not in base
    ml = setup_installer._ML_REQUIREMENTS.read_text(encoding='utf-8').lower()
    assert 'insightface' in ml
    # ...and the scoped installer targets the capability's own interpreter, not
    # sys.executable unconditionally.
    assert 'insightface' in setup_installer._CAPABILITY_PACKAGES['face_scoring']


def test_out_of_range_python_is_explained_in_plain_english_not_a_pip_traceback(
        app, monkeypatch):
    """The most likely state of a brand-new install: a Python too recent for the
    insightface wheels. The install log must LEAD with why, not end with a
    cryptic source-build failure."""
    from app import setup_installer, capabilities
    lines = []
    monkeypatch.setattr(setup_installer, '_append', lambda a, l: lines.append(l))
    monkeypatch.setattr(setup_installer, '_run_pip', lambda a, cmd: 1)
    monkeypatch.setattr(setup_installer, '_capability_python',
                        lambda a: setup_installer.sys.executable)
    monkeypatch.setattr(capabilities, 'python_ml_status',
                        lambda: {'version': '3.14.0', 'ml_supported': False,
                                 'ml_range': '3.10–3.12'})
    with app.app_context():
        setup_installer._run_ml_capability('face_scoring')
    joined = '\n'.join(lines)
    assert '3.14.0' in joined and '3.10–3.12' in joined
    assert 'face_scoring.python' in joined
