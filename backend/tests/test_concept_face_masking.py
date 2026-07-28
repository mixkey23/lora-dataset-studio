"""Concept face masking — issue #15, reported by shivdbz2010 (GitHub).

A concept LoRA also learns the FACES of its dataset, so combining it with a
character LoRA makes the two fight over the identity. Masking the faces during
concept training teaches the act without the identity.

The central assertion, and the reason this file exists: the export guard used to
read "concept => no masks at all", because when it was written the only mask that
existed was a PERSON mask (subject white, background black), which would indeed
erase the concept. A FACE mask is the OPPOSITE polarity (face black, everything
else white) and does not have that defect. The guard must therefore tell the two
apart instead of confusing them.
"""
import json
import os

from PIL import Image

from app.extensions import db
from app.models import FaceDatasetImage
from app.services import face_dataset_service as svc
from app.config import LOCAL_USER, save_config

CONCEPT_DESC = 'balancing a spoon'   # names no body part -> no face conflict


def _dataset(tmp_path, kind='concept', desc=CONCEPT_DESC, n=3, trigger='cfm_act'):
    save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
    kw = {'kind': kind}
    if kind == 'concept':
        kw['concept_desc'] = desc
    ds = svc.create_dataset(LOCAL_USER, 'CFM', trigger, **kw)
    img_dir = svc._dataset_dir(ds.id)
    for i in range(n):
        fn = f'k{i}.png'
        Image.new('RGB', (64, 64)).save(os.path.join(img_dir, fn))
        db.session.add(FaceDatasetImage(dataset_id=ds.id, status='keep', filename=fn))
    db.session.commit()
    return ds


def _fake_face_masks(paths, out_dir, expand=None, timeout=None):
    os.makedirs(out_dir, exist_ok=True)
    for p in paths:
        Image.new('L', (8, 8), 255).save(
            os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + '.png'))
    return {'ok': True, 'written': len(paths),
            'results': {p: {'state': 'masked', 'boxes': [[.4, .3, .6, .5]]} for p in paths}}


# --- THE central assertion -------------------------------------------------
def test_concept_optin_produces_face_masks_and_a_mask_path(app, tmp_path, monkeypatch):
    """A concept dataset that opted in exports FACE masks and the job config
    carries mask_path — while the person-mask guard stays untouched."""
    from app.services import lora_training as lt
    with app.app_context():
        ds = _dataset(tmp_path)
        lt.update_train_settings(LOCAL_USER, ds.id, {'mask_faces': True})

        def boom(*a, **k):
            raise AssertionError('a CONCEPT must never get a PERSON mask')

        monkeypatch.setattr(lt, 'generate_person_masks', boom)
        monkeypatch.setattr(lt.face_mask, 'generate_face_masks', _fake_face_masks)

        out = lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=True)
        masks_dir = lt._masks_dir(out)
        assert os.path.isdir(masks_dir), 'face masks were not written'
        fields = lt._mask_fields(out)
        assert fields['mask_path'] == masks_dir
        # ...and at the FACE weight, not the person-mask constant.
        assert fields['mask_min_value'] == lt.face_mask.min_weight()


def test_concept_without_optin_is_byte_identical_to_before(app, tmp_path, monkeypatch):
    """No opt-in -> no masks of either polarity. An existing concept dataset must
    not change behaviour just because the feature shipped."""
    from app.services import lora_training as lt
    with app.app_context():
        ds = _dataset(tmp_path, trigger='cfm_noopt')

        def boom(*a, **k):
            raise AssertionError('no mask pass may run without the opt-in')

        monkeypatch.setattr(lt, 'generate_person_masks', boom)
        monkeypatch.setattr(lt.face_mask, 'generate_face_masks', boom)
        out = lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=True)
        assert not os.path.isdir(lt._masks_dir(out))
        assert lt._mask_fields(out) == {}


def test_character_and_style_never_face_mask(app, tmp_path, monkeypatch):
    """Concept only: a Character wants its identity learned, a Style must learn
    how it renders a face."""
    from app.services import lora_training as lt
    with app.app_context():
        for kind, trigger in (('character', 'cfm_char'), ('style', 'cfm_style')):
            ds = _dataset(tmp_path, kind=kind, trigger=trigger)
            # Force the stored flag even though the UI would not offer it.
            ds.train_settings = json.dumps({'mask_faces': True})
            db.session.commit()
            assert svc.face_masking_enabled(ds) is False

            def boom(*a, **k):
                raise AssertionError(f'{kind} must never get a face mask')

            monkeypatch.setattr(lt.face_mask, 'generate_face_masks', boom)
            monkeypatch.setattr(lt, 'generate_person_masks',
                                lambda paths, out_dir: {'ok': True, 'written': 0, 'results': {}})
            lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=True)


def test_person_mask_keeps_the_historical_weight_when_the_face_knob_moves(app, tmp_path,
                                                                         monkeypatch):
    """The new face knob must not silently re-weight every character run's
    background — the two polarities do not want the same weight."""
    from app.services import lora_training as lt
    with app.app_context():
        save_config({'aitoolkit': {'dir': str(tmp_path / 'ai2')},
                     'face_mask': {'min_weight': 0.6}})
        ds = _dataset(tmp_path, kind='character', trigger='cfm_charw')

        def fake_person(paths, out_dir):
            os.makedirs(out_dir, exist_ok=True)
            for p in paths:
                Image.new('L', (8, 8), 255).save(
                    os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + '.png'))
            return {'ok': True, 'written': len(paths), 'results': {}}

        monkeypatch.setattr(lt, 'generate_person_masks', fake_person)
        out = lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=True)
        assert lt._mask_fields(out)['mask_min_value'] == lt._PERSON_MASK_MIN_VALUE


def test_masks_unavailable_degrades_and_is_reported(app, tmp_path, monkeypatch):
    """No InsightFace -> no masks, no crash, and the run is FLAGGED so the progress
    view can say it trained unmasked instead of lying by silence."""
    from app.services import lora_training as lt
    from app.job_queue import queue_manager
    with app.app_context():
        ds = _dataset(tmp_path, trigger='cfm_nodetect')
        lt.update_train_settings(LOCAL_USER, ds.id, {'mask_faces': True})
        monkeypatch.setattr(lt.face_mask, 'generate_face_masks',
                            lambda *a, **k: {})          # capability absent
        out = lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=True)
        assert not os.path.isdir(lt._masks_dir(out))
        assert lt._mask_fields(out) == {}
        assert queue_manager._get_system_state('training_masks_skipped') is True


# --- the knobs -------------------------------------------------------------
def test_expand_and_min_weight_are_clamped_not_trusted(app):
    """A hand-edited config.json must degrade, never kill an export."""
    from app.services import face_mask as fm
    with app.app_context():
        save_config({'face_mask': {'expand': 99, 'min_weight': 0.0}})
        assert fm.expand_factor() == fm.EXPAND_MAX
        # zero is refused ON PURPOSE: ai-toolkit divides the mask by its own mean,
        # so an all-black mask at 0.0 divides by zero and NaNs the run; and an
        # unpenalised region degenerates (the only published sweep of this knob).
        assert fm.min_weight() == fm.MIN_WEIGHT_MIN
        save_config({'face_mask': {'expand': 'nonsense', 'min_weight': None}})
        assert fm.expand_factor() == 2.0
        assert fm.min_weight() == 0.1


def test_dilate_box_is_the_shared_arithmetic():
    """The preview draws what the trainer will be given, so this formula is
    mirrored in frontend/src/utils/faceMaskBox.js — same numbers, both sides."""
    import importlib.util
    from app import config as cfg
    spec = importlib.util.spec_from_file_location(
        'face_mask_infer', str(cfg.BACKEND_DIR / 'infer' / 'face_mask_infer.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # 0.2 wide, 0.2 tall, centred at (0.5, 0.5); expand 2.0 with the 10% upward bias
    x1, y1, x2, y2 = mod.dilate_box((0.4, 0.4, 0.6, 0.6), 2.0)
    assert (round(x1, 6), round(x2, 6)) == (0.3, 0.7)
    assert (round(y1, 6), round(y2, 6)) == (0.28, 0.68)
    # expand 1.0 = the face box itself, only shifted
    a = mod.dilate_box((0.0, 0.0, 1.0, 1.0), 1.0)
    assert round(a[1], 6) == -0.1 and round(a[3], 6) == 0.9


# --- the "the face IS the concept" warning ---------------------------------
def test_a_face_anchored_concept_is_flagged_but_not_blocked(app, tmp_path):
    """Masking the head can erase a concept that lives on the face. We warn — the
    user is the only one who knows their dataset — and never refuse."""
    with app.app_context():
        risky = _dataset(tmp_path, desc='ahegao expression', trigger='cfm_face')
        assert svc.concept_face_conflict(risky) is True
        safe = _dataset(tmp_path, desc='balancing a spoon', trigger='cfm_safe')
        assert svc.concept_face_conflict(safe) is False
        # the flag never gates the opt-in
        from app.services import lora_training as lt
        lt.update_train_settings(LOCAL_USER, risky.id, {'mask_faces': True})
        assert svc.face_masking_enabled(risky) is True


def test_coverage_summary_counts_what_the_user_must_see(app):
    from app.services import face_mask as fm
    res = {'a': {'state': 'masked'}, 'b': {'state': 'masked'}, 'c': {'state': 'no_face'},
           'd': {'state': 'too_large'}, 'e': {'state': 'error'}}
    assert fm.coverage_summary(res) == {'total': 5, 'masked': 2, 'no_face': 1,
                                        'too_large': 1, 'failed': 1}
