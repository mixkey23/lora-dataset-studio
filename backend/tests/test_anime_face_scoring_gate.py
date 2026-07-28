"""Face SIMILARITY scoring is off for `anime` datasets — and says why.

InsightFace/antelopev2 is a detector+embedder trained on PHOTOGRAPHED faces. On a
drawn character it mostly detects nothing, and when it does the cosine it returns
is meaningless. Before this gate the pass still ran and failed OPEN: grey "no face
detected" tiles everywhere, or worse a plausible-looking number, with nothing on
screen saying the tool simply cannot read this kind of image.

What this file locks:
  1. ONE decision point (`face_scoring_block_reason`) that every scoring entry
     point consults, so the rule can never diverge across the three lanes.
  2. The three lanes actually refuse: the dataset pass, the Studio cell scoring,
     and best-epoch selection — each returning the REASON in its own shape.
  3. Strict non-regression for `human` (and every other subject type): the pass
     runs exactly as before.
  4. The gate is scoped to InsightFace ONLY. `face_crop_to_square_webp` /
     `detect_head_bbox` go through Qwen3-VL, a general vision model that reads a
     drawn head fine — head-cropping an anime reference must keep working. Gating
     it "by analogy" would have broken a feature that was never broken.
  5. Nothing is destroyed: face_score/face_state rows written before the dataset
     was marked anime survive the gate untouched (flip the subject type back to
     Human and they are all still there).
"""
import io
import json
import os

import pytest


def _png(color=(255, 0, 0), size=(64, 64)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new('RGB', size, color).save(buf, 'PNG')
    return buf.getvalue()


def _dataset(svc, user, subject_type):
    ds = svc.create_dataset(user, f'ds-{subject_type}', 'ztrig', subject_type=subject_type)
    d = svc._dataset_dir(ds.id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'ref.webp'), 'wb') as fh:
        fh.write(_png())
    ds.ref_filename = 'ref.webp'
    svc.db.session.commit()
    from app.models import FaceDatasetImage
    fn = 'kept.webp'
    with open(os.path.join(d, fn), 'wb') as fh:
        fh.write(_png((0, 255, 0)))
    img = FaceDatasetImage(dataset_id=ds.id, source='import', status='keep',
                           filename=fn, framing='face')
    svc.db.session.add(img)
    svc.db.session.commit()
    return ds, img


# --- 1. the single decision point ---------------------------------------------

def test_block_reason_is_the_one_place_the_rule_lives(app):
    """A gate posted in three places diverges. Every lane calls THIS."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    with app.app_context():
        anime, _ = _dataset(svc, LOCAL_USER, 'anime')
        human, _ = _dataset(svc, LOCAL_USER, 'human')
        reason = svc.face_scoring_block_reason(anime)
        assert isinstance(reason, str) and reason
        # It must EXPLAIN, not merely refuse (the whole point of the change).
        assert 'photographic' in reason.lower() and 'drawn' in reason.lower()
        assert svc.face_scoring_block_reason(human) is None
        # Legacy rows (column NULL) score exactly as before.
        human.subject_type = None
        assert svc.face_scoring_block_reason(human) is None
        assert svc.face_scoring_block_reason(None) is None


def test_every_other_subject_type_still_scores(app):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import face_variations as fv
    with app.app_context():
        for st in fv.SUBJECT_TYPES:
            ds, _ = _dataset(svc, LOCAL_USER, st)
            blocked = svc.face_scoring_block_reason(ds) is not None
            assert blocked == (st == 'anime'), st


# --- 2. the dataset pass refuses, and says why --------------------------------

def test_anime_dataset_pass_does_not_run_and_carries_the_reason(app, monkeypatch):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    def _boom(*a, **k):   # noqa: ANN001
        raise AssertionError('InsightFace must never be invoked for an anime dataset')

    monkeypatch.setattr('app.services.face_similarity.score_dataset_faces', _boom)
    with app.app_context():
        ds, img = _dataset(svc, LOCAL_USER, 'anime')
        counts, err = svc.analyze_faces(LOCAL_USER, ds.id)
        assert counts == {}
        assert err and err['kind'] == 'subject_not_photographic'
        assert err['detail'] == svc.face_scoring_block_reason(ds)
        # No silent write either.
        assert img.face_state is None and img.face_score is None


def test_human_dataset_pass_is_strictly_unchanged(app, monkeypatch):
    """Non-regression: the gate must be invisible to every photographic dataset."""
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import face_similarity as fsim

    calls = []
    with app.app_context():
        ds, img = _dataset(svc, LOCAL_USER, 'human')
        path = svc._img_path(img)

        def _fake(ref, paths, **k):   # noqa: ANN001
            calls.append(paths)
            return {path: {'state': 'scorable', 'sim': 0.62}}, None

        monkeypatch.setattr(fsim, 'score_dataset_faces', _fake)
        counts, err = svc.analyze_faces(LOCAL_USER, ds.id)
        assert calls == [[path]]
        assert err is None and counts == {'scorable': 1}
        row = svc.db.session.get(FaceDatasetImage, img.id)
        assert row.face_state == 'scorable' and row.face_score == 0.62


def test_the_anime_reason_beats_the_missing_reference_error(app):
    """An anime dataset with no reference gets the USEFUL message, not "set a
    reference photo first" (which would send the user to fix the wrong thing)."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds, _ = _dataset(svc, LOCAL_USER, 'anime')
        ds.ref_filename = None
        svc.db.session.commit()
        counts, err = svc.analyze_faces(LOCAL_USER, ds.id)
        assert counts == {} and err['kind'] == 'subject_not_photographic'


def test_the_dataset_payload_publishes_the_reason(app):
    """The UI must not re-derive the rule — it reads the server's string."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    with app.app_context():
        anime, _ = _dataset(svc, LOCAL_USER, 'anime')
        human, _ = _dataset(svc, LOCAL_USER, 'human')
        pa = svc.dataset_payload(LOCAL_USER, anime.id)
        ph = svc.dataset_payload(LOCAL_USER, human.id)
        assert pa['face_scoring_blocked'] == svc.face_scoring_block_reason(anime)
        assert ph['face_scoring_blocked'] is None


def test_the_route_refuses_without_shelling_out(app, client, monkeypatch):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    monkeypatch.setattr('app.services.face_similarity._run_scorer',
                        lambda *a, **k: pytest.fail('no subprocess for an anime dataset'))
    with app.app_context():
        ds, _ = _dataset(svc, LOCAL_USER, 'anime')
        dsid = ds.id
    d = client.post(f'/api/dataset/{dsid}/analyze-faces').get_json()
    assert d['ok'] is True and d['analyzed'] == 0
    assert d['scoring_error']['kind'] == 'subject_not_photographic'


# --- 3. the two other InsightFace lanes ---------------------------------------

def test_best_epoch_refuses_with_its_reason(app, monkeypatch):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt

    monkeypatch.setattr('app.services.face_similarity.score_dataset_faces',
                        lambda *a, **k: pytest.fail('InsightFace on an anime dataset'))
    with app.app_context():
        ds, _ = _dataset(svc, LOCAL_USER, 'anime')
        out = lt.score_checkpoint_samples(LOCAL_USER, ds.id, '', 'zimage')
        assert out['available'] is False
        assert out['reason'] == svc.face_scoring_block_reason(ds)


def test_studio_cell_scoring_refuses_with_its_reason(app, monkeypatch):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import lora_test_studio as lts

    monkeypatch.setattr('app.services.face_similarity.score_dataset_faces',
                        lambda *a, **k: pytest.fail('InsightFace on an anime dataset'))
    with app.app_context():
        ds, _ = _dataset(svc, LOCAL_USER, 'anime')
        out = lts.score_faces(LOCAL_USER, ds.id)
        assert out['scored'] == 0
        assert out['scoring_error']['kind'] == 'subject_not_photographic'
        assert out['scoring_error']['detail'] == svc.face_scoring_block_reason(ds)


# --- 4. head-cropping is NOT InsightFace, and stays on -------------------------

def test_head_crop_still_runs_for_an_anime_reference(app, monkeypatch):
    """`detect_head_bbox` is Qwen3-VL (a general vision model), not InsightFace —
    it reads a drawn head. Gating it would have removed a working feature."""
    from app.services import face_dataset_service as svc

    seen = []

    def _vision(image_bytes, prompt, **k):   # noqa: ANN001
        seen.append(prompt)
        return json.dumps({'x1': 300, 'y1': 200, 'x2': 700, 'y2': 600})

    monkeypatch.setattr('app.services.vision_ollama.describe_image_ollama', _vision)
    webp, detected = svc.face_crop_to_square_webp(_png(size=(512, 512)),
                                                  size=256, return_detected=True)
    assert seen, 'the vision model must still be consulted'
    assert detected is True and webp[:4] == b'RIFF'


def test_the_gate_names_insightface_lanes_only(app):
    """Belt and braces: the block helper is about SIMILARITY. Nothing in the crop
    path consults it (a future refactor that wires it in fails here)."""
    import inspect
    from app.services import face_dataset_service as svc
    for fn in (svc.face_crop_to_square_webp, svc.detect_head_bbox):
        assert 'face_scoring_block_reason' not in inspect.getsource(fn)


# --- 5. existing scores are IGNORED, never destroyed ---------------------------

def test_scores_written_before_the_switch_survive(app):
    """Marking a dataset anime must not delete data the user could still want: the
    rows stay, so flipping the subject type back to Human restores them intact."""
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds, img = _dataset(svc, LOCAL_USER, 'human')
        img.face_state, img.face_score = 'scorable', 0.71
        svc.db.session.commit()
        ds.subject_type = 'anime'
        svc.db.session.commit()
        counts, err = svc.analyze_faces(LOCAL_USER, ds.id)
        assert err['kind'] == 'subject_not_photographic' and counts == {}
        row = svc.db.session.get(FaceDatasetImage, img.id)
        assert row.face_state == 'scorable' and row.face_score == 0.71
        # ... and the payload still carries them (nothing is scrubbed server-side).
        p = svc.dataset_payload(LOCAL_USER, ds.id)
        assert p['images'][0]['face_score'] == 0.71
