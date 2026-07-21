"""Render-style axis (Wave 3): the target AESTHETIC of generated dataset
variations, orthogonal to kind/train_type. The load-bearing invariant:
render_style='photoreal' (the default, and the historical no-op) must produce
BYTE-IDENTICAL wrapper output to before this axis existed — every other
assertion here is secondary to that one.
"""
import pytest

from app.services import face_variations as fv
from app.services import render_style_presets as rsp


def _patch_overrides(monkeypatch, mapping):
    """Make cfg.get answer identity_prompts.<kind> from `mapping`, default else
    (mirrors test_identity_prompts_override.py's helper)."""
    import app.config as cfg

    def fake_get(key, default=None):
        if key.startswith('identity_prompts.'):
            return mapping.get(key.split('.', 1)[1], default)
        return default

    monkeypatch.setattr(cfg, 'get', fake_get)


# --- render_style_presets.py (pure) -----------------------------------------

def test_generation_tail_for_photoreal_and_custom_are_none():
    assert rsp.generation_tail_for('photoreal') is None
    assert rsp.generation_tail_for('custom') is None
    assert rsp.generation_tail_for(None) is None
    assert rsp.generation_tail_for('bogus-unknown-style') is None


def test_generation_tail_for_known_non_photoreal_styles():
    for style in ('render_3d', 'anime_2d', 'cartoon_semireal', 'illustration'):
        tail = rsp.generation_tail_for(style)
        assert isinstance(tail, str) and tail.strip()


def test_render_style_label():
    assert rsp.render_style_label('anime_2d') == '2D anime'
    assert rsp.render_style_label(None) == 'Photoreal'
    assert rsp.render_style_label('bogus') == 'Photoreal'


# --- normalize_render_style ---------------------------------------------------

def test_normalize_render_style_allowlist():
    from app.services import face_dataset_service as svc
    for style in svc.RENDER_STYLES:
        assert svc.normalize_render_style(style) == style
        assert svc.normalize_render_style(style.upper()) == style   # case-fold


def test_normalize_render_style_photoreal_and_unknown_fall_to_none():
    from app.services import face_dataset_service as svc
    assert svc.normalize_render_style('photoreal') is None
    assert svc.normalize_render_style('bogus') is None
    assert svc.normalize_render_style('') is None
    assert svc.normalize_render_style(None) is None


# --- byte-identical default path (THE critical invariant) --------------------

def test_wrap_variation_default_is_byte_identical(monkeypatch):
    _patch_overrides(monkeypatch, {})
    assert fv.wrap_variation('p') == f'{fv.IDENTITY_GUARD} p'
    assert fv.wrap_variation('p', render_style='photoreal') == f'{fv.IDENTITY_GUARD} p'
    assert fv.wrap_variation('p', ref_count=2) == f'{fv.IDENTITY_GUARD_MULTI} p'
    assert fv.wrap_variation('p', ref_count=2, render_style='photoreal') == f'{fv.IDENTITY_GUARD_MULTI} p'
    # An unrecognized/custom style is ALSO a no-op (no tail defined for it).
    assert fv.wrap_variation('p', render_style='custom') == f'{fv.IDENTITY_GUARD} p'
    assert fv.wrap_variation('p', render_style='bogus') == f'{fv.IDENTITY_GUARD} p'


def test_wrap_variation_klein_default_is_byte_identical(monkeypatch):
    _patch_overrides(monkeypatch, {})
    default = fv.wrap_variation_klein('turn to profile', framing='bust')
    assert fv.wrap_variation_klein('turn to profile', framing='bust',
                                   render_style='photoreal') == default
    assert fv.wrap_variation_klein('turn to profile', framing='bust',
                                   render_style='custom') == default
    assert fv.wrap_variation_klein('turn to profile', framing='bust',
                                   render_style='bogus') == default
    assert default.startswith('Create a new photograph of the same person')
    assert fv.IDENTITY_GUARD_KLEIN in default
    assert default.endswith('Professional realistic photograph, SFW.')


# --- non-photoreal presets swap the tail, keep the identity lock -------------

def test_wrap_variation_anime_2d_swaps_tail_and_base_wording(monkeypatch):
    _patch_overrides(monkeypatch, {})
    out = fv.wrap_variation('p', render_style='anime_2d')
    assert out.startswith(fv._IDENTITY_GUARD_BASE_STYLED)
    assert 'SFW, realistic photographic portrait.' not in out
    assert rsp.generation_tail_for('anime_2d') in out
    # Photo-specific wording is gone; the rest of the identity lock (eye
    # shape/color, nose, jawline, lips, face proportions) stays.
    assert 'SAME person' not in out
    assert 'SAME character' in out
    assert 'skin tone and texture' not in out
    assert 'same eye shape and color, nose, jawline, lips, and face proportions' in out


def test_wrap_variation_klein_render_3d_swaps_subject_noun_and_ending_no_duplication(monkeypatch):
    _patch_overrides(monkeypatch, {})
    out = fv.wrap_variation_klein('turn to profile', framing='bust', render_style='render_3d')
    assert out.startswith('Create a new image of the same character')
    assert 'Create a new photograph' not in out
    assert 'same person' not in out
    tail = rsp.generation_tail_for('render_3d')
    # Exactly one occurrence — the identity block must NOT repeat the ending's tail.
    assert out.count(tail) == 1
    assert 'Sharp focus, natural skin texture with visible pores' not in out
    assert 'Professional realistic photograph' not in out
    assert 'skin tone and texture' not in out


def test_wrap_variation_klein_nsfw_non_photoreal_ending(monkeypatch):
    _patch_overrides(monkeypatch, {})
    out = fv.wrap_variation_klein('p', nsfw=True, render_style='anime_2d')
    assert 'Explicit nudity is allowed; render natural, anatomically correct forms.' in out
    assert out.count(rsp.generation_tail_for('anime_2d')) == 1
    assert 'SFW' not in out


# --- identity_prompts override always wins, over BOTH default and tail -------

def test_override_wins_over_render_style_tail(monkeypatch):
    _patch_overrides(monkeypatch, {'face_single': 'MY CUSTOM GUARD.'})
    out = fv.wrap_variation('p', render_style='anime_2d')
    assert out == 'MY CUSTOM GUARD. p'
    assert rsp.generation_tail_for('anime_2d') not in out


def test_klein_identity_override_wins_over_render_style_tail(monkeypatch):
    _patch_overrides(monkeypatch, {'klein_identity': 'KEEP THE FACE.'})
    out = fv.wrap_variation_klein('p', render_style='anime_2d')
    assert 'KEEP THE FACE.' in out
    assert 'Restage the shot to match this description' not in out
    # The override still appears exactly once (no duplicated tail injection).
    assert out.count('KEEP THE FACE.') == 1


# --- _effective_klein_lora_strength ------------------------------------------

def test_effective_klein_lora_strength_forces_off_for_non_photoreal(app):
    from app.services import face_dataset_service as svc
    from app.services import face_dataset_service as fds
    from app.config import LOCAL_USER
    with app.app_context():
        ds = fds.create_dataset(LOCAL_USER, 'RS', 'zchar_rs')
        ds.render_style = 'anime_2d'
        assert svc._effective_klein_lora_strength(ds, None) == 0.0


def test_effective_klein_lora_strength_photoreal_passthrough_none(app):
    from app.services import face_dataset_service as fds
    from app.config import LOCAL_USER
    with app.app_context():
        ds = fds.create_dataset(LOCAL_USER, 'RS2', 'zchar_rs2')
        assert ds.render_style is None
        assert fds._effective_klein_lora_strength(ds, None) is None


def test_effective_klein_lora_strength_explicit_value_always_wins(app):
    from app.services import face_dataset_service as fds
    from app.config import LOCAL_USER
    with app.app_context():
        ds = fds.create_dataset(LOCAL_USER, 'RS3', 'zchar_rs3')
        ds.render_style = 'anime_2d'
        assert fds._effective_klein_lora_strength(ds, 0.7) == 0.7
        assert fds._effective_klein_lora_strength(ds, 0.0) == 0.0
        ds.render_style = None
        assert fds._effective_klein_lora_strength(ds, 0.3) == 0.3


# --- update_dataset_settings persistence -------------------------------------

def test_update_dataset_settings_persists_render_style(app):
    from app.services import face_dataset_service as fds
    from app.config import LOCAL_USER
    with app.app_context():
        ds = fds.create_dataset(LOCAL_USER, 'RS4', 'zchar_rs4')
        res = fds.update_dataset_settings(LOCAL_USER, ds.id, render_style='anime_2d')
        assert res['ok'] is True
        assert ds.render_style == 'anime_2d'


def test_update_dataset_settings_unknown_render_style_normalizes_not_400(app):
    from app.services import face_dataset_service as fds
    from app.config import LOCAL_USER
    with app.app_context():
        ds = fds.create_dataset(LOCAL_USER, 'RS5', 'zchar_rs5')
        res = fds.update_dataset_settings(LOCAL_USER, ds.id, render_style='not-a-real-style')
        assert res['ok'] is True
        assert ds.render_style is None   # unknown -> None (photoreal), never a 400


def test_update_dataset_settings_none_leaves_render_style_untouched(app):
    from app.services import face_dataset_service as fds
    from app.config import LOCAL_USER
    with app.app_context():
        ds = fds.create_dataset(LOCAL_USER, 'RS6', 'zchar_rs6', render_style='render_3d')
        fds.update_dataset_settings(LOCAL_USER, ds.id, name='RS6 renamed')
        assert ds.render_style == 'render_3d'


def test_create_dataset_accepts_render_style(app):
    from app.services import face_dataset_service as fds
    from app.config import LOCAL_USER
    with app.app_context():
        ds = fds.create_dataset(LOCAL_USER, 'RS7', 'zchar_rs7', render_style='illustration')
        assert ds.render_style == 'illustration'
        ds2 = fds.create_dataset(LOCAL_USER, 'RS8', 'zchar_rs8', render_style='photoreal')
        assert ds2.render_style is None
