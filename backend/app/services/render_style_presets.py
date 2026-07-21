"""Render-style presets (Wave 3): the target AESTHETIC of generated dataset
variations, orthogonal to a dataset's kind (character/concept/style) and its
training family. Pure mechanics module — mirrors musubi_tuner.py, no DB, no
Flask, trivially unit-tested.

'photoreal' is the historical default and MUST stay a no-op: every wrapper
that consults `generation_tail_for()` treats a None return as "keep your
existing hardcoded default", so a photoreal dataset produces byte-identical
prompts to before this module existed.
"""
from __future__ import annotations

RENDER_STYLES = ('photoreal', 'render_3d', 'anime_2d', 'cartoon_semireal',
                 'illustration', 'custom')

RENDER_STYLE_PRESETS = {
    'photoreal': {
        'label': 'Photoreal',
        # None = the caller keeps its own hardcoded photographic default verbatim.
        'generation_tail': None,
    },
    'render_3d': {
        'label': '3D render',
        'generation_tail': ('a stylized 3D render — CGI character shading, no photographic '
                            'grain or skin pores, clean render lighting.'),
    },
    'anime_2d': {
        'label': '2D anime',
        'generation_tail': ('a flat cel-shaded 2D anime illustration — clean line art, no '
                            'photographic realism, no skin texture or pores.'),
    },
    'cartoon_semireal': {
        'label': 'Semi-realistic cartoon',
        'generation_tail': ('a semi-realistic painted illustration — stylized proportions and '
                            'shading, not a photograph.'),
    },
    'illustration': {
        'label': 'Illustration / concept art',
        'generation_tail': ('a digital painting / concept-art illustration, visible brushwork, '
                            'not a photograph.'),
    },
    # Free text lives on the dataset's own prompt_suffix — this preset carries no
    # tail of its own, same no-op contract as photoreal.
    'custom': {'label': 'Custom', 'generation_tail': None},
}


def render_style_label(value) -> str:
    return RENDER_STYLE_PRESETS.get(value or 'photoreal', RENDER_STYLE_PRESETS['photoreal'])['label']


def generation_tail_for(render_style) -> str | None:
    """The photographic-tail replacement for this render style, or None when the
    caller should keep its own hardcoded default (photoreal, custom, or an
    unrecognized value — never guess at an unknown style's wording)."""
    return RENDER_STYLE_PRESETS.get(render_style or 'photoreal', {}).get('generation_tail')


# --- Negative prompt (Qwen Edit, Wave 4 follow-up) ----------------------------
# Klein runs at a guidance-distilled CFG=1, where the sampler ignores the
# negative conditioning entirely (see wrap_variation_klein's docstring) — a
# negative prompt would be a no-op there, so this only feeds Qwen Edit's
# workflow (a real, non-empty CLIPTextEncode negative node). Unlike
# `generation_tail`, this ALWAYS returns a non-empty string — there is no
# legacy "byte-identical when absent" contract to preserve here, a negative
# prompt is simply always useful for this engine.
_NEGATIVE_PHOTOREAL = (
    'lowres, worst quality, blurry, deformed, disfigured, bad anatomy, extra limbs, '
    'missing limbs, bad hands, mutated, cartoon, anime, illustration, painting, '
    'drawing, 3d render, cgi, watermark, text')
# Shared by every non-photoreal style: same anti-photorealism direction
# regardless of which stylized look is targeted (3D/anime/cartoon/illustration
# all want to steer AWAY from photographic rendering the same way).
_NEGATIVE_STYLIZED = (
    'lowres, worst quality, photorealistic, ultrarealistic, distorted anatomy, '
    'malformed limbs, unrealistic proportions, extra limbs, missing limbs, bad face, '
    'overprocessed skin, pixelated, compression artifact, wax figure appearance, '
    'excessive smoothing, oversharpened')


def generation_negative_for(render_style) -> str:
    """The negative prompt for this render style — anti-photorealism terms for
    a stylized target, generic anti-photo-artifact/anti-stylization terms
    otherwise (photoreal, custom, or an unrecognized value)."""
    return _NEGATIVE_STYLIZED if generation_tail_for(render_style) else _NEGATIVE_PHOTOREAL
