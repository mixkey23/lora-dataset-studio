"""Nano Banana (Gemini image API) variation generator for the face Dataset Maker.

Sends the reference photo + a variation prompt to the Gemini image model and
returns the generated image bytes. No GPU, no ComfyUI involvement — runs fully
off-device, so dataset generation can happen while local generations run.
SFW only by provider policy (fits the face-dataset use case by design).

CHOOSING THE MODEL
------------------
`engines.nanobanana_model` is free text (Settings ▸ Image engines): Google ships
image models faster than this app ships releases, and a dropdown frozen into a
build would be stale the day it lands. Resolution order, read at CALL time so a
change in Settings applies without a restart:

    engines.nanobanana_model  >  NANOBANANA_MODEL (env)  >  DEFAULT_MODEL

The environment variable is deliberately still honoured, and above the built-in
default: it existed before the setting did and some installs set it. It is only
overridden when the user actually types a slug in Settings — i.e. by an explicit,
more recent choice. This is why the config default is BLANK rather than a copy of
DEFAULT_MODEL (see config.DEFAULTS['engines']).

FAILING LOUDLY
--------------
Failures raise a NAMED cause (see engine_errors) instead of returning None, and
the ones that would refuse every remaining row identically — rejected key,
unknown model, a model that cannot take reference images — are FATAL and stop the
batch. That matters most for the model field: a text-only or non-existent slug
used to surface as "empty response (often a content-policy refusal)", sending the
user to rewrite a prompt when the fix was one word in Settings.

None keeps exactly one meaning: Gemini answered 200 with no image (safety block
or a text-only answer). The API key never appears in a message or a log line.
"""
from __future__ import annotations
import base64
import logging
import os

import requests

from .. import config as cfg
from .engine_errors import EngineError, EngineFatal

logger = logging.getLogger(__name__)

# Nano Banana Pro (GA) — best-in-class face consistency. This is the FALLBACK,
# not a lock: see get_model() for the resolution order.
DEFAULT_MODEL = 'gemini-3-pro-image'
_ENV_VAR = 'NANOBANANA_MODEL'
_API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_NO_KEY = ('no Gemini API key saved — add GEMINI_API_KEY in '
           'Settings > Image engines')

# Fragments Gemini uses when the request is wrong ABOUT THE MODEL rather than
# about this particular prompt: an unknown slug, a model that cannot answer with
# an image, or one that refuses image inputs (the dataset generator always sends
# reference images, so a text-only model can never work here). Matching on the
# provider's own words keeps a genuine per-prompt 400 from cancelling a batch.
_MODEL_FAULT_HINTS = (
    'modalit',              # "does not support the requested response modalities: image"
    'image input',
    'input image',
    'inline_data', 'inlinedata',
    'is not found', 'not found', 'not supported', 'unsupported',
    'does not support', 'is not supported',
)


class NanoBananaError(EngineError):
    """A named Nano Banana failure. User-facing text, never carries the key."""


class NanoBananaFatal(NanoBananaError, EngineFatal):
    """A failure that would repeat on every remaining row (rejected key, unknown
    model, a model that cannot take reference images)."""


def _api_key():
    return cfg.secret('GEMINI_API_KEY')


def get_model() -> str:
    """The model this engine will ask for: setting > env var > built-in default.

    Read fresh on every call — a slug typed in Settings must apply to the very
    next generation, with no restart."""
    return ((cfg.get('engines.nanobanana_model') or '').strip()
            or (os.environ.get(_ENV_VAR) or '').strip()
            or DEFAULT_MODEL)


def _error_message(resp) -> str:
    """Gemini's own explanation, trimmed. Its documented envelope is
    {"error": {"code", "message", "status"}}; an edge failure answers something
    else entirely, so fall back to the raw text. Handing back the provider's
    exact words is the point — 'request failed' leaves nothing to act on."""
    try:
        body = resp.json()
    except Exception:                                  # noqa: BLE001 — non-JSON edge error
        body = None
    if isinstance(body, dict):
        err = body.get('error')
        if isinstance(err, dict):
            msg = str(err.get('message') or '').strip()
            if msg:
                return msg[:300]
        elif isinstance(err, str) and err.strip():
            return err.strip()[:300]
    try:
        return (resp.text or '').strip()[:300]
    except Exception:                                  # noqa: BLE001
        return ''


def _raise_for_status(resp, *, model: str) -> None:
    """Turn a non-200 into the most specific exception we can justify.

    401/403 (the key) and 404 (the slug) would fail every other row the same way,
    so they are FATAL. 429 and 5xx are transient and stay per-row, so one bad
    minute never cancels a run that would have finished. A 400 is the interesting
    one: it is fatal only when Gemini's message blames the MODEL — a request the
    user can never fix by retrying — and stays per-row otherwise."""
    status = resp.status_code
    if status == 200:
        return
    detail = _error_message(resp)
    suffix = f': {detail}' if detail else ''
    if status in (401, 403):
        raise NanoBananaFatal(f'Gemini rejected the API key (HTTP {status}){suffix}')
    if status == 404:
        raise NanoBananaFatal(
            f'Gemini does not serve the model "{model}" (HTTP 404){suffix} — '
            'check the model in Settings > Image engines')
    if status == 429:
        raise NanoBananaError(f'Gemini rate-limited the request (HTTP 429){suffix}')
    if status == 400 and any(h in detail.lower() for h in _MODEL_FAULT_HINTS):
        raise NanoBananaFatal(
            f'Gemini refused the request for model "{model}" (HTTP 400){suffix} — '
            'this engine always sends your reference images with the prompt, so '
            'the model must be an IMAGE model that accepts image input; check the '
            'model in Settings > Image engines')
    raise NanoBananaError(f'Gemini returned HTTP {status}{suffix}')


def parse_image_response(data) -> bytes | None:
    """Extract the first inline image from a generateContent response."""
    try:
        for cand in data.get('candidates', []):
            for part in (cand.get('content') or {}).get('parts', []):
                inline = part.get('inlineData') or part.get('inline_data') or {}
                if inline.get('data'):
                    return base64.b64decode(inline['data'])
    except (TypeError, ValueError, KeyError):
        return None
    return None


def generate_variation(ref_bytes: bytes | list[bytes], prompt: str, model: str | None = None,
                       aspect_ratio: str = '1:1') -> bytes | None:
    """Reference photo(s) + variation prompt -> generated image bytes, or None.

    `ref_bytes` : une image (bytes) ou une LISTE d'images de la même personne
    (multi-références — gemini-3-pro-image accepte jusqu'à 14 images d'entrée et
    s'appuie sur toutes pour la cohérence d'identité). La principale en premier.
    `aspect_ratio` (ex. '1:1' visage, '3:4' buste/corps) évite de letterboxer les
    plans corps. Tries with imageConfig first (Pro models); on a 400 retries once
    with a slim payload for models that don't accept imageConfig.

    None means one thing only: Gemini answered 200 without an image (safety block
    or text-only answer). Everything else raises with the cause named."""
    key = _api_key()
    if not key:
        # An exception, not None: a missing key must never read to the user as
        # "the provider refused your prompt".
        raise NanoBananaFatal(_NO_KEY)
    mdl = (model or '').strip() or get_model()
    refs = ref_bytes if isinstance(ref_bytes, (list, tuple)) else [ref_bytes]
    parts = [{"text": prompt}]
    for rb in refs:
        parts.append({"inlineData": {"mimeType": "image/webp",
                                     "data": base64.b64encode(rb).decode('ascii')}})
    payloads = [
        {"contents": [{"parts": parts}],
         "generationConfig": {"responseModalities": ["TEXT", "IMAGE"],
                              "imageConfig": {"aspectRatio": aspect_ratio}}},
        {"contents": [{"parts": parts}],
         "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}},
    ]
    for i, payload in enumerate(payloads):
        try:
            r = requests.post(_API.format(model=mdl),
                              headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                              json=payload, timeout=(10, 180))
        except requests.RequestException as e:
            raise NanoBananaError(f'could not reach Gemini: {e}')
        if r.status_code == 400 and i == 0:
            continue  # retry without imageConfig
        _raise_for_status(r, model=mdl)
        try:
            data = r.json()
        except ValueError as e:
            raise NanoBananaError(f'Gemini returned a non-JSON response: {e}')
        img = parse_image_response(data)
        if img is None:
            logger.warning("nanobanana: no image in response from %s "
                           "(safety block or text-only)", mdl)
        return img
    return None
