"""Shared Qwen-Image-Edit-2511 asset resolution (Wave 4). Extracted from
qwen_multiangle_helper.py (Wave 1) so the general-purpose "Generate
variations" engine (qwen_edit_helper.py) and the narrow camera-rotation
engine (qwen_multiangle_helper.py) resolve the IDENTICAL checkpoint/LoRA
files by canonical filename scan of ComfyUI's model folders — no config
duplication, the user never enters the same DiT/VAE/TE path twice.

Same resolution contract as klein_edit_helper.py: canonical filename first,
narrow file-name token fallback, NEVER a blind first-file guess (a shared
'QwenImage' folder holds base + Edit + everything else, unlike Klein's own
dedicated folder). Pure mechanics — no DB, no Flask beyond config reads.
"""
from __future__ import annotations
import os

from .. import config as cfg
from . import comfy_model_paths

_MODEL_SUFFIXES = ('.safetensors', '.gguf', '.sft')

# Canonical filenames, from the real workflow this engine was built against.
_CANONICAL_UNET = 'Qwen-Image-Edit-2511-FP8_e4m3fn.safetensors'
_CANONICAL_VAE = 'qwen_image_vae.safetensors'
_CANONICAL_TEXT_ENCODER = 'qwen_2.5_vl_7b_fp8_scaled.safetensors'
_CANONICAL_CONSISTENCY_LORA = 'consistence_edit_v2.safetensors'
_CANONICAL_LIGHTNING_LORA = 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors'

_UNET_TOKENS = ('qwen-image-edit-2511', 'qwen_image_edit_2511', 'qwen-image-edit', 'qwen_image_edit')
_VAE_TOKENS = ('qwen_image_vae', 'qwen-image-vae')
_TEXT_ENCODER_TOKENS = ('qwen_2.5_vl', 'qwen2.5_vl', 'qwen_2_5_vl')

_FOLDER_DISCOVERY_TOKENS = ('qwen',)   # broad: real installs just use one 'QwenImage' folder

# Minimum plausible on-disk size per asset, for model_integrity's advisory
# `too_small` floor — deliberately conservative. Mirrors klein_edit_helper.KLEIN_MIN_BYTES.
QWEN_EDIT_MIN_BYTES = {
    'unet': 1024 ** 3,               # 1 GB   (fp8 20B DiT is much larger)
    'text_encoder': 512 * 1024 ** 2,  # 512 MB (Qwen2.5-VL-7B fp8 is several GB)
    'vae': 8 * 1024 ** 2,             # 8 MB
    'consistency_lora': 512 * 1024,
    'lightning_lora': 512 * 1024,
}


def _model_folders(comfy_type):
    """(prefix, [model files]) candidates for a model of `comfy_type` across every
    search root: every subfolder whose NAME contains a broad 'qwen' token, plus
    the root's own top-level files unfiltered — callers apply their own NARROW
    file-level token filter to tell apart what's actually inside."""
    out = []
    for base_dir in comfy_model_paths.search_roots(comfy_type):
        try:
            entries = os.listdir(base_dir)
        except OSError:
            continue
        subs = sorted(d for d in entries
                      if any(tok in d.lower() for tok in _FOLDER_DISCOVERY_TOKENS)
                      and os.path.isdir(os.path.join(base_dir, d)))
        for sub in subs:
            try:
                names = sorted(n for n in os.listdir(os.path.join(base_dir, sub))
                               if n.lower().endswith(_MODEL_SUFFIXES))
            except OSError:
                continue
            if names:
                out.append((sub, names))
        root_names = sorted(n for n in entries
                            if n.lower().endswith(_MODEL_SUFFIXES)
                            and os.path.isfile(os.path.join(base_dir, n)))
        if root_names:
            out.append(('', root_names))
    return out


def _resolve_model(comfy_type, canonical, file_tokens, selected=None):
    """ComfyUI-relative loader value for a model of `comfy_type`, or None if
    nothing matches. Preference: the caller's choice, then the canonical
    filename, then a NARROW file-name token match — never a blind first-file
    guess, since the folder name alone can't discriminate a shared folder."""
    folders = _model_folders(comfy_type)
    if not folders:
        return None
    bare_pick = os.path.basename(selected) if selected else None
    if bare_pick:
        for sub, names in folders:
            if bare_pick in names:
                return os.path.join(sub, bare_pick)
    for sub, names in folders:
        if canonical in names:
            return os.path.join(sub, canonical)
    for sub, names in folders:
        for n in names:
            if any(tok in n.lower() for tok in file_tokens):
                return os.path.join(sub, n)
    return None


def _lora_abs(rel_name):
    """Absolute path of a loras-relative name under the FIRST loras search root
    that holds it, else None (mirrors klein_edit_helper._lora_abs)."""
    if not rel_name:
        return None
    for root in comfy_model_paths.search_roots('loras'):
        cand = os.path.join(root, rel_name)
        if os.path.exists(cand):
            return cand
    return None


def _resolve_lora(canonical, file_tokens):
    """(relative_name, absolute_path) of a configurable LoRA, resolved like the
    UNET/VAE/text-encoder above — none of these files are auto-downloaded by
    this app, so a stale value must not silently point at nothing."""
    folders = _model_folders('loras')
    if not folders:
        return None, None
    for sub, names in folders:
        if canonical in names:
            rel = os.path.join(sub, canonical)
            return rel, _lora_abs(rel)
    for sub, names in folders:
        for n in names:
            if any(tok in n.lower() for tok in file_tokens):
                rel = os.path.join(sub, n)
                return rel, _lora_abs(rel)
    return None, None


def resolve_unet(selected=None):
    return _resolve_model('diffusion_models', _CANONICAL_UNET, _UNET_TOKENS, selected)


def resolve_vae():
    return _resolve_model('vae', _CANONICAL_VAE, _VAE_TOKENS)


def resolve_text_encoder():
    """NEVER a bare 'qwen' match — the base (non-Edit) Qwen-Image text encoder
    and Z-Image's qwen3vl_* live in the same folder."""
    return _resolve_model('text_encoders', _CANONICAL_TEXT_ENCODER, _TEXT_ENCODER_TOKENS)


def resolve_consistency_lora():
    return _resolve_lora(_CANONICAL_CONSISTENCY_LORA, ('consistence_edit', 'consistency_edit'))


def resolve_lightning_lora():
    return _resolve_lora(_CANONICAL_LIGHTNING_LORA, ('lightning',))


def _abs_under_roots(comfy_type, rel_name):
    if not rel_name:
        return None
    for root in comfy_model_paths.search_roots(comfy_type):
        cand = os.path.join(root, rel_name)
        if os.path.exists(cand):
            return cand
    return None


def base_asset_paths():
    """{'unet'|'text_encoder'|'vae'|'consistency_lora'|'lightning_lora': absolute
    path} for each of the 5 shared assets PRESENT on disk — everything BOTH
    engines need, minus the multi-angle LoRA (that one stays
    qwen_multiangle_helper-only, it has no role outside camera rotation)."""
    paths = {}
    unet = resolve_unet()
    if unet:
        p = _abs_under_roots('diffusion_models', unet)
        if p:
            paths['unet'] = p
    te = resolve_text_encoder()
    if te:
        p = _abs_under_roots('text_encoders', te)
        if p:
            paths['text_encoder'] = p
    vae = resolve_vae()
    if vae:
        p = _abs_under_roots('vae', vae)
        if p:
            paths['vae'] = p
    _, cons_path = resolve_consistency_lora()
    if cons_path and os.path.exists(cons_path):
        paths['consistency_lora'] = cons_path
    _, light_path = resolve_lightning_lora()
    if light_path and os.path.exists(light_path):
        paths['lightning_lora'] = light_path
    return paths
