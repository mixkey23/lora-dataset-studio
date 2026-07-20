"""Focused helper to enqueue a SINGLE Qwen-Image multi-angle edit job.

Rotates the camera angle of an existing render using Qwen-Image-Edit-2511 plus
the community 'fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA' — a GENERATION
feature, not a training one (see the 'qwen_image' training family in
lora_training.py for the unrelated LoRA-training path). Modeled directly on
klein_edit_helper.py: same asset-resolution pattern (canonical filename first,
narrow token fallback), same degrade-not-fail LoRA injection, same job-queue
handoff. The shipped workflow (qwen_multiangle.json) is adapted from a real,
working ComfyUI export: its API-oriented I/O nodes (base64 image in, cached
image out) were swapped for the standard LoadImage/SaveImage pair this app's
job_queue already knows how to drive, and the fixed 1664x928 empty latent was
swapped for a GetImageSize-driven size matching the actual source image.
"""
from __future__ import annotations
import logging
import os
import random
import shutil
import time
import uuid

from .. import config as cfg
from . import comfy_model_paths
from ..utils.comfyui import load_workflow_local
from ..job_queue import queue_manager

logger = logging.getLogger(__name__)

WORKFLOW_QWEN_MULTIANGLE_PATH = cfg.BACKEND_DIR / 'workflows' / 'qwen_multiangle.json'

# Nodes this helper rewires — fail LOUDLY if the workflow file changes shape
# instead of silently enqueuing a job with the wrong source/prompt/model.
_REQUIRED_NODES = ('115', '112', '106', '121', '108', '95', '93', '109', '114', '102', '122', '123')

# The engine's model dependencies. REQUIRED = the graph is invalid without it
# (block + tell the user what's missing); the multi-angle LoRA is REQUIRED
# here (unlike Klein's optional consistency LoRA) — without it this engine has
# no purpose. RECOMMENDED = quality/speed only, degrade gracefully.
QWEN_MA_REQUIRED = ('qwen_ma_unet', 'qwen_ma_text_encoder', 'qwen_ma_vae', 'qwen_ma_multiangle_lora')
QWEN_MA_RECOMMENDED = ('qwen_ma_consistency_lora', 'qwen_ma_lightning_lora')

_MODEL_SUFFIXES = ('.safetensors', '.gguf', '.sft')

# Canonical filenames, from the real workflow this engine was built against.
# Matching is canonical-first with NARROW token fallbacks (mirrors Klein's
# _find_model_file/_klein_unet_folders reasoning): a loose 'qwen' match would
# just as easily grab Z-Image's qwen3vl_* or the base (non-Edit) Qwen-Image
# text encoder/UNET sitting in the same shared folder.
_CANONICAL_UNET = 'Qwen-Image-Edit-2511-FP8_e4m3fn.safetensors'
_CANONICAL_VAE = 'qwen_image_vae.safetensors'
_CANONICAL_TEXT_ENCODER = 'qwen_2.5_vl_7b_fp8_scaled.safetensors'
_CANONICAL_MULTIANGLE_LORA = 'qwen-image-edit-2511-multiple-angles-lora.safetensors'
_CANONICAL_CONSISTENCY_LORA = 'consistence_edit_v2.safetensors'
_CANONICAL_LIGHTNING_LORA = 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors'

_UNET_TOKENS = ('qwen-image-edit-2511', 'qwen_image_edit_2511', 'qwen-image-edit', 'qwen_image_edit')
_VAE_TOKENS = ('qwen_image_vae', 'qwen-image-vae')
_TEXT_ENCODER_TOKENS = ('qwen_2.5_vl', 'qwen2.5_vl', 'qwen_2_5_vl')


class QwenMultiangleModelsMissing(Exception):
    """A graph-critical asset (UNET / text-encoder / VAE / multi-angle LoRA) is
    not on disk, so a valid job can't be built. `.missing` lists every absent
    REQUIRED+RECOMMENDED asset so the caller can show one actionable message."""
    def __init__(self, missing):
        self.missing = list(missing)
        super().__init__('Qwen multi-angle models missing: ' + ', '.join(self.missing))


_FOLDER_DISCOVERY_TOKENS = ('qwen',)   # broad: real installs just use one 'QwenImage' folder


def _model_folders(comfy_type):
    """(prefix, [model files]) candidates for a model of `comfy_type` across every
    search root: every subfolder whose NAME contains a broad 'qwen' token (real
    installs use one shared 'QwenImage' folder for base + Edit + everything else,
    unlike Klein's own dedicated 'klein' folder), plus the root's own top-level
    files unfiltered — callers apply their own NARROW file-level token filter to
    tell apart what's actually inside (e.g. Qwen-Image-Edit-2511 vs. the base
    Qwen-Image checkpoint, or the multi-angle LoRA vs. an unrelated one)."""
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
    nothing matches. Returns the value WITH its subfolder prefix (e.g.
    'QwenImage\\Qwen-Image-Edit-2511-FP8_e4m3fn.safetensors'). Preference: the
    caller's choice, then the canonical filename, then a NARROW file-name token
    match — never a blind first-file guess, since the folder name alone can't
    discriminate a shared 'QwenImage' folder the way Klein's dedicated folder does."""
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


def resolve_qwen_ma_unet(selected=None):
    """`unet_name` for node 108."""
    return _resolve_model('diffusion_models', _CANONICAL_UNET, _UNET_TOKENS, selected)


def resolve_qwen_ma_vae():
    """`vae_name` for node 95."""
    return _resolve_model('vae', _CANONICAL_VAE, _VAE_TOKENS)


def resolve_qwen_ma_text_encoder():
    """`clip_name` for node 93. NEVER a bare 'qwen' match — the base (non-Edit)
    Qwen-Image text encoder and Z-Image's qwen3vl_* live in the same folder."""
    return _resolve_model('text_encoders', _CANONICAL_TEXT_ENCODER, _TEXT_ENCODER_TOKENS)


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
    UNET/VAE/text-encoder above (canonical-first, narrow file-token fallback)
    rather than a free-text config path — none of these files are
    auto-downloaded by this app, so a stale config value must not silently
    point at nothing."""
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


def resolve_qwen_ma_multiangle_lora():
    return _resolve_lora(_CANONICAL_MULTIANGLE_LORA, ('multiple-angles', 'multiple_angles'))


def resolve_qwen_ma_consistency_lora():
    return _resolve_lora(_CANONICAL_CONSISTENCY_LORA, ('consistence_edit', 'consistency_edit'))


def resolve_qwen_ma_lightning_lora():
    return _resolve_lora(_CANONICAL_LIGHTNING_LORA, ('lightning',))


def qwen_ma_missing_assets():
    """Which qwen_multiangle assets are NOT on disk, as short action names (a
    subset of QWEN_MA_REQUIRED + QWEN_MA_RECOMMENDED). Mirrors
    klein_edit_helper.klein_missing_assets()."""
    missing = []
    if not resolve_qwen_ma_unet():
        missing.append('qwen_ma_unet')
    if not resolve_qwen_ma_text_encoder():
        missing.append('qwen_ma_text_encoder')
    if not resolve_qwen_ma_vae():
        missing.append('qwen_ma_vae')
    _, lora_path = resolve_qwen_ma_multiangle_lora()
    if not (lora_path and os.path.exists(lora_path)):
        missing.append('qwen_ma_multiangle_lora')
    _, cons_path = resolve_qwen_ma_consistency_lora()
    if not (cons_path and os.path.exists(cons_path)):
        missing.append('qwen_ma_consistency_lora')
    _, light_path = resolve_qwen_ma_lightning_lora()
    if not (light_path and os.path.exists(light_path)):
        missing.append('qwen_ma_lightning_lora')
    return missing


# Minimum plausible on-disk size per asset, for model_integrity's advisory
# `too_small` floor — deliberately conservative (well under the real size) so
# it never cries wolf on a legitimate file. Mirrors klein_edit_helper.KLEIN_MIN_BYTES.
QWEN_MA_MIN_BYTES = {
    'qwen_ma_unet': 1024 ** 3,               # 1 GB   (fp8 20B DiT is much larger)
    'qwen_ma_text_encoder': 512 * 1024 ** 2,  # 512 MB (Qwen2.5-VL-7B fp8 is several GB)
    'qwen_ma_vae': 8 * 1024 ** 2,             # 8 MB
    'qwen_ma_multiangle_lora': 512 * 1024,    # 512 KB
    'qwen_ma_consistency_lora': 512 * 1024,
    'qwen_ma_lightning_lora': 512 * 1024,
}


def _abs_under_roots(comfy_type, rel_name):
    if not rel_name:
        return None
    for root in comfy_model_paths.search_roots(comfy_type):
        cand = os.path.join(root, rel_name)
        if os.path.exists(cand):
            return cand
    return None


def _qwen_ma_asset_paths():
    """{asset name: absolute path} for each asset PRESENT on disk."""
    paths = {}
    unet = resolve_qwen_ma_unet()
    if unet:
        p = _abs_under_roots('diffusion_models', unet)
        if p:
            paths['qwen_ma_unet'] = p
    te = resolve_qwen_ma_text_encoder()
    if te:
        p = _abs_under_roots('text_encoders', te)
        if p:
            paths['qwen_ma_text_encoder'] = p
    vae = resolve_qwen_ma_vae()
    if vae:
        p = _abs_under_roots('vae', vae)
        if p:
            paths['qwen_ma_vae'] = p
    _, ma_path = resolve_qwen_ma_multiangle_lora()
    if ma_path and os.path.exists(ma_path):
        paths['qwen_ma_multiangle_lora'] = ma_path
    _, cons_path = resolve_qwen_ma_consistency_lora()
    if cons_path and os.path.exists(cons_path):
        paths['qwen_ma_consistency_lora'] = cons_path
    _, light_path = resolve_qwen_ma_lightning_lora()
    if light_path and os.path.exists(light_path):
        paths['qwen_ma_lightning_lora'] = light_path
    return paths


def qwen_ma_invalid_assets():
    """Assets that ARE on disk under the resolved name but are NOT real,
    loadable weights (e.g. a licence-gate HTML page saved as .safetensors).
    Mirrors klein_edit_helper.klein_invalid_assets()."""
    from . import model_integrity
    out = []
    for asset, path in _qwen_ma_asset_paths().items():
        res = model_integrity.validate_model_file(path, min_bytes=QWEN_MA_MIN_BYTES.get(asset))
        if res['ok']:
            continue
        out.append({'asset': asset, 'filename': res['filename'],
                    'verdict': res['verdict'], 'blocking': res['blocking'],
                    'reason': res['reason']})
    return out


# --- Custom-node preflight -------------------------------------------------
# The class_types this workflow needs that are NOT core ComfyUI / comfy_extras —
# mirrors klein_edit_helper.KLEIN_NODE_PACKS. Unknown pack/url (None, None) is
# the generic fallback for any other custom node a future workflow change adds.
QWEN_MA_NODE_PACKS = {
    'TextEncodeQwenImageEditPlus': (None, None),
    'FluxKontextMultiReferenceLatentMethod': (None, None),
    'FluxKontextImageScale': (None, None),
    'CFGNorm': (None, None),
    'ModelSamplingAuraFlow': (None, None),
}

_NODES_OK_TTL_S = 300
_nodes_ok_until = 0.0


def _workflow_class_types(workflow):
    return {n.get('class_type') for n in (workflow or {}).values()
            if isinstance(n, dict) and n.get('class_type')}


def qwen_ma_missing_nodes(workflow=None):
    """[{class_type, pack, url}] for every node class this workflow needs that
    the target ComfyUI does NOT expose. FAIL-OPEN on a probe failure — mirrors
    klein_edit_helper.klein_missing_nodes()."""
    global _nodes_ok_until
    shipped = workflow is None
    if shipped:
        if time.time() < _nodes_ok_until:
            return []
        workflow = load_workflow_local(str(WORKFLOW_QWEN_MULTIANGLE_PATH)) or {}
    from ..utils.comfyui import fetch_object_info_classes
    available = fetch_object_info_classes()
    if available is None:
        return []
    out = []
    for ct in sorted(_workflow_class_types(workflow) - available):
        pack, url = QWEN_MA_NODE_PACKS.get(ct, (None, None))
        out.append({'class_type': ct, 'pack': pack, 'url': url})
    if shipped and not out:
        _nodes_ok_until = time.time() + _NODES_OK_TTL_S
    return out


def format_missing_nodes_message(missing_nodes):
    """Human sentence for a qwen-multiangle node-missing 409 — mirrors
    klein_edit_helper.format_missing_nodes_message()."""
    bits = []
    for n in missing_nodes:
        ct, pack, url = n.get('class_type'), n.get('pack'), n.get('url')
        if pack and url:
            bits.append(f"{ct} (from {pack}: {url})")
        elif pack:
            bits.append(f"{ct} (from {pack})")
        else:
            bits.append(str(ct))
    return ("Your ComfyUI is missing custom node(s) this Qwen Multi-angle workflow "
            "needs: " + '; '.join(bits) + ". Install via ComfyUI-Manager, then restart ComfyUI.")


def _bypass_node(workflow, node_id, passthrough_input):
    """Delete node_id and reconnect every consumer of its output slot 0 to the
    node's own `passthrough_input` upstream — mirrors
    klein_edit_helper._bypass_node()."""
    node = workflow.get(node_id)
    if not node:
        return
    upstream = node.get('inputs', {}).get(passthrough_input)
    if upstream is None:
        return
    for other in workflow.values():
        for k, v in list(other.get('inputs', {}).items()):
            if isinstance(v, list) and len(v) == 2 and v[0] == node_id:
                other['inputs'][k] = upstream
    workflow.pop(node_id, None)


def _comfy_input_dir() -> str:
    d = cfg.comfyui_dir('input')
    if not d:
        raise RuntimeError('ComfyUI is not configured')
    return str(d)


def _comfy_output_dir():
    d = cfg.comfyui_dir('output')
    return str(d) if d else None


# Fixed vocabulary from the multi-angle LoRA's model card — the trigger format
# is `<sks> [azimuth] [elevation] [distance]`, that exact order, `<sks>`
# mandatory. Byte-exact with what the LoRA was trained on: an unknown token
# here is a silent no-op at generation time, not a loud error, so this stays
# strict rather than accepting free text.
QWEN_MA_AZIMUTHS = ('front view', 'front-right quarter view', 'right side view',
                    'back-right quarter view', 'back view', 'back-left quarter view',
                    'left side view', 'front-left quarter view')
QWEN_MA_ELEVATIONS = ('eye-level shot', 'high-angle shot', 'low-angle shot', 'elevated shot')
QWEN_MA_DISTANCES = ('close-up', 'medium shot', 'wide shot')


def build_angle_prompt(azimuth, elevation, distance) -> str:
    """The exact `<sks> {azimuth} {elevation} {distance}` trigger string the
    multi-angle LoRA was trained on. Raises ValueError on any token outside the
    LoRA's trained vocabulary."""
    if azimuth not in QWEN_MA_AZIMUTHS:
        raise ValueError(f"unknown azimuth: {azimuth!r}")
    if elevation not in QWEN_MA_ELEVATIONS:
        raise ValueError(f"unknown elevation: {elevation!r}")
    if distance not in QWEN_MA_DISTANCES:
        raise ValueError(f"unknown distance: {distance!r}")
    return f"<sks> {azimuth} {elevation} {distance}"


def enqueue_qwen_multiangle(user_id, source_filename, azimuth, elevation, distance,
                            source_path=None, qwen_model=None,
                            multiangle_strength=None, consistency_strength=None,
                            lightning_enabled=True, lightning_strength=None,
                            extra_metadata=None):
    """Copy the source into ComfyUI input, configure the qwen_multiangle
    workflow for the requested camera angle, and enqueue it. Returns the app
    job_id. Raises ValueError on a missing source / unloadable workflow /
    missing required node / unknown angle vocabulary, RuntimeError if ComfyUI
    isn't configured, QwenMultiangleModelsMissing if a graph-critical asset is
    absent. `lightning_enabled` toggles the Lightning speed LoRA (node 102);
    when off, sampling falls back to a non-distilled step/cfg envelope (both
    are UNTESTED extrapolations — no research vault entry for this engine yet)."""
    if source_path is None:
        out_dir = _comfy_output_dir()
        if out_dir is None:
            raise RuntimeError('ComfyUI is not configured')
        source_path = os.path.join(out_dir, source_filename)
    if not os.path.exists(source_path):
        raise ValueError(f"source image not found: {source_filename}")
    prompt = build_angle_prompt(azimuth, elevation, distance)

    workflow = load_workflow_local(str(WORKFLOW_QWEN_MULTIANGLE_PATH))
    if not workflow:
        raise ValueError("failed to load Qwen multi-angle workflow")
    for node in _REQUIRED_NODES:
        if node not in workflow:
            raise ValueError(f"workflow node {node} missing — qwen_multiangle.json has changed")

    unet_ref = resolve_qwen_ma_unet(qwen_model)
    vae_ref = resolve_qwen_ma_vae()
    te_ref = resolve_qwen_ma_text_encoder()
    ma_lora, ma_path = resolve_qwen_ma_multiangle_lora()
    missing = qwen_ma_missing_assets()
    if any(a in missing for a in QWEN_MA_REQUIRED):
        raise QwenMultiangleModelsMissing(missing)

    comfy_input_dir = _comfy_input_dir()
    uid = uuid.uuid4().hex[:8]
    comfy_input = f"qwen_ma_source_{uid}_{source_filename}"
    shutil.copy2(source_path, os.path.join(comfy_input_dir, comfy_input))

    workflow["115"]["inputs"]["image"] = comfy_input
    workflow["112"]["inputs"]["prompt"] = prompt
    workflow["108"]["inputs"]["unet_name"] = unet_ref
    workflow["95"]["inputs"]["vae_name"] = vae_ref
    workflow["93"]["inputs"]["clip_name"] = te_ref
    workflow["106"]["inputs"]["seed"] = random.randint(0, 2 ** 64 - 1)
    # UNIQUE prefix per job — mirrors klein_edit_helper's ComfyUI-output-counter
    # note (a shared prefix made every tile display the same file).
    workflow["121"]["inputs"]["filename_prefix"] = f"{user_id}_QwenMultiangle_{uid}"

    # Multi-angle LoRA (required — already verified present above).
    ma_strength = cfg.get('qwen_multiangle.multiangle_strength', 1.0)
    if multiangle_strength is not None:
        ma_strength = max(0.0, min(1.5, float(multiangle_strength)))
    workflow["109"]["inputs"]["lora_name"] = ma_lora
    workflow["109"]["inputs"]["strength_model"] = ma_strength

    # Consistency LoRA (optional, node 114) — degrade by bypassing, never fail.
    cons_lora, cons_path = resolve_qwen_ma_consistency_lora()
    cons_strength = cfg.get('qwen_multiangle.consistency_strength', 1.0)
    if consistency_strength is not None:
        cons_strength = max(0.0, min(1.5, float(consistency_strength)))
    if not cons_path or not os.path.exists(cons_path):
        logger.warning(f"consistency LoRA not found at {cons_path} — bypassing node 114")
        _bypass_node(workflow, "114", "model")
    elif not cons_strength or float(cons_strength) <= 0:
        logger.info("consistency LoRA strength 0 — bypassing node 114 (LoRA off)")
        _bypass_node(workflow, "114", "model")
    else:
        workflow["114"]["inputs"]["lora_name"] = cons_lora
        workflow["114"]["inputs"]["strength_model"] = cons_strength

    # Lightning speed LoRA (optional, node 102) — bypass + fall back to a
    # non-distilled step/cfg envelope when off or the file is absent.
    light_lora, light_path = resolve_qwen_ma_lightning_lora()
    light_strength = cfg.get('qwen_multiangle.lightning_strength', 1.0)
    if lightning_strength is not None:
        light_strength = max(0.0, min(1.5, float(lightning_strength)))
    use_lightning = (lightning_enabled and light_path and os.path.exists(light_path)
                     and light_strength and float(light_strength) > 0)
    if "102" in workflow:
        if use_lightning:
            workflow["102"]["inputs"]["lora_name"] = light_lora
            workflow["102"]["inputs"]["strength_model"] = light_strength
        else:
            if lightning_enabled and not (light_path and os.path.exists(light_path)):
                logger.warning(f"Lightning LoRA not found at {light_path} — bypassing node 102")
            _bypass_node(workflow, "102", "model")
    if "106" in workflow:
        # Lightning's 4-step/cfg-1 envelope only applies while its LoRA is
        # active; off, fall back to a real-CFG envelope (EXTRAPOLATED —
        # untested, mirrors the flux2klein non-distilled default).
        workflow["106"]["inputs"]["steps"] = 4 if use_lightning else 20
        workflow["106"]["inputs"]["cfg"] = 1 if use_lightning else 4

    job_id = str(uuid.uuid4())
    meta = {"model_name": "qwen_multiangle_dataset"}
    if extra_metadata:
        meta.update(extra_metadata)
    queue_manager.add_job(job_type="image", user_id=str(user_id), workflow_data=workflow,
                          prompt=prompt, job_id=job_id, metadata=meta)
    return job_id
