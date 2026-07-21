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
from . import qwen_edit_assets as qea
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

# Multi-angle LoRA: the one asset that stays specific to THIS engine (no role
# outside camera rotation) — everything else (UNET/VAE/TE/consistency/
# Lightning) is resolved from the SHARED qwen_edit_assets module (Wave 4) so
# this engine and the general-purpose "Generate variations" Qwen Edit engine
# resolve the identical files, never asking the user to configure them twice.
_CANONICAL_MULTIANGLE_LORA = 'qwen-image-edit-2511-multiple-angles-lora.safetensors'


class QwenMultiangleModelsMissing(Exception):
    """A graph-critical asset (UNET / text-encoder / VAE / multi-angle LoRA) is
    not on disk, so a valid job can't be built. `.missing` lists every absent
    REQUIRED+RECOMMENDED asset so the caller can show one actionable message."""
    def __init__(self, missing):
        self.missing = list(missing)
        super().__init__('Qwen multi-angle models missing: ' + ', '.join(self.missing))


# --- Asset resolution: thin delegators to the shared qwen_edit_assets module
# (Wave 4) for everything except the multi-angle LoRA, which stays local since
# it has no purpose outside this engine. Public names kept EXACTLY as before
# (capabilities.py and test_qwen_multiangle.py reference them) — behavior is
# unchanged, only the implementation moved.

def resolve_qwen_ma_unet(selected=None):
    """`unet_name` for node 108."""
    return qea.resolve_unet(selected)


def resolve_qwen_ma_vae():
    """`vae_name` for node 95."""
    return qea.resolve_vae()


def resolve_qwen_ma_text_encoder():
    """`clip_name` for node 93."""
    return qea.resolve_text_encoder()


def resolve_qwen_ma_multiangle_lora():
    return qea._resolve_lora(_CANONICAL_MULTIANGLE_LORA, ('multiple-angles', 'multiple_angles'))


def resolve_qwen_ma_consistency_lora():
    return qea.resolve_consistency_lora()


def resolve_qwen_ma_lightning_lora():
    return qea.resolve_lightning_lora()


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
# `too_small` floor. The 4 shared assets delegate to qea.QWEN_EDIT_MIN_BYTES;
# only the multi-angle LoRA's floor is local.
QWEN_MA_MIN_BYTES = {
    'qwen_ma_unet': qea.QWEN_EDIT_MIN_BYTES['unet'],
    'qwen_ma_text_encoder': qea.QWEN_EDIT_MIN_BYTES['text_encoder'],
    'qwen_ma_vae': qea.QWEN_EDIT_MIN_BYTES['vae'],
    'qwen_ma_multiangle_lora': 512 * 1024,    # 512 KB
    'qwen_ma_consistency_lora': qea.QWEN_EDIT_MIN_BYTES['consistency_lora'],
    'qwen_ma_lightning_lora': qea.QWEN_EDIT_MIN_BYTES['lightning_lora'],
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
