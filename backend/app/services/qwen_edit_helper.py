"""General-purpose "Generate variations" engine on Qwen-Image-Edit-2511 (Wave 4).

Unlike qwen_multiangle_helper.py (a FOCUSED single-purpose engine locked to a
fixed `<sks> azimuth elevation distance` vocabulary), this helper takes an
arbitrary already-wrapped edit prompt from the shot catalog
(face_variations.VARIATION_CATALOG via wrap_variation_qwen_edit) — a peer of
klein_edit_helper.py, not a replacement for either existing engine.

Same asset-resolution contract as klein_edit_helper.py / qwen_multiangle_helper.py
(canonical filename first, narrow token fallback, never a blind first-file
guess) via the SHARED qwen_edit_assets module — this engine and Qwen
Multi-angle resolve the identical UNET/VAE/TE files, so the user never
configures the same path twice.

Workflow redesign (user-supplied, real ComfyUI capture): this engine no
longer uses the Kontext-lineage nodes (`TextEncodeQwenImageEditPlus`,
`FluxKontextMultiReferenceLatentMethod`, `FluxKontextImageScale`) — instead
`TextEncodeQwenImageEditPlusCustom_lrzjason` + `QwenEditConfigPreparer` +
`QwenEditAdaptiveLongestEdge` (the lrzjason ComfyUI-QwenEditUtils pack) let
a VLM (Qwen2.5-VL, the same text encoder) read the reference image and the
user's plain-language EDIT instruction together, rather than a hand-built
"create a new X of the same Y" prompt. There is also no consistency LoRA in
this graph: that LoRA exists to fight drift across SEVERAL reference images
(Qwen's own multi-ref use case) — Character datasets here always start from
ONE reference image, so it doesn't apply and was deliberately left out.

Same degrade-not-fail LoRA injection as Klein's consistency LoRA
(klein_edit_helper.py:590-629) for the one LoRA this graph DOES have
(Lightning, speed-only): a missing file or an explicit 0 strength bypasses
the node instead of failing the job.

Multi-reference chaining (Klein's `extra_ref_paths`) is NOT supported here —
this workflow is single-reference by design (see above).
"""
from __future__ import annotations
import logging
import os
import random
import shutil
import uuid
import time

from .. import config as cfg
from . import qwen_edit_assets as qea
from ..utils.comfyui import load_workflow_local
from ..job_queue import queue_manager

logger = logging.getLogger(__name__)

WORKFLOW_QWEN_EDIT_PATH = cfg.BACKEND_DIR / 'workflows' / 'qwen_edit_variation.json'

# Nodes this helper rewires or otherwise depends on the shape of — fail LOUDLY
# if the workflow file changes shape instead of silently enqueuing a job with
# the wrong source/prompt/model.
_REQUIRED_NODES = ('39', '10', '17', '38', '31', '7', '6', '21', '23', '33', '13', '8', '40')

# REQUIRED = the graph is invalid without it. No consistency LoRA here (see
# module docstring) — Lightning is the only LoRA, and it's RECOMMENDED-only
# (quality/speed, the graph still runs without it via the non-distilled
# step/cfg fallback).
QWEN_EDIT_REQUIRED = ('qwen_edit_unet', 'qwen_edit_text_encoder', 'qwen_edit_vae')
QWEN_EDIT_RECOMMENDED = ('qwen_edit_lightning_lora',)


class QwenEditModelsMissing(Exception):
    """A graph-critical asset (UNET / text-encoder / VAE) is not on disk, so a
    valid job can't be built. `.missing` lists every absent REQUIRED+RECOMMENDED
    asset so the caller can show one actionable message."""
    def __init__(self, missing):
        self.missing = list(missing)
        super().__init__('Qwen Edit models missing: ' + ', '.join(self.missing))


def qwen_edit_missing_assets():
    """Which qwen_edit assets are NOT on disk, as short action names (a subset
    of QWEN_EDIT_REQUIRED + QWEN_EDIT_RECOMMENDED). Mirrors
    klein_edit_helper.klein_missing_assets()."""
    missing = []
    if not qea.resolve_unet():
        missing.append('qwen_edit_unet')
    if not qea.resolve_text_encoder():
        missing.append('qwen_edit_text_encoder')
    if not qea.resolve_vae():
        missing.append('qwen_edit_vae')
    _, light_path = qea.resolve_lightning_lora()
    if not (light_path and os.path.exists(light_path)):
        missing.append('qwen_edit_lightning_lora')
    return missing


QWEN_EDIT_MIN_BYTES = {
    'qwen_edit_unet': qea.QWEN_EDIT_MIN_BYTES['unet'],
    'qwen_edit_text_encoder': qea.QWEN_EDIT_MIN_BYTES['text_encoder'],
    'qwen_edit_vae': qea.QWEN_EDIT_MIN_BYTES['vae'],
    'qwen_edit_lightning_lora': qea.QWEN_EDIT_MIN_BYTES['lightning_lora'],
}


def qwen_edit_invalid_assets():
    """Assets that ARE on disk under the resolved name but are NOT real,
    loadable weights (e.g. a licence-gate HTML page saved as .safetensors).
    Mirrors klein_edit_helper.klein_invalid_assets(). The shared resolver
    (qea.base_asset_paths()) also reports 'consistency_lora' — this engine
    doesn't use it (see module docstring), so it's skipped here rather than
    KeyError'd; Qwen Multi-angle's own probe still covers it."""
    from . import model_integrity
    out = []
    paths = qea.base_asset_paths()
    key_map = {'unet': 'qwen_edit_unet', 'text_encoder': 'qwen_edit_text_encoder',
              'vae': 'qwen_edit_vae', 'lightning_lora': 'qwen_edit_lightning_lora'}
    for qea_key, path in paths.items():
        asset = key_map.get(qea_key)
        if asset is None:
            continue
        res = model_integrity.validate_model_file(path, min_bytes=QWEN_EDIT_MIN_BYTES.get(asset))
        if res['ok']:
            continue
        out.append({'asset': asset, 'filename': res['filename'],
                    'verdict': res['verdict'], 'blocking': res['blocking'],
                    'reason': res['reason']})
    return out


# --- Custom-node preflight. The lrzjason ComfyUI-QwenEditUtils pack — exact
# repo/download URL unconfirmed, left (None, None) like every other
# not-yet-verified pack in this codebase rather than guessing a link.
QWEN_EDIT_NODE_PACKS = {
    'QwenEditAdaptiveLongestEdge': (None, None),
    'QwenEditConfigPreparer': (None, None),
    'TextEncodeQwenImageEditPlusCustom_lrzjason': (None, None),
    'CFGNorm': (None, None),
    'ModelSamplingAuraFlow': (None, None),
}

_NODES_OK_TTL_S = 300
_nodes_ok_until = 0.0


def _workflow_class_types(workflow):
    return {n.get('class_type') for n in (workflow or {}).values()
           if isinstance(n, dict) and n.get('class_type')}


def qwen_edit_missing_nodes(workflow=None):
    """[{class_type, pack, url}] for every node class this workflow needs that
    the target ComfyUI does NOT expose. FAIL-OPEN on a probe failure — mirrors
    klein_edit_helper.klein_missing_nodes()."""
    global _nodes_ok_until
    shipped = workflow is None
    if shipped:
        if time.time() < _nodes_ok_until:
            return []
        workflow = load_workflow_local(str(WORKFLOW_QWEN_EDIT_PATH)) or {}
    from ..utils.comfyui import fetch_object_info_classes
    available = fetch_object_info_classes()
    if available is None:
        return []
    out = []
    for ct in sorted(_workflow_class_types(workflow) - available):
        pack, url = QWEN_EDIT_NODE_PACKS.get(ct, (None, None))
        out.append({'class_type': ct, 'pack': pack, 'url': url})
    if shipped and not out:
        _nodes_ok_until = time.time() + _NODES_OK_TTL_S
    return out


def format_missing_nodes_message(missing_nodes):
    """Human sentence for a qwen-edit node-missing 409 — mirrors
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
    return ("Your ComfyUI is missing custom node(s) this Qwen Edit workflow "
            "needs: " + '; '.join(bits) + ". Install via ComfyUI-Manager, then restart ComfyUI.")


def _bypass_node(workflow, node_id, passthrough_input):
    """Delete node_id and reconnect every consumer of its output slot 0 to the
    node's own `passthrough_input` upstream — mirrors klein_edit_helper._bypass_node()."""
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


def enqueue_qwen_edit_variation(user_id, source_filename, edit_prompt, negative_prompt='',
                                qwen_model=None, extra_metadata=None, source_path=None,
                                sampler_steps=None, lightning_enabled=True,
                                lightning_strength=None):
    """Copy the source into ComfyUI input, configure the qwen_edit_variation
    workflow with the (already-wrapped) prompt, and enqueue it. Returns the
    app job_id. Raises ValueError on a missing source / unloadable workflow /
    missing required node, RuntimeError if ComfyUI isn't configured,
    QwenEditModelsMissing if a graph-critical asset is absent.
    `negative_prompt` (render_style-aware, see
    face_variations/render_style_presets) goes straight into the plain
    CLIPTextEncode negative node — unlike Klein, this graph doesn't run at a
    guidance-distilled CFG=1, so the negative conditioning has a real effect.
    The empty latent's width/height are NOT set here — they're wired
    statically in the workflow file itself (GetImageSize on the adaptively-
    resized reference -> EmptyLatentImage), so the output always matches the
    reference's aspect ratio without any code-side calculation.
    `lightning_enabled` toggles the Lightning speed LoRA (the only LoRA in
    this graph); off falls back to a non-distilled step/cfg envelope (both
    step counts are extrapolated, not yet measured against a real run — same
    honesty flag as Qwen Multi-angle)."""
    if source_path is None:
        out_dir = _comfy_output_dir()
        if out_dir is None:
            raise RuntimeError('ComfyUI is not configured')
        source_path = os.path.join(out_dir, source_filename)
    if not os.path.exists(source_path):
        raise ValueError(f"source image not found: {source_filename}")

    workflow = load_workflow_local(str(WORKFLOW_QWEN_EDIT_PATH))
    if not workflow:
        raise ValueError("failed to load Qwen Edit workflow")
    for node in _REQUIRED_NODES:
        if node not in workflow:
            raise ValueError(f"workflow node {node} missing — qwen_edit_variation.json has changed")

    unet_ref = qea.resolve_unet(qwen_model)
    vae_ref = qea.resolve_vae()
    te_ref = qea.resolve_text_encoder()
    missing = qwen_edit_missing_assets()
    if any(a in missing for a in QWEN_EDIT_REQUIRED):
        raise QwenEditModelsMissing(missing)

    comfy_input_dir = _comfy_input_dir()
    uid = uuid.uuid4().hex[:8]
    comfy_input = f"qwen_edit_source_{uid}_{source_filename}"
    shutil.copy2(source_path, os.path.join(comfy_input_dir, comfy_input))

    workflow["39"]["inputs"]["image"] = comfy_input
    workflow["10"]["inputs"]["prompt"] = edit_prompt
    workflow["21"]["inputs"]["text"] = negative_prompt or ''
    workflow["31"]["inputs"]["unet_name"] = unet_ref
    workflow["7"]["inputs"]["vae_name"] = vae_ref
    workflow["6"]["inputs"]["clip_name"] = te_ref
    workflow["17"]["inputs"]["seed"] = random.randint(0, 2 ** 64 - 1)
    # UNIQUE prefix per job — same ComfyUI-output-counter reasoning as Klein's
    # own note (a shared prefix made every tile display the same file).
    workflow["38"]["inputs"]["filename_prefix"] = f"{user_id}_QwenEdit_{uid}"

    # Lightning speed LoRA (optional, node 23, the ONLY LoRA in this graph —
    # see module docstring for why there's no consistency LoRA) — bypass +
    # fall back to a non-distilled step/cfg envelope when off or absent.
    light_lora, light_path = qea.resolve_lightning_lora()
    light_strength = cfg.get('qwen_edit.lightning_strength', 1.0)
    if lightning_strength is not None:
        light_strength = max(0.0, min(1.5, float(lightning_strength)))
    use_lightning = (lightning_enabled and light_path and os.path.exists(light_path)
                     and light_strength and float(light_strength) > 0)
    if "23" in workflow:
        if use_lightning:
            workflow["23"]["inputs"]["lora_name"] = light_lora
            workflow["23"]["inputs"]["strength_model"] = light_strength
        else:
            if lightning_enabled and not (light_path and os.path.exists(light_path)):
                logger.warning(f"Lightning LoRA not found at {light_path} — bypassing node 23")
            _bypass_node(workflow, "23", "model")
    if "17" in workflow:
        if sampler_steps is not None:
            workflow["17"]["inputs"]["steps"] = max(1, int(sampler_steps))
        else:
            # 8 steps matches the reference capture this workflow was built
            # from (Lightning-8steps LoRA); the non-Lightning fallback is
            # EXTRAPOLATED, untested — same honesty flag as Qwen Multi-angle.
            workflow["17"]["inputs"]["steps"] = 8 if use_lightning else 20
        workflow["17"]["inputs"]["cfg"] = 1 if use_lightning else 4

    job_id = str(uuid.uuid4())
    meta = {"model_name": "qwen_edit_dataset"}
    if extra_metadata:
        meta.update(extra_metadata)
    queue_manager.add_job(job_type="image", user_id=str(user_id), workflow_data=workflow,
                          prompt=edit_prompt, job_id=job_id, metadata=meta)
    return job_id
