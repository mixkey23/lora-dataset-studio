"""Second local LoRA-training engine (Wave 2): kohya-ss/musubi-tuner, scoped
to Qwen-Image ONLY. Pure mechanics module — mirrors how klein_edit_helper.py
is a thin mechanics module invoked by a service: lora_training.py owns the
ONE shared launch/queue/status/watcher system for every engine, this module
only knows how to (1) resolve its own install + Qwen-Image weight paths,
(2) write the dataset TOML, (3) run the two pre-caching scripts, and
(4) build the accelerate-launch argv + spawn the training subprocess.

ai-toolkit stays a blanket requirement for every training route regardless
of engine (confirmed product decision) — musubi-tuner only replaces the
subprocess-launch step for a qwen_image dataset whose engine is 'musubi'.
Fresh launches only this wave: musubi-tuner's own resume mechanism was not
verifiable from its docs, so there is no continue/resume path here.

CLI flags/TOML keys below are verified against the live kohya-ss/musubi-tuner
docs (docs/qwen_image.md, docs/dataset_config.md, fetched 2026-07-20/23), not
guessed. `--save_every_n_steps` is still not confirmed present for the
Qwen-Image script, so it stays omitted.

`--fp8_base`/`--fp8_scaled`/`--blocks_to_swap` were confirmed on a second doc
pass (2026-07-23) — the doc ships an exact VRAM table for Qwen-Image LoRA
training (no options ~42GB, fp8_base+fp8_scaled ~30GB, +blocks_to_swap 16
~24GB, +blocks_to_swap 45 ~12GB), which anchored the FIRST version of
MUSUBI_PROFILES below (3 profiles, kohya-ss doc only).

MUSUBI_PROFILES v2 (2026-07-23, same day, repo owner's own request): the
repo owner obtained the ACTUAL ready-made per-VRAM-tier TOML configs shipped
by a third-party musubi-tuner GUI, SECourses' `SECourses_Musubi_Trainer`
(github.com/FurkanGozukara/SECourses_Musubi_Trainer,
`Qwen_Training_Configs/LoRA_Training/100 epoch/Tier*.toml`) — these are
REAL, field-tested configs (not a doc example), giving us far more
granularity than the 4-point official table, PLUS several values the
official kohya-ss doc never mentions at all:

    network_alpha = 128          (decoupled from network_dim — Tier3-6 drop
                                   dim to 64/32/24 for VRAM but KEEP alpha 128)
    optimizer_type = "AdaFactor" (not adamw8bit)
    optimizer_args = scale_parameter=False relative_step=False
                     warmup_init=False weight_decay=0.01  (AdaFactor needs
                     these to NOT run in its own auto-LR mode)
    learning_rate = 0.00015      (not 5e-5)
    timestep_sampling = "sigmoid" (matches what this app already defaults to
                                   via the shared qwen_image family setting —
                                   NOT a change from before)
    weighting_scheme = "mode"    (not "none" — WAS hardcoded before this pass)
    dataset_resolution = 1328 (or 1024/768 on the tightest tiers)

Two SECourses fields were deliberately NOT ported: `compile`/`compile_backend`
etc. (torch.compile) and `caching_teo_device` (CPU-offload the text-encoder
caching step) — SECourses installs their OWN musubi-tuner fork
(`FurkanGozukara/musubi-tuner`, see their RunPod install script), not vanilla
kohya-ss, and neither flag is confirmed present on kohya-ss's own scripts
(the `--device` flag in particular is absent from kohya-ss's own
qwen_image_cache_text_encoder_outputs.py doc example) — emitting an
unconfirmed flag would crash argparse outright rather than degrade, so both
stay omitted per this module's "never guess a flag" rule.

MUSUBI_PROFILES entries are named by VRAM (each maps 1:1 to a named
SECourses tier file, cited in its own `note`), replacing the earlier
'fast'/'high_quality'/'rtx5090' names (this setting has not shipped in a
release yet — CLAUDE.md's rename-needs-an-alias rule doesn't apply to an
unreleased key). Unset (None/'auto') keeps today's behaviour byte-identical:
shared qwen_image rank/optimizer/lr/timestep/resolution, no fp8, no swap,
weighting_scheme 'none' (the ORIGINAL hardcoded value, still the default).
"""
from __future__ import annotations
import logging
import os
import subprocess

from .. import config as cfg

logger = logging.getLogger(__name__)

# Scripts are at src/musubi_tuner/*.py relative to the repo root — verified
# against the current docs/qwen_image.md example commands, not the repo's
# top-level (an earlier, stale fetch suggested root-level scripts; the
# doc's own examples are authoritative).
QWEN_IMAGE_SCRIPTS = {
    'cache_latents': os.path.join('src', 'musubi_tuner', 'qwen_image_cache_latents.py'),
    'cache_text_encoder': os.path.join('src', 'musubi_tuner', 'qwen_image_cache_text_encoder_outputs.py'),
    'train': os.path.join('src', 'musubi_tuner', 'qwen_image_train_network.py'),
}

# model_version flag value per our 'image'/'edit' variant (mirrors the
# qwen_image training family's variant split in lora_training.py).
MODEL_VERSION = {'image': 'original', 'edit': 'edit-2511'}

# AdaFactor needs these to NOT run in its own auto-LR mode (transformers'
# AdaFactor defaults to relative_step=True, which ignores --learning_rate
# entirely) — every SECourses tier pairs AdaFactor with this exact arg set.
_ADAFACTOR_ARGS = ('scale_parameter=False', 'relative_step=False',
                   'warmup_init=False', 'weight_decay=0.01')

# GPU/quality profiles (musubi-tuner engine ONLY — ai-toolkit's own rank/
# optimizer/lr sliders are untouched; Auto keeps this dataset's shared
# settings). Each profile is the FULL recipe from one real, named SECourses
# tier config (see module docstring) — not just VRAM knobs: rank, alpha,
# optimizer, learning rate, weighting_scheme and resolution all come from
# that tier's own shipped TOML.
MUSUBI_PROFILES = {
    'vram48': {
        'label': '48GB+ VRAM — fastest (rank 128)',
        'rank': 128, 'alpha': 128, 'fp8_base': False, 'fp8_scaled': False,
        'blocks_to_swap': 0, 'resolution': 1328,
        'optimizer': 'AdaFactor', 'optimizer_args': _ADAFACTOR_ARGS,
        'learning_rate': 0.00015, 'weighting_scheme': 'mode',
        'note': ('SECourses Tier1_51300_MB_VRAM_Fastest.toml — full rank 128, no '
                 'fp8, no block-swap. Needs ~51GB (A100 80GB/H100/RTX 6000 Ada-class).'),
    },
    'vram29': {
        'label': '~29GB VRAM, no fp8 (rank 128)',
        'rank': 128, 'alpha': 128, 'fp8_base': False, 'fp8_scaled': False,
        'blocks_to_swap': 35, 'resolution': 1328,
        'optimizer': 'AdaFactor', 'optimizer_args': _ADAFACTOR_ARGS,
        'learning_rate': 0.00015, 'weighting_scheme': 'mode',
        'note': ('SECourses Tier1_29300_MB_VRAM.toml — full rank 128 at full bf16 '
                 'precision (no fp8 quality loss), traded for more block-swap '
                 '(slower). Alternative to the RTX 5090 profile below for the '
                 'same card if you prefer bf16 over fp8.'),
    },
    'rtx5090': {
        'label': 'RTX 5090 / 32GB — fp8 (rank 128)',
        'rank': 128, 'alpha': 128, 'fp8_base': True, 'fp8_scaled': True,
        'blocks_to_swap': 20, 'resolution': 1328,
        'optimizer': 'AdaFactor', 'optimizer_args': _ADAFACTOR_ARGS,
        'learning_rate': 0.00015, 'weighting_scheme': 'mode',
        'note': ('SECourses Tier2_30000_MB.toml — fp8_base+fp8_scaled fits full '
                 'rank 128 in ~30GB with headroom on a 32GB card, faster than the '
                 '~29GB bf16 profile above.'),
    },
    'vram22': {
        'label': '~22GB VRAM (24GB cards, e.g. 3090/4090) — fp8 (rank 128)',
        'rank': 128, 'alpha': 128, 'fp8_base': True, 'fp8_scaled': True,
        'blocks_to_swap': 36, 'resolution': 1328,
        'optimizer': 'AdaFactor', 'optimizer_args': _ADAFACTOR_ARGS,
        'learning_rate': 0.00015, 'weighting_scheme': 'mode',
        'note': 'SECourses Tier2_21700_MB.toml — full rank 128, fp8, heavier block-swap for 24GB cards.',
    },
    'vram11': {
        'label': '~11GB VRAM — fp8 + lower rank (64)',
        'rank': 64, 'alpha': 128, 'fp8_base': True, 'fp8_scaled': True,
        'blocks_to_swap': 58, 'resolution': 1328,
        'optimizer': 'AdaFactor', 'optimizer_args': _ADAFACTOR_ARGS,
        'learning_rate': 0.00015, 'weighting_scheme': 'mode',
        'note': ('SECourses Tier3_10700_MB_VRAM.toml — rank drops to 64 (alpha stays '
                 '128, decoupled) plus maximum block-swap (58) to fit ~11GB cards.'),
    },
    'vram5': {
        'label': '~5GB VRAM — minimum (rank 24, 768px)',
        'rank': 24, 'alpha': 128, 'fp8_base': True, 'fp8_scaled': True,
        'blocks_to_swap': 58, 'resolution': 768,
        'optimizer': 'AdaFactor', 'optimizer_args': _ADAFACTOR_ARGS,
        'learning_rate': 0.00015, 'weighting_scheme': 'mode',
        'note': ('SECourses Tier6_5075_MB_VRAM.toml — the floor: rank 24, training '
                 'resolution dropped to 768px (from 1328), maximum block-swap. Fits '
                 'entry-level 6-8GB cards.'),
    },
}
MUSUBI_PROFILE_CHOICES = tuple(MUSUBI_PROFILES.keys())


def musubi_dir():
    d = cfg.musubi_tuner_path('dir')
    if not d:
        raise RuntimeError('musubi-tuner is not configured')
    return d


def _venv_python():
    p = cfg.musubi_tuner_path('venv_python')
    if not p:
        raise RuntimeError('musubi-tuner is not configured')
    return p


def is_installed() -> bool:
    """musubi-tuner installed AND its Qwen-Image scripts present?"""
    p = cfg.musubi_tuner_path('venv_python')
    if not p or not p.is_file():
        return False
    d = cfg.musubi_tuner_path('dir')
    return all((d / rel).is_file() for rel in QWEN_IMAGE_SCRIPTS.values())


def qwen_image_weights() -> dict:
    """{'dit', 'vae', 'text_encoder'} -> configured path (possibly empty)."""
    return {
        'dit': cfg.get('musubi_tuner.qwen_image_dit') or '',
        'vae': cfg.get('musubi_tuner.qwen_image_vae') or '',
        'text_encoder': cfg.get('musubi_tuner.qwen_image_text_encoder') or '',
    }


def missing_qwen_image_weights() -> list:
    """Which of dit/vae/text_encoder are unset or don't exist on disk."""
    return [k for k, v in qwen_image_weights().items()
           if not v or not os.path.isfile(v)]


def write_dataset_toml(dataset_folder: str, cache_dir: str, resolution: int,
                       caption_ext: str = 'txt') -> str:
    """Writes the musubi-tuner dataset TOML next to `dataset_folder` (sibling
    file, `<dataset_folder>_musubi.toml`) and returns its path. Keys verified
    against docs/dataset_config.md. A single square `resolution` (no
    multi-scale bucket list like ai-toolkit) — enable_bucket stays off for
    this first cut, kept simple over exact aspect-ratio bucketing."""
    toml_path = f'{dataset_folder}_musubi.toml'
    content = (
        '[general]\n'
        f'resolution = [{resolution}, {resolution}]\n'
        f'caption_extension = ".{caption_ext}"\n'
        'batch_size = 1\n'
        'num_repeats = 1\n'
        'enable_bucket = false\n'
        'bucket_no_upscale = false\n'
        '\n'
        '[[datasets]]\n'
        f'image_directory = "{dataset_folder}"\n'
        f'cache_directory = "{cache_dir}"\n'
    )
    with open(toml_path, 'w', encoding='utf-8') as fh:
        fh.write(content)
    return toml_path


def _run_step(argv: list, cwd: str, env: dict, log_path: str, step_label: str) -> None:
    """Runs one pre-caching step to completion (blocking), appending its
    output to `log_path`. Raises RuntimeError with the log tail on a
    nonzero exit — surfaces as a normal launch failure to the caller, no
    training-in-progress flag has been set yet at this point."""
    with open(log_path, 'a', encoding='utf-8') as logf:
        logf.write(f'\n--- {step_label} ---\n{" ".join(argv)}\n')
        logf.flush()
        result = subprocess.run(argv, cwd=cwd, env=env, stdout=logf,
                                stderr=subprocess.STDOUT)
    if result.returncode != 0:
        tail = ''
        try:
            with open(log_path, encoding='utf-8', errors='ignore') as fh:
                tail = fh.read()[-2000:]
        except OSError:
            pass
        raise RuntimeError(f'musubi-tuner {step_label} failed (exit {result.returncode}): {tail}')


def run_precache(toml_path: str, model_version: str, log_path: str) -> None:
    """Runs the two Qwen-Image pre-caching scripts (latents, then text-encoder
    outputs) sequentially and synchronously — both are fast relative to
    training, and training must never start against a stale/missing cache."""
    weights = qwen_image_weights()
    cwd = str(musubi_dir())
    python = str(_venv_python())
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    _run_step(
        [python, QWEN_IMAGE_SCRIPTS['cache_latents'],
         '--dataset_config', toml_path, '--vae', weights['vae'],
         '--model_version', model_version],
        cwd, env, log_path, 'cache latents')
    _run_step(
        [python, QWEN_IMAGE_SCRIPTS['cache_text_encoder'],
         '--dataset_config', toml_path, '--text_encoder', weights['text_encoder'],
         '--batch_size', '1', '--model_version', model_version],
        cwd, env, log_path, 'cache text-encoder outputs')


def build_train_argv(toml_path: str, model_version: str, rank: int,
                     learning_rate: float, optimizer: str, timestep_type: str,
                     max_train_epochs: int, output_dir: str, output_name: str,
                     seed: int = 42, fp8_base: bool = False,
                     fp8_scaled: bool = False, blocks_to_swap: int = 0,
                     network_alpha: int | None = None,
                     weighting_scheme: str = 'none',
                     optimizer_args: tuple | list | None = None) -> list:
    """Pure function: the accelerate-launch argv for qwen_image_train_network.py.
    Rank/lr/optimizer/weighting_scheme/network_alpha come from lora_training.py
    — either the shared family-scoped helpers (Auto/no profile: unchanged
    behaviour) or a MUSUBI_PROFILES entry's full recipe (see module
    docstring). This function does not re-derive any of that, it only accepts
    what the caller resolved.

    `timestep_type` maps our family-level 'sigmoid' default onto musubi's
    `--timestep_sampling` vocabulary; only 'shift'/'sigmoid'/'uniform' are
    confirmed, any other value degrades to 'shift'.

    `fp8_base`/`fp8_scaled`/`blocks_to_swap` are confirmed doc flags (see
    module docstring's VRAM table). `fp8_scaled` is only ever meaningful
    paired with `fp8_base` — emitted defensively only when `fp8_base` is also
    set. `blocks_to_swap<=0` omits the flag entirely (0 = no swap = today's
    default behaviour).

    `network_alpha` (None = omit, decoupled from `rank` — a MUSUBI_PROFILES
    entry can keep alpha=128 while dropping rank for VRAM) and
    `weighting_scheme` (default 'none', the ORIGINAL hardcoded value —
    'mode' is a profile-only override) are both confirmed via SECourses'
    real, shipped tier configs (see module docstring), not the kohya-ss doc
    directly. `optimizer_args` (e.g. AdaFactor's scale_parameter=False etc.)
    is emitted as one `--optimizer_args k=v k2=v2 ...` flag, a standard
    kohya-ecosystem convention — omitted entirely when falsy."""
    weights = qwen_image_weights()
    sampling = timestep_type if timestep_type in ('shift', 'sigmoid', 'uniform') else 'shift'
    argv = [
        '-m', 'accelerate.commands.launch',
        '--num_cpu_threads_per_process', '1', '--mixed_precision', 'bf16',
        QWEN_IMAGE_SCRIPTS['train'],
        '--dit', weights['dit'], '--vae', weights['vae'],
        '--text_encoder', weights['text_encoder'], '--model_version', model_version,
        '--dataset_config', toml_path, '--sdpa', '--mixed_precision', 'bf16',
        '--timestep_sampling', sampling, '--weighting_scheme', (weighting_scheme or 'none'),
        '--discrete_flow_shift', '2.2',
        '--optimizer_type', optimizer, '--learning_rate', str(learning_rate),
        '--gradient_checkpointing',
        '--max_data_loader_n_workers', '2', '--persistent_data_loader_workers',
        '--network_module', 'networks.lora_qwen_image', '--network_dim', str(rank),
        '--max_train_epochs', str(max_train_epochs), '--save_every_n_epochs', '1',
        '--seed', str(seed),
        '--output_dir', output_dir, '--output_name', output_name,
    ]
    if network_alpha:
        argv += ['--network_alpha', str(network_alpha)]
    if optimizer_args:
        argv += ['--optimizer_args', *list(optimizer_args)]
    if fp8_base:
        argv.append('--fp8_base')
        if fp8_scaled:
            argv.append('--fp8_scaled')
    if blocks_to_swap and blocks_to_swap > 0:
        argv += ['--blocks_to_swap', str(blocks_to_swap)]
    return argv


def spawn_training(argv: list, cwd: str, env: dict, log_path: str) -> subprocess.Popen:
    """Spawns the training subprocess — same shape as ai-toolkit's own spawn
    (venv python + argv, merged stdout/stderr into log_path, no console
    window on Windows) so lora_training.py's existing watcher thread works
    on either engine's Popen unchanged."""
    logf = open(log_path, 'a', encoding='utf-8')
    logf.write(f'\n--- accelerate launch (training) ---\n')
    logf.flush()
    return subprocess.Popen(
        [str(_venv_python())] + argv,
        cwd=cwd, env=env, shell=False,
        stdout=logf, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
