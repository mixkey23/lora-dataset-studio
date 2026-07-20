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
docs (docs/qwen_image.md, docs/dataset_config.md, fetched 2026-07-20), not
guessed. `--network_alpha`/`--fp8_base`/`--blocks_to_swap`/
`--save_every_n_steps` were not confirmed present for the Qwen-Image script
in that fetch, so they're deliberately omitted rather than guessed at.
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
                     seed: int = 42) -> list:
    """Pure function: the accelerate-launch argv for qwen_image_train_network.py,
    built from the verified doc example. Rank/lr/optimizer come from
    lora_training.py's existing family-scoped helpers (_lora_rank(ds,
    'qwen_image') etc.) so the recipe stays identical across engines — this
    function does not re-derive any of that. No `--network_alpha`: not
    confirmed present for the Qwen-Image script (see module docstring), so
    musubi runs at its own alpha=dim default rather than guessing the flag.

    `timestep_type` maps our family-level 'sigmoid' default onto musubi's
    `--timestep_sampling` vocabulary; only 'shift' is confirmed from the doc
    example, so any other value degrades to 'shift' rather than guessing an
    unverified flag value."""
    weights = qwen_image_weights()
    sampling = timestep_type if timestep_type in ('shift', 'sigmoid', 'uniform') else 'shift'
    return [
        '-m', 'accelerate.commands.launch',
        '--num_cpu_threads_per_process', '1', '--mixed_precision', 'bf16',
        QWEN_IMAGE_SCRIPTS['train'],
        '--dit', weights['dit'], '--vae', weights['vae'],
        '--text_encoder', weights['text_encoder'], '--model_version', model_version,
        '--dataset_config', toml_path, '--sdpa', '--mixed_precision', 'bf16',
        '--timestep_sampling', sampling, '--weighting_scheme', 'none',
        '--discrete_flow_shift', '2.2',
        '--optimizer_type', optimizer, '--learning_rate', str(learning_rate),
        '--gradient_checkpointing',
        '--max_data_loader_n_workers', '2', '--persistent_data_loader_workers',
        '--network_module', 'networks.lora_qwen_image', '--network_dim', str(rank),
        '--max_train_epochs', str(max_train_epochs), '--save_every_n_epochs', '1',
        '--seed', str(seed),
        '--output_dir', output_dir, '--output_name', output_name,
    ]


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
