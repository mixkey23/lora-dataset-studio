"""What the MACHINE looked like when a run was launched.

Two training runs can differ without a single dataset image or setting having
moved: a `git pull` in ai-toolkit changes the trainer itself, a torch/CUDA
upgrade changes the kernels, and re-downloading "Krea 2 Raw" can put a DIFFERENT
file behind the same name. None of that was recorded, so any comparison that
blamed the dataset for the gap was guessing.

Everything here is BEST-EFFORT and CHEAP:

  * nothing imports torch (a cold import costs seconds) — the version is READ
    out of the ai-toolkit venv's `torch/version.py`, a two-line file;
  * the ai-toolkit revision is one `git rev-parse`, hard-timeout 4 s, and only
    when that install is actually a git checkout;
  * the GPU name is one `nvidia-smi` query;
  * a base-model file is identified by size + a SAMPLED hash (head/tail), never
    by reading 12 GB.

Every probe is memoised for `_TTL` seconds and every failure degrades to None —
an unknown is recorded as absent, never as a claim. A launch must never wait on,
or fail because of, provenance.

Paths are redacted (`~`) before they are stored: a snapshot ends up in the Share
config and in pasted diagnostics.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
import time
from pathlib import Path

from .. import config as cfg
from ..utils.redact import redact_user_paths

logger = logging.getLogger(__name__)

_TTL = 600.0                    # 10 min — none of this changes between two runs
_GIT_TIMEOUT = 4                # seconds; a slow/locked repo must not hold a launch
_SMI_TIMEOUT = 5
_SAMPLE_BYTES = 1 << 20         # 1 MiB head + 1 MiB tail for a big model file

_cache: dict = {}


def _no_window():
    return getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def _memo(key, producer):
    """Memoise `producer()` for _TTL seconds. A raising producer caches None so a
    broken probe is not retried on every launch."""
    now = time.time()
    hit = _cache.get(key)
    if hit is not None and (now - hit[0]) < _TTL:
        return hit[1]
    try:
        value = producer()
    except Exception:
        logger.debug('environment probe %s failed', key, exc_info=True)
        value = None
    _cache[key] = (now, value)
    return value


def clear_cache() -> None:
    """Drop every memoised probe (tests, and Settings changing the ai-toolkit dir)."""
    _cache.clear()


# --- ai-toolkit revision -------------------------------------------------------

def _aitoolkit_probe():
    root = cfg.aitoolkit_path('dir')
    if not root or not Path(root).is_dir():
        return None
    if not (Path(root) / '.git').exists():
        # A zip/manual install has no revision to report. Honest silence.
        return None
    out = {}
    for key, args in (('commit', ['rev-parse', '--short=10', 'HEAD']),
                      ('committed_at', ['log', '-1', '--format=%cI'])):
        try:
            proc = subprocess.run(['git', '-C', str(root), *args],
                                  capture_output=True, text=True,
                                  timeout=_GIT_TIMEOUT,
                                  creationflags=_no_window())
        except (OSError, subprocess.SubprocessError):
            return out or None
        if proc.returncode != 0:
            return out or None
        value = (proc.stdout or '').strip()
        if value:
            out[key] = value
    return out or None


def aitoolkit_revision():
    """{'commit': 'a1b2c3d4e5', 'committed_at': '...'} for the ai-toolkit checkout
    that will train, or None (not configured / not a git checkout / git absent)."""
    return _memo('aitoolkit', _aitoolkit_probe)


# --- torch / CUDA (read, never imported) ---------------------------------------

_VERSION_RE = re.compile(r"^__version__\s*[:=][^'\"]*['\"]([^'\"]+)['\"]", re.MULTILINE)
_CUDA_RE = re.compile(r"^cuda\s*[:=][^'\"]*['\"]([^'\"]+)['\"]", re.MULTILINE)


def _site_packages_candidates(python_path: Path):
    """Plausible site-packages dirs for an interpreter, without running it.
    Windows keeps them at `<env>/Lib/site-packages`, POSIX at
    `<env>/lib/python3.X/site-packages` — both layouts are tried because LDS runs
    on Windows, on Linux and inside the Docker image."""
    env = python_path.parent.parent      # .../Scripts/python.exe -> env root
    yield env / 'Lib' / 'site-packages'
    lib = env / 'lib'
    if lib.is_dir():
        try:
            for child in sorted(lib.iterdir()):
                if child.name.startswith('python'):
                    yield child / 'site-packages'
        except OSError:
            return


def _torch_probe():
    python = cfg.aitoolkit_path('venv_python')
    if not python:
        return None
    for sp in _site_packages_candidates(Path(python)):
        vf = sp / 'torch' / 'version.py'
        try:
            text = vf.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        m = _VERSION_RE.search(text)
        if not m:
            continue
        out = {'torch': m.group(1)}
        c = _CUDA_RE.search(text)
        if c:
            out['cuda'] = c.group(1)
        return out
    return None


def torch_info():
    """{'torch': '2.7.0+cu128', 'cuda': '12.8'} of the ai-toolkit venv — the
    interpreter that TRAINS, not the one serving this request — or None.

    Deliberately parsed out of `torch/version.py` instead of imported: importing
    torch in a cold venv costs seconds and this runs on the launch path."""
    return _memo('torch', _torch_probe)


# --- GPU -----------------------------------------------------------------------

def _gpu_probe():
    # memory.total rides along in the SAME query — it costs nothing extra and it
    # is what lets the training panel advise "you can turn the memory savers off"
    # instead of making every user guess what their card takes. nvidia-smi prints
    # it as "24564 MiB"; anything unparseable is simply absent (never a claim).
    proc = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
         '--format=csv,noheader'],
        capture_output=True, text=True, timeout=_SMI_TIMEOUT,
        creationflags=_no_window())
    if proc.returncode != 0:
        return None
    line = ((proc.stdout or '').strip().splitlines() or [''])[0]
    parts = [p.strip() for p in line.split(',')]
    name = parts[0] if parts else ''
    if not name:
        return None
    out = {'name': name}
    if len(parts) > 1 and parts[1]:
        out['driver'] = parts[1]
    if len(parts) > 2:
        m = re.search(r'(\d+)', parts[2])
        if m:
            # Reported in MiB. Rounded to one decimal GiB — a 24 GB card reports
            # 24564 MiB (23.99 GiB), so flooring to an int would call it 23.
            out['vram_gb'] = round(int(m.group(1)) / 1024.0, 1)
    return out


def gpu_info():
    """{'name': 'NVIDIA GeForce RTX 4090', 'driver': '...', 'vram_gb': 24.0} or
    None (no NVIDIA card, nvidia-smi absent, a cloud launch from a GPU-less
    machine). `vram_gb` may be missing on its own if the driver didn't report it —
    callers must treat an absent value as "unknown", never as "small"."""
    return _memo('gpu', _gpu_probe)


def local_vram_gb():
    """Total VRAM of this machine's first NVIDIA GPU in GiB, or None when it can't
    be known. The ONLY consumer-facing use is advisory (see the memory-saving
    levers in lora_training): a None must always degrade to generic guidance, and
    a detected value never overrides a user's explicit choice."""
    gpu = gpu_info() or {}
    v = gpu.get('vram_gb')
    return v if isinstance(v, (int, float)) and v > 0 else None


# --- base-model file identity --------------------------------------------------

def file_signature(path):
    """A cheap, stable identity for a possibly ENORMOUS file: size + a sha1 over
    the first and last 1 MiB. Re-downloading "Krea 2 Raw" and landing a different
    file changes this; renaming or moving it does not.

    Hashing a 12 GB safetensors end to end would take minutes on the launch path,
    so the signature is deliberately SAMPLED — it proves a difference, it does not
    prove byte-for-byte identity. Returns None when the file can't be read."""
    try:
        p = Path(path)
        size = p.stat().st_size
        h = hashlib.sha1(str(size).encode())
        with open(p, 'rb') as fh:
            h.update(fh.read(_SAMPLE_BYTES))
            if size > _SAMPLE_BYTES * 2:
                fh.seek(-_SAMPLE_BYTES, os.SEEK_END)
                h.update(fh.read(_SAMPLE_BYTES))
        return {'name': p.name, 'size': size, 'sig': h.hexdigest()[:16],
                'sampled': size > _SAMPLE_BYTES * 2}
    except (OSError, ValueError):
        return None


def _resolve_base_path(base_model):
    """Absolute path of the base weights a launch will actually load, or None.

    An absolute value IS the file (custom weights). A ComfyUI-relative name is
    joined against the configured checkpoint roots — a bounded number of
    `isfile` probes, never the recursive walk `list_models` does. The official
    hosted base resolves to nothing local, which is a truthful None."""
    if not base_model:
        return None
    raw = str(base_model)
    if os.path.isabs(raw):
        return raw if os.path.isfile(raw) else None
    try:
        from . import comfy_model_paths as cmp
        roots = cmp.search_roots('checkpoints')
    except Exception:
        return None
    for root in roots:
        cand = os.path.join(root, raw)
        if os.path.isfile(cand):
            return cand
    return None


def base_model_identity(base_model):
    """Identity of the FILE behind a base-model name, or None when there is no
    local file (official hosted base) or it can't be read. Memoised per (path,
    size, mtime) so a repeat launch on the same weights costs one `stat`."""
    path = _resolve_base_path(base_model)
    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = ('base', path, st.st_size, int(st.st_mtime))
    hit = _cache.get(key)
    if hit is not None:
        return hit[1]
    info = file_signature(path)
    if info is not None:
        # The file name alone leaks nothing, but custom weights may live under the
        # home dir — and the folder is what identifies the file for a human.
        info['folder'] = redact_user_paths(os.path.dirname(path))
    _cache[key] = (time.time(), info)
    return info


# --- the whole picture ---------------------------------------------------------

def capture(base_model=None) -> dict:
    """The environment stamp for one launch. Never raises; missing probes are
    simply absent keys, so `{}` means "this machine told us nothing" and the UI
    can say so instead of inventing."""
    env = {}
    try:
        from ..version import APP_VERSION
        env['app'] = APP_VERSION
    except Exception:
        pass
    at = aitoolkit_revision()
    if at:
        env['aitoolkit'] = at
    ti = torch_info()
    if ti:
        env.update(ti)
    gpu = gpu_info()
    if gpu:
        env['gpu'] = gpu.get('name')
        if gpu.get('driver'):
            env['gpu_driver'] = gpu['driver']
        if gpu.get('vram_gb'):
            env['gpu_vram_gb'] = gpu['vram_gb']
    base = base_model_identity(base_model)
    if base:
        env['base_file'] = base
    return env
