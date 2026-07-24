import os
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")

import argparse
import codecs
import concurrent.futures
import hashlib
import json
import queue
import random
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlsplit

try:
    import requests
    from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
    from huggingface_hub.errors import (
        EntryNotFoundError,
        GatedRepoError,
        HFValidationError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )
    from huggingface_hub.utils import WeakFileLock, build_hf_headers
except ImportError as exc:
    print(
        "Missing or outdated download dependencies. Run: "
        'python -m pip install --upgrade "huggingface_hub>=0.34" hf_xet requests',
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

# Model configurations
MODEL_CONFIGS = {
    "qwen_image": {
        "repo_id": "MonsterMMORPG/Wan_GGUF",
        "files": [
            "qwen_2.5_vl_7b_bf16.safetensors",
            "qwen_train_vae.safetensors",
            "qwen_image_bf16.safetensors"
        ],
        "name": "Qwen Image Training Models",
        "description": "Qwen 2.5 VL 7B model with BF16 precision for image training",
        "default_dir": "Training_Models_Qwen"
    },
    "qwen_image_2512": {
        "repo_id": "MonsterMMORPG/Wan_GGUF",
        "files": [
            "qwen_2.5_vl_7b_bf16.safetensors",
            "qwen_train_vae.safetensors",
            "Qwen_Image_2512_BF16.safetensors"
        ],
        "name": "Qwen Image (2512) Training Models",
        "description": "Qwen Image 2512 model with BF16 precision for image training",
        "default_dir": "Training_Models_Qwen"
    },
    "qwen_image_edit_plus": {
        "repo_id": "MonsterMMORPG/Wan_GGUF",
        "files": [
            "qwen_2.5_vl_7b_bf16.safetensors",
            "qwen_train_vae.safetensors",
            "Qwen_Image_Edit_Plus_2509_bf16.safetensors"
        ],
        "name": "Qwen Image Edit Plus (2509) Training Models",
        "description": "Qwen Image Edit Plus 2509 model with BF16 precision for image editing training with multiple control images",
        "default_dir": "Training_Models_Qwen"
    },
    "qwen_image_edit_2511": {
        "repo_id": "MonsterMMORPG/Wan_GGUF",
        "files": [
            "qwen_2.5_vl_7b_bf16.safetensors",
            "qwen_train_vae.safetensors",
            "Qwen_Image_Edit_2511_BF16.safetensors"
        ],
        "name": "Qwen Image Edit (2511) Training Models",
        "description": "Qwen Image Edit 2511 model with BF16 precision for image editing training with multiple control images",
        "default_dir": "Training_Models_Qwen"
    },
    "wan21_t2v": {
        "repo_id": "MonsterMMORPG/Wan_GGUF",
        "files": [
            "Wan2_1_VAE_bf16.safetensors",
            "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors",
            "umt5-xxl-enc-bf16.safetensors",
            "wan2.1_t2v_14B_bf16.safetensors"
        ],
        "name": "Wan 2.1 Text to Video Training Models",
        "description": "Wan 2.1 14B model with BF16 precision for text-to-video training",
        "default_dir": "Training_Models_Wan"
    },
    "wan22_t2v": {
        "repo_id": "MonsterMMORPG/Wan_GGUF",
        "files": [
            "Wan2_1_VAE_bf16.safetensors",
            "umt5-xxl-enc-bf16.safetensors",
            "Wan-2.2-T2V-High-Noise-BF16.safetensors",
            "Wan-2.2-T2V-Low-Noise-BF16.safetensors"
        ],
        "name": "Wan 2.2 Text to Video Training Models",
        "description": "Wan 2.2 14B models with BF16 precision for text-to-video training (high and low noise variants)",
        "default_dir": "Training_Models_Wan"
    },
    "wan22_i2v": {
        "repo_id": "MonsterMMORPG/Wan_GGUF",
        "files": [
            "Wan2_1_VAE_bf16.safetensors",
            "umt5-xxl-enc-bf16.safetensors",
            "Wan-2.2-I2V-Low-Noise-BF16.safetensors",
            "Wan-2.2-I2V-High-Noise-BF16.safetensors"
        ],
        "name": "Wan 2.2 Image to Video Training Models",
        "description": "Wan 2.2 14B models with BF16 precision for image-to-video training (high and low noise variants)",
        "default_dir": "Training_Models_Wan"
    },
    "flux2_dev": {
        "repo_id": "MonsterMMORPG/Wan_GGUF",
        "files": [
            {"remote": "FLUX_2_Models/FLUX_2_Dev_BF16.safetensors", "local": "FLUX_2_Dev_BF16.safetensors"},
            {"remote": "FLUX_2_Models/FLUX_2_Klein_Train_VAE.safetensors", "local": "FLUX_2_Klein_Train_VAE.safetensors"},
            {"remote": "FLUX_2_Models/Mistral3_FLUX2_BF16.safetensors", "local": "Mistral3_FLUX2_BF16.safetensors"},
        ],
        "name": "FLUX 2 Dev Training Models",
        "description": "FLUX 2 Dev bundle (BF16) with Klein training VAE and Mistral3 text encoder",
        "default_dir": "Training_Models_FLUX_2_and_Klein",
    },
    "flux_klein_9b": {
        "repo_id": "MonsterMMORPG/Wan_GGUF",
        "files": [
            {"remote": "FLUX_2_Models/FLUX2-Klein-Base-9B.safetensors", "local": "FLUX2-Klein-Base-9B.safetensors"},
            {"remote": "FLUX_2_Models/FLUX_2_Klein_Train_VAE.safetensors", "local": "FLUX_2_Klein_Train_VAE.safetensors"},
            {"remote": "FLUX_2_Models/qwen_3_8b.safetensors", "local": "qwen_3_8b.safetensors"},
        ],
        "name": "FLUX Klein 9B Training Models",
        "description": "FLUX Klein Base 9B bundle with Klein training VAE and qwen_3_8b",
        "default_dir": "Training_Models_FLUX_2_and_Klein",
    },
    "flux_klein_4b": {
        "repo_id": "MonsterMMORPG/Wan_GGUF",
        "files": [
            {"remote": "FLUX_2_Models/FLUX2-Klein-Base-4B.safetensors", "local": "FLUX2-Klein-Base-4B.safetensors"},
            {"remote": "FLUX_2_Models/FLUX_2_Klein_Train_VAE.safetensors", "local": "FLUX_2_Klein_Train_VAE.safetensors"},
            {"remote": "FLUX_2_Models/qwen_3_4b.safetensors", "local": "qwen_3_4b.safetensors"},
        ],
        "name": "FLUX Klein 4B Training Models",
        "description": "FLUX Klein Base 4B bundle with Klein training VAE and qwen_3_4b",
        "default_dir": "Training_Models_FLUX_2_and_Klein",
    },
    "z_image_base": {
        "repo_id": "MonsterMMORPG/Wan_GGUF",
        "files": [
            {"remote": "FLUX_2_Models/Z_Image_Train_VAE.safetensors", "local": "Z_Image_Train_VAE.safetensors"},
            {"remote": "Z_Image_Training_Text_Encoder.safetensors", "local": "Z_Image_Training_Text_Encoder.safetensors"},
            {"remote": "Z_Image_BF16.safetensors", "local": "Z_Image_BF16.safetensors"},
        ],
        "name": "Z-Image Base Training Models",
        "description": "Z-Image Base bundle (BF16) with Z_Image training VAE and Z_Image_Training_Text_Encoder",
        "default_dir": "Training_Models_Z_Image",
    },
    "z_image_turbo": {
        "repo_id": "MonsterMMORPG/Wan_GGUF",
        "files": [
            {"remote": "FLUX_2_Models/Z_Image_Train_VAE.safetensors", "local": "Z_Image_Train_VAE.safetensors"},
            {"remote": "Z_Image_Training_Text_Encoder.safetensors", "local": "Z_Image_Training_Text_Encoder.safetensors"},
            {"remote": "FLUX_2_Models/zimage_turbo_training_adapter_v2.safetensors", "local": "zimage_turbo_training_adapter_v2.safetensors"},
            {"remote": "Z-Image-Turbo-Models/Z_Image_Turbo_BF16.safetensors", "local": "Z_Image_Turbo_BF16.safetensors"},
        ],
        "name": "Z-Image Turbo Training Models",
        "description": "Z-Image Turbo bundle (BF16) with training VAE, Z_Image_Training_Text_Encoder, and turbo adapter v2",
        "default_dir": "Training_Models_Z_Image",
    },
    "ideogram4": {
        "repo_id": "Comfy-Org/Ideogram-4",
        "files": [
            {"remote": "diffusion_models/ideogram4_fp8_scaled.safetensors", "local": "ideogram4_fp8_scaled.safetensors"},
            {"remote": "text_encoders/qwen3vl_8b_fp8_scaled.safetensors", "local": "qwen3vl_8b_fp8_scaled.safetensors"},
            {"remote": "vae/flux2-vae.safetensors", "local": "flux2-vae.safetensors"},
        ],
        "name": "Ideogram 4 Training Models",
        "description": "Ideogram 4 bundle with FP8 scaled DiT, Qwen3-VL 8B FP8 text encoder, and FLUX.2 VAE",
        "default_dir": "Training_Models_Ideogram_4",
    },
    "krea2": {
        "files": [
            {"repo_id": "krea/Krea-2-Raw", "remote": "raw.safetensors", "local": "Krea_2_Raw_Base.safetensors"},
            {"repo_id": "Comfy-Org/Qwen-Image_ComfyUI", "remote": "split_files/vae/qwen_image_vae.safetensors", "local": "qwen_image_vae.safetensors"},
            {"repo_id": "Comfy-Org/Qwen3-VL", "remote": "text_encoders/qwen3vl_4b_bf16.safetensors", "local": "qwen3vl_4b_bf16.safetensors"},
        ],
        "name": "Krea 2 Training Models",
        "description": "Krea 2 Raw bundle with raw training DiT, Qwen Image VAE, and Qwen3-VL 4B BF16 text encoder",
        "default_dir": "Training_Models_Krea_2",
    }
}

# Interactive menu order
MODEL_MENU = [
    "qwen_image",
    "qwen_image_2512",
    "qwen_image_edit_plus",
    "qwen_image_edit_2511",
    "wan21_t2v",
    "wan22_t2v",
    "wan22_i2v",
    "flux2_dev",
    "flux_klein_9b",
    "flux_klein_4b",
    "z_image_base",
    "z_image_turbo",
    "ideogram4",
    "krea2",
]

DOWNLOAD_CONFIG = {
    "max_retries": 5,
    "retry_delay": 2.0,
    "max_retry_delay": 30.0,
    "metadata_timeout": 30.0,
    "replace_retries": 5,
    "hash_chunk_size": 16 * 1024 * 1024,
    "download_backend": "auto",
    "range_connections": 16,
    "range_read_size": 4 * 1024 * 1024,
    "range_connect_timeout": 30.0,
    "range_read_timeout": 60.0,
    "prefer_parallel_ranges": True,
    "aria2_path": None,
    "aria2_connections": 16,
    "aria2_min_split_size": "20M",
    "aria2_piece_length": "4M",
    "aria2_file_allocation": "none",
    "aria2_disk_cache": "64M",
    "aria2_auto_save_interval": 5,
    "aria2_connect_timeout": 30,
    "aria2_timeout": 60,
    "aria2_max_tries": 5,
    "aria2_retry_wait": 2,
    "non_tty_progress_interval": 5.0,
}

DOWNLOAD_BACKENDS = ("auto", "aria2", "ranges", "hub")
STATE_VERSION = 3
ARIA2_MANIFEST_VERSION = 1
CACHE_DIR_NAME = ".model_download_cache"
HEX_40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+)$", re.IGNORECASE)
MODEL_CHOICE_TOKEN_RE = re.compile(r"([0-9]+)(?:\s*-\s*([0-9]+))?")
ARIA2_VERSION_RE = re.compile(r"aria2\s+version\s+([^\s]+)", re.IGNORECASE)
ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
ARIA2_PROGRESS_RE = re.compile(
    r"\[#(?P<gid>[0-9a-f]+)\s+"
    r"(?P<completed>[^\s/\]]+)/(?P<total>[^\s(\]]+)"
    r"(?:\((?P<percent>[^)]+)\))?"
    r"(?P<details>[^\]]*)\]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RemoteFileInfo:
    repo_id: str
    filename: str
    requested_revision: str
    commit_hash: str
    etag: str
    size: int
    location: str

    def digest_spec(self) -> Tuple[Optional[str], Optional[str]]:
        if HEX_64_RE.fullmatch(self.etag):
            return "sha256", self.etag
        if HEX_40_RE.fullmatch(self.etag):
            return "git-sha1", self.etag
        return None, None


@dataclass(frozen=True)
class DownloadResult:
    status: str
    path: Path
    size: int = 0
    error: Optional[str] = None


class IntegrityError(RuntimeError):
    pass


class RangeNotSupportedError(RuntimeError):
    pass


class ResumeError(RuntimeError):
    pass


class Aria2UnavailableError(RuntimeError):
    pass


class Aria2DownloadError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        exit_code: Optional[int] = None,
        partial_preserved: bool = False,
    ):
        super().__init__(message)
        self.exit_code = exit_code
        self.partial_preserved = partial_preserved


class RobustDownloader:
    """Pinned, resumable downloader with aria2, Python ranges, and Hub fallbacks.

    aria2c is used only as a transfer engine. Repository revisions are still pinned,
    resumable state is content-bound, and Python independently validates final size
    and digest before an atomic install. Existing Python range chunks remain usable.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        target_dir: str = ".",
        revision: str = "main",
        token: Optional[str] = None,
        offline: bool = False,
        force: bool = False,
        reuse_dirs: Optional[List[str]] = None,
    ):
        self.config = dict(config)
        self.target_dir = Path(target_dir).expanduser().resolve()
        self.revision = str(revision or "main").strip() or "main"
        self.token = token
        self.offline = offline
        self.force = force
        self.download_backend = str(
            self.config.get("download_backend", "auto") or "auto"
        ).strip().lower()
        if self.download_backend not in DOWNLOAD_BACKENDS:
            raise ValueError(
                f"Unsupported download backend {self.download_backend!r}; "
                f"choose one of {', '.join(DOWNLOAD_BACKENDS)}"
            )
        self.target_dir.mkdir(parents=True, exist_ok=True)

        self.cache_root = self.target_dir / CACHE_DIR_NAME
        self.stage_root = self.cache_root / "hub"
        self.state_root = self.cache_root / "verified"
        self.aria2_root = self.cache_root / "aria2"
        self.stage_root.mkdir(parents=True, exist_ok=True)
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.aria2_root.mkdir(parents=True, exist_ok=True)

        self._metadata_cache: Dict[Tuple[str, str, str], RemoteFileInfo] = {}
        self._resolved_commits: Dict[str, str] = {}
        self._progress_lock = threading.Lock()
        self._active_progress = False
        self._last_progress_len = 0
        try:
            self._progress_is_interactive = bool(sys.stdout.isatty())
        except (AttributeError, OSError):
            self._progress_is_interactive = False
        self._last_non_tty_progress_at = 0.0
        self._last_non_tty_progress_message: Optional[str] = None
        self._local_index_lock = threading.Lock()
        self._local_file_index: Optional[Dict[int, List[Path]]] = None
        self._local_file_keys: Set[str] = set()
        self._aria2_probe_done = False
        self._aria2_executable: Optional[str] = None
        self._aria2_version: Optional[str] = None
        self._aria2_unavailable_reason: Optional[str] = None

        configured_reuse_dirs = reuse_dirs or []
        if isinstance(configured_reuse_dirs, (str, os.PathLike)):
            configured_reuse_dirs = [configured_reuse_dirs]
        self.reuse_dirs: List[Path] = []
        seen_reuse_dirs: Set[str] = set()
        for directory in [self.target_dir, *configured_reuse_dirs]:
            path = Path(directory).expanduser().resolve()
            key = self._path_identity(path)
            if key not in seen_reuse_dirs:
                seen_reuse_dirs.add(key)
                self.reuse_dirs.append(path)

    # ------------------------------ Console ---------------------------------

    def _get_terminal_width(self) -> int:
        try:
            return shutil.get_terminal_size(fallback=(100, 20)).columns
        except OSError:
            return 100

    def _clear_progress_line_locked(self):
        clear_len = max(self._last_progress_len, self._get_terminal_width())
        sys.stdout.write("\r" + " " * clear_len + "\r")
        sys.stdout.flush()
        self._last_progress_len = 0
        self._active_progress = False

    def show_progress_line(self, message: str):
        with self._progress_lock:
            if not self._progress_is_interactive:
                now = time.monotonic()
                interval = max(
                    0.1,
                    float(self.config.get("non_tty_progress_interval", 5.0)),
                )
                if (
                    self._last_non_tty_progress_message is None
                    or now - self._last_non_tty_progress_at >= interval
                ):
                    print(message, flush=True)
                    self._last_non_tty_progress_at = now
                    self._last_non_tty_progress_message = message
                return

            message = message[:max(1, self._get_terminal_width() - 1)]
            sys.stdout.write("\r" + message)
            if self._last_progress_len > len(message):
                sys.stdout.write(" " * (self._last_progress_len - len(message)))
                sys.stdout.write("\r" + message)
            sys.stdout.flush()
            self._last_progress_len = len(message)
            self._active_progress = True

    def finalize_progress_line(self, message: Optional[str] = None):
        with self._progress_lock:
            if not self._progress_is_interactive:
                if message is not None and message != self._last_non_tty_progress_message:
                    print(message, flush=True)
                self._last_non_tty_progress_at = 0.0
                self._last_non_tty_progress_message = None
                self._last_progress_len = 0
                self._active_progress = False
                return

            if message is not None:
                message = message[:max(1, self._get_terminal_width() - 1)]
                sys.stdout.write("\r" + message)
                if self._last_progress_len > len(message):
                    sys.stdout.write(" " * (self._last_progress_len - len(message)))
                sys.stdout.write("\n")
                sys.stdout.flush()
            elif self._active_progress:
                sys.stdout.write("\n")
                sys.stdout.flush()
            self._last_progress_len = 0
            self._active_progress = False

    def log(self, message: str):
        with self._progress_lock:
            if self._active_progress:
                self._clear_progress_line_locked()
            print(message, flush=True)

    @staticmethod
    def format_bytes(value: float) -> str:
        value = max(0.0, float(value))
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if value < 1024.0 or unit == "TB":
                return f"{value:.1f} {unit}"
            value /= 1024.0
        return f"{value:.1f} TB"

    # -------------------------- Paths and state ------------------------------

    @staticmethod
    def _normalize_local_name(local_filename: str) -> str:
        if not isinstance(local_filename, str) or not local_filename.strip():
            raise ValueError("Local filename must be a non-empty string")
        normalized = local_filename.replace("\\", "/")
        posix_path = PurePosixPath(normalized)
        windows_path = PureWindowsPath(local_filename)
        if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
            raise ValueError(f"Local filename must be relative: {local_filename!r}")
        if any(part in ("", ".", "..") for part in normalized.split("/")):
            raise ValueError(f"Unsafe local filename: {local_filename!r}")
        if posix_path.parts and posix_path.parts[0].casefold() == CACHE_DIR_NAME.casefold():
            raise ValueError(f"Local filename uses reserved directory {CACHE_DIR_NAME!r}")
        return "/".join(posix_path.parts)

    @staticmethod
    def _validate_remote_name(remote_filename: str) -> str:
        if not isinstance(remote_filename, str) or not remote_filename.strip():
            raise ValueError("Remote filename must be a non-empty string")
        if "\\" in remote_filename:
            raise ValueError(f"Remote paths must use '/': {remote_filename!r}")
        path = PurePosixPath(remote_filename)
        if path.is_absolute() or any(part in ("", ".", "..") for part in remote_filename.split("/")):
            raise ValueError(f"Unsafe remote filename: {remote_filename!r}")
        return "/".join(path.parts)

    def resolve_local_path(self, local_filename: str) -> Tuple[str, Path]:
        normalized = self._normalize_local_name(local_filename)
        path = self.target_dir.joinpath(*PurePosixPath(normalized).parts).resolve()
        try:
            inside_target = os.path.commonpath((str(self.target_dir), str(path))) == str(self.target_dir)
        except ValueError:
            inside_target = False
        if not inside_target:
            raise ValueError(f"Local filename escapes target directory: {local_filename!r}")
        return normalized, path

    def _state_path(self, local_name: str) -> Path:
        key = hashlib.sha256(os.path.normcase(local_name).encode("utf-8")).hexdigest()
        return self.state_root / f"{key}.json"

    def _destination_lock_path(self, local_name: str) -> Path:
        return self._state_path(local_name).with_suffix(".lock")

    def _read_state(self, local_name: str) -> Optional[Dict[str, Any]]:
        path = self._state_path(local_name)
        try:
            with path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
            if isinstance(state, dict) and state.get("version") == STATE_VERSION:
                return state
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError) as exc:
            self.log(f"[WARNING] Ignoring invalid verification state {path.name}: {exc}")
        return None

    def _write_json_atomic(self, path: Path, data: Dict[str, Any]):
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
        )
        try:
            with temp_path.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            try:
                temp_path.unlink()
            except OSError:
                pass

    @staticmethod
    def _stat_mtime_ns(stat_result: os.stat_result) -> int:
        return getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000))

    @staticmethod
    def _stat_ctime_ns(stat_result: os.stat_result) -> int:
        return getattr(stat_result, "st_ctime_ns", int(stat_result.st_ctime * 1_000_000_000))

    def _state_matches(
        self,
        state: Optional[Dict[str, Any]],
        local_name: str,
        path: Path,
        repo_id: str,
        remote_filename: str,
        info: Optional[RemoteFileInfo] = None,
    ) -> bool:
        if not state or not path.is_file():
            return False
        try:
            stat_result = path.stat()
        except OSError:
            return False
        common_matches = (
            state.get("local_filename") == local_name
            and state.get("repo_id") == repo_id
            and state.get("remote_filename") == remote_filename
            and state.get("requested_revision") == self.revision
            and state.get("size") == stat_result.st_size
            and state.get("mtime_ns") == self._stat_mtime_ns(stat_result)
            and state.get("ctime_ns") == self._stat_ctime_ns(stat_result)
        )
        if not common_matches:
            return False
        if info is None:
            return True
        return (
            state.get("commit_hash") == info.commit_hash
            and state.get("etag") == info.etag
            and state.get("size") == info.size
        )

    def _mark_verified(
        self,
        info: RemoteFileInfo,
        local_name: str,
        path: Path,
        digest_algorithm: Optional[str],
        digest_value: Optional[str],
    ):
        stat_result = path.stat()
        state = {
            "version": STATE_VERSION,
            "repo_id": info.repo_id,
            "remote_filename": info.filename,
            "local_filename": local_name,
            "requested_revision": self.revision,
            "commit_hash": info.commit_hash,
            "etag": info.etag,
            "size": stat_result.st_size,
            "mtime_ns": self._stat_mtime_ns(stat_result),
            "ctime_ns": self._stat_ctime_ns(stat_result),
            "digest_algorithm": digest_algorithm,
            "digest_value": digest_value,
            "verified_at": time.time(),
        }
        try:
            self._write_json_atomic(self._state_path(local_name), state)
        except OSError as exc:
            self.log(f"[WARNING] Could not save verification state: {exc}")

    # -------------------------- Retry and metadata ---------------------------

    @staticmethod
    def _normalize_etag(etag: str) -> str:
        value = str(etag or "").strip()
        if value.startswith("W/"):
            value = value[2:]
        return value.strip('"').lower()

    @staticmethod
    def _status_code(exc: BaseException) -> Optional[int]:
        response = getattr(exc, "response", None)
        return getattr(response, "status_code", None)

    def _is_retryable(self, exc: BaseException) -> bool:
        permanent = (
            EntryNotFoundError,
            GatedRepoError,
            HFValidationError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
        if isinstance(exc, permanent):
            return False
        if isinstance(exc, RangeNotSupportedError):
            return False
        if isinstance(exc, IntegrityError):
            return True
        if isinstance(exc, HfHubHTTPError):
            status = self._status_code(exc)
            return status is None or status in (408, 409, 425, 429) or status >= 500
        if isinstance(exc, LocalEntryNotFoundError):
            return True
        if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
            return True
        if isinstance(exc, requests.exceptions.HTTPError):
            status = self._status_code(exc)
            return status is None or status in (408, 409, 425, 429) or status >= 500
        if isinstance(exc, OSError) and getattr(exc, "errno", None) in (13, 28, 30):
            return False
        return True

    def _retry_delay(self, attempt: int, exc: BaseException) -> float:
        delay = min(
            float(self.config["retry_delay"]) * (2 ** attempt),
            float(self.config["max_retry_delay"]),
        )
        response = getattr(exc, "response", None)
        retry_after = getattr(response, "headers", {}).get("Retry-After") if response is not None else None
        try:
            delay = max(delay, float(retry_after))
        except (TypeError, ValueError):
            pass
        return min(float(self.config["max_retry_delay"]), delay * random.uniform(0.85, 1.15))

    def _call_with_retries(self, description: str, operation):
        max_retries = max(1, int(self.config["max_retries"]))
        for attempt in range(max_retries):
            try:
                return operation()
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                if attempt == max_retries - 1 or not self._is_retryable(exc):
                    raise
                delay = self._retry_delay(attempt, exc)
                self.log(
                    f"[RETRY] {description} failed ({type(exc).__name__}); "
                    f"retrying in {delay:.1f}s [{attempt + 2}/{max_retries}]"
                )
                time.sleep(delay)
        raise RuntimeError(f"Retry loop ended unexpectedly for {description}")

    @staticmethod
    def _normalize_download_location(base_url: str, location: str) -> str:
        value = urljoin(base_url, str(location or base_url).strip())
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
            raise IntegrityError(f"Hub returned an unsafe download location: {value!r}")
        if "\r" in value or "\n" in value:
            raise IntegrityError("Hub returned a download location containing a newline")
        return value

    def get_remote_info(self, repo_id: str, filename: str) -> RemoteFileInfo:
        filename = self._validate_remote_name(filename)
        key = (repo_id, filename, self.revision)
        if key in self._metadata_cache:
            return self._metadata_cache[key]
        if self.offline:
            raise LocalEntryNotFoundError("Offline mode is enabled")

        lookup_revision = self._resolved_commits.get(repo_id, self.revision)
        url = hf_hub_url(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            revision=lookup_revision,
        )

        def fetch_info() -> RemoteFileInfo:
            metadata = get_hf_file_metadata(
                url,
                token=self.token,
                timeout=float(self.config["metadata_timeout"]),
            )
            etag = self._normalize_etag(getattr(metadata, "etag", ""))
            commit_hash = str(getattr(metadata, "commit_hash", "") or "")
            size = getattr(metadata, "size", None)
            location = self._normalize_download_location(
                url,
                str(getattr(metadata, "location", "") or url),
            )
            if not etag or not commit_hash or not isinstance(size, int) or size < 0:
                raise IntegrityError(
                    f"Hub metadata is incomplete for {repo_id}/{filename} "
                    f"(etag={bool(etag)}, commit={bool(commit_hash)}, size={size!r})"
                )
            resolved_commit = self._resolved_commits.get(repo_id)
            if resolved_commit is not None and commit_hash != resolved_commit:
                raise IntegrityError(
                    f"Repository revision changed while resolving the bundle: "
                    f"{resolved_commit} != {commit_hash}"
                )
            return RemoteFileInfo(
                repo_id=repo_id,
                filename=filename,
                requested_revision=self.revision,
                commit_hash=commit_hash,
                etag=etag,
                size=size,
                location=location,
            )

        info = self._call_with_retries(f"metadata for {filename}", fetch_info)
        self._resolved_commits.setdefault(repo_id, info.commit_hash)
        self._metadata_cache[key] = info
        return info

    def _refresh_download_location(self, info: RemoteFileInfo) -> str:
        """Refresh a potentially expiring CDN/Xet URL without changing content identity."""
        pinned_url = hf_hub_url(
            repo_id=info.repo_id,
            filename=info.filename,
            repo_type="model",
            revision=info.commit_hash,
        )

        def refresh() -> str:
            metadata = get_hf_file_metadata(
                pinned_url,
                token=self.token,
                timeout=float(self.config["metadata_timeout"]),
            )
            etag = self._normalize_etag(getattr(metadata, "etag", ""))
            commit_hash = str(getattr(metadata, "commit_hash", "") or "")
            size = getattr(metadata, "size", None)
            if commit_hash != info.commit_hash or etag != info.etag or size != info.size:
                raise IntegrityError(
                    "Download metadata changed after the repository commit was pinned; "
                    "refusing to resume against a different object"
                )
            return self._normalize_download_location(
                pinned_url,
                str(getattr(metadata, "location", "") or pinned_url),
            )

        return self._call_with_retries(f"fresh download URL for {info.filename}", refresh)

    def get_file_url(self, repo_id: str, filename: str) -> str:
        return hf_hub_url(repo_id, self._validate_remote_name(filename), revision=self.revision)

    def get_file_sha256(self, repo_id: str, filename: str) -> Optional[str]:
        algorithm, value = self.get_remote_info(repo_id, filename).digest_spec()
        return value if algorithm == "sha256" else None

    # ------------------------- Content verification -------------------------

    def _compute_digest(self, path: Path, algorithm: str, display_name: str) -> str:
        file_size = path.stat().st_size
        if algorithm == "git-sha1":
            digest = hashlib.sha1()
            digest.update(f"blob {file_size}\0".encode("ascii"))
        else:
            digest = hashlib.new(algorithm)

        self.log(f"[VERIFYING] {display_name} ({algorithm})")
        bytes_read = 0
        start_time = time.monotonic()
        last_update = 0.0
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(int(self.config["hash_chunk_size"]))
                if not chunk:
                    break
                digest.update(chunk)
                bytes_read += len(chunk)
                now = time.monotonic()
                if now - last_update >= 0.25:
                    percent = bytes_read * 100.0 / file_size if file_size else 100.0
                    speed = bytes_read / max(0.001, now - start_time)
                    self.show_progress_line(
                        f"[VERIFYING] {display_name}: {percent:.1f}% "
                        f"({self.format_bytes(bytes_read)}/{self.format_bytes(file_size)}) "
                        f"{self.format_bytes(speed)}/s"
                    )
                    last_update = now
        value = digest.hexdigest().lower()
        self.finalize_progress_line(f"[HASHED] {display_name}: {value[:16]}...")
        return value

    def _verify_path(
        self,
        path: Path,
        info: RemoteFileInfo,
        display_name: str,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        try:
            actual_size = path.stat().st_size
        except OSError:
            return False, None, None
        if actual_size != info.size:
            self.log(
                f"[WARNING] Size mismatch for {display_name}: "
                f"{self.format_bytes(actual_size)} != {self.format_bytes(info.size)}"
            )
            return False, None, None

        algorithm, expected = info.digest_spec()
        if algorithm is None or expected is None:
            return True, None, None
        try:
            actual = self._compute_digest(path, algorithm, display_name)
        except OSError as exc:
            self.log(f"[ERROR] Could not verify {display_name}: {exc}")
            return False, algorithm, None
        if actual != expected:
            self.log(f"[ERROR] Integrity mismatch for {display_name}")
            self.log(f"  Expected: {expected}")
            self.log(f"  Got:      {actual}")
            return False, algorithm, actual
        return True, algorithm, actual

    def verify_file_sha256(self, filepath: str, expected_sha: str, filename: str = "") -> bool:
        path = Path(filepath)
        try:
            actual = self._compute_digest(path, "sha256", filename or path.name)
        except OSError as exc:
            self.log(f"[ERROR] Could not verify {filename or path.name}: {exc}")
            return False
        return actual == self._normalize_etag(expected_sha)

    # ----------------------------- Download ---------------------------------

    def _existing_is_current(
        self,
        info: RemoteFileInfo,
        local_name: str,
        path: Path,
    ) -> bool:
        if self.force or not path.is_file():
            return False
        state = self._read_state(local_name)
        if self._state_matches(state, local_name, path, info.repo_id, info.filename, info):
            self.log(f"[SKIP] {local_name} is already verified ({self.format_bytes(info.size)})")
            return True
        valid, algorithm, digest = self._verify_path(path, info, local_name)
        if not valid:
            return False
        if algorithm is None:
            self.log(
                f"[WARNING] {local_name} has no content digest in Hub metadata; "
                "it will be refreshed once before being trusted"
            )
            return False
        self._mark_verified(info, local_name, path, algorithm, digest)
        self.log(f"[SKIP] {local_name} matches Hub metadata ({self.format_bytes(info.size)})")
        return True

    def _offline_existing_is_trusted(
        self,
        repo_id: str,
        remote_filename: str,
        local_name: str,
        path: Path,
    ) -> bool:
        state = self._read_state(local_name)
        return self._state_matches(
            state,
            local_name,
            path,
            repo_id,
            remote_filename,
            info=None,
        )

    # --------------------------- Local reuse --------------------------------

    @staticmethod
    def _path_identity(path: Path) -> str:
        return os.path.normcase(os.path.abspath(str(path)))

    @staticmethod
    def _is_download_artifact(filename: str) -> bool:
        folded = filename.casefold()
        return (
            folded.endswith((".parts.json", ".tmp", ".lock"))
            or re.search(r"\.part[0-9]+$", folded) is not None
        )

    def _build_local_file_index(self) -> Dict[int, List[Path]]:
        index: Dict[int, List[Path]] = {}
        keys: Set[str] = set()

        def handle_scan_error(exc: OSError):
            self.log(f"[WARNING] Could not scan a local reuse folder: {exc}")

        for root in self.reuse_dirs:
            if not root.is_dir():
                continue
            for current_dir, dirnames, filenames in os.walk(
                root,
                topdown=True,
                onerror=handle_scan_error,
                followlinks=False,
            ):
                current_path = Path(current_dir)
                dirnames[:] = [
                    dirname
                    for dirname in dirnames
                    if dirname.casefold() != CACHE_DIR_NAME.casefold()
                    and not (current_path / dirname).is_symlink()
                ]
                for filename in filenames:
                    if self._is_download_artifact(filename):
                        continue
                    candidate = current_path / filename
                    try:
                        if candidate.is_symlink() or not candidate.is_file():
                            continue
                        size = candidate.stat().st_size
                    except OSError:
                        continue
                    key = self._path_identity(candidate)
                    if key in keys:
                        continue
                    keys.add(key)
                    index.setdefault(size, []).append(candidate)

        self._local_file_keys = keys
        return index

    def _local_candidates(
        self,
        info: RemoteFileInfo,
        local_name: str,
        destination: Path,
    ) -> List[Path]:
        with self._local_index_lock:
            if self._local_file_index is None:
                self._local_file_index = self._build_local_file_index()
            candidates = list(self._local_file_index.get(info.size, []))

        destination_key = self._path_identity(destination)
        preferred_names = {
            PurePosixPath(local_name).name.casefold(),
            PurePosixPath(info.filename).name.casefold(),
        }
        candidates = [
            candidate
            for candidate in candidates
            if self._path_identity(candidate) != destination_key
        ]
        candidates.sort(
            key=lambda candidate: (
                candidate.name.casefold() not in preferred_names,
                self._path_identity(candidate),
            )
        )
        return candidates

    def _register_local_file(self, path: Path):
        try:
            size = path.stat().st_size
        except OSError:
            return
        key = self._path_identity(path)
        with self._local_index_lock:
            if self._local_file_index is None or key in self._local_file_keys:
                return
            self._local_file_keys.add(key)
            self._local_file_index.setdefault(size, []).append(path)

    def _new_local_copy_stage_path(self, destination: Path) -> Path:
        stage_dir = self.cache_root / "local-copies"
        stage_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(
            f"{destination}|{os.getpid()}|{threading.get_ident()}|{time.time_ns()}".encode("utf-8")
        ).hexdigest()
        return stage_dir / f"{key}.tmp"

    def _try_local_reuse(
        self,
        info: RemoteFileInfo,
        local_name: str,
        destination: Path,
    ) -> Optional[DownloadResult]:
        algorithm, expected_digest = info.digest_spec()
        if algorithm is None or expected_digest is None:
            return None

        for source in self._local_candidates(info, local_name, destination):
            try:
                if source.stat().st_size != info.size:
                    continue
            except OSError:
                continue

            self.log(f"[LOCAL] Trying existing file: {source}")
            staged: Optional[Path] = None
            valid = False
            digest: Optional[str] = None
            try:
                staged = self._new_local_copy_stage_path(destination)
                shutil.copyfile(source, staged)
                valid, verified_algorithm, digest = self._verify_path(
                    staged,
                    info,
                    f"{local_name} (local copy)",
                )
                valid = valid and verified_algorithm == algorithm and digest == expected_digest
            except KeyboardInterrupt:
                raise
            except OSError as exc:
                self.log(f"[WARNING] Could not copy local candidate {source}: {exc}")
            finally:
                if not valid and staged is not None:
                    try:
                        staged.unlink()
                    except OSError:
                        pass

            if not valid:
                self.log(f"[LOCAL] Candidate did not match; leaving it untouched: {source}")
                continue

            if staged is None:
                continue
            try:
                self._atomic_replace(staged, destination)
            finally:
                try:
                    staged.unlink()
                except OSError:
                    pass
            self._mark_verified(info, local_name, destination, algorithm, digest)
            self._cleanup_range_parts(destination)
            self._register_local_file(destination)
            self.log(
                f"[COPIED] {local_name} reused from {source} "
                f"({self.format_bytes(info.size)})"
            )
            return DownloadResult("copied", destination, info.size)

        return None


    # ------------------------------ aria2c ----------------------------------

    @staticmethod
    def _url_origin(url: str) -> Tuple[str, str, int]:
        parsed = urlsplit(url)
        scheme = parsed.scheme.lower()
        host = (parsed.hostname or "").lower()
        default_port = 443 if scheme == "https" else 80
        try:
            port = parsed.port or default_port
        except ValueError as exc:
            raise IntegrityError(f"Invalid download URL port: {url!r}") from exc
        return scheme, host, port

    def _aria2_candidate_paths(self) -> List[str]:
        explicit = self.config.get("aria2_path")
        environment_candidate = os.environ.get("ARIA2C")
        names = ("aria2c.exe", "aria2c") if os.name == "nt" else ("aria2c",)
        candidates: List[str] = []

        if explicit:
            # An explicit CLI/config path is authoritative so a typo is reported
            # instead of silently executing a different binary from PATH.
            candidates.append(str(explicit))
        else:
            if environment_candidate:
                candidates.append(str(environment_candidate))
            directories: List[Path] = []
            try:
                directories.append(Path(__file__).resolve().parent)
            except (NameError, OSError):
                pass
            try:
                directories.append(Path(sys.executable).resolve().parent)
            except OSError:
                pass
            for directory in directories:
                for name in names:
                    candidates.append(str(directory / name))
            for name in names:
                located = shutil.which(name)
                if located:
                    candidates.append(located)

        normalized: List[str] = []
        seen: Set[str] = set()
        for candidate in candidates:
            expanded = os.path.expandvars(os.path.expanduser(candidate))
            located = shutil.which(expanded)
            if located:
                expanded = located
            else:
                path = Path(expanded)
                if not path.is_file():
                    continue
                expanded = str(path.resolve())
            key = os.path.normcase(os.path.abspath(expanded))
            if key not in seen:
                seen.add(key)
                normalized.append(expanded)
        return normalized

    def _probe_aria2(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        if self._aria2_probe_done:
            return (
                self._aria2_executable,
                self._aria2_version,
                self._aria2_unavailable_reason,
            )

        self._aria2_probe_done = True
        explicit = self.config.get("aria2_path")
        environment_candidate = os.environ.get("ARIA2C")
        candidates = self._aria2_candidate_paths()
        if not candidates:
            if explicit:
                reason = f"configured aria2c executable was not found: {explicit}"
            elif environment_candidate:
                reason = (
                    f"ARIA2C pointed to a missing executable ({environment_candidate}); "
                    "no aria2c was found beside the script/executable or on PATH"
                )
            else:
                reason = "aria2c was not found beside the script/executable or on PATH"
            self._aria2_unavailable_reason = reason
            return None, None, reason

        failures: List[str] = []
        for candidate in candidates:
            try:
                completed = subprocess.run(
                    [candidate, "--version"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=10,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                failures.append(f"{candidate}: {exc}")
                continue
            output = completed.stdout or ""
            if completed.returncode != 0:
                failures.append(f"{candidate}: exit code {completed.returncode}")
                continue
            match = ARIA2_VERSION_RE.search(output)
            self._aria2_executable = candidate
            self._aria2_version = match.group(1) if match else "unknown"
            self._aria2_unavailable_reason = None
            return candidate, self._aria2_version, None

        reason = "aria2c could not be started"
        if failures:
            reason += ": " + "; ".join(failures[:3])
        self._aria2_unavailable_reason = reason
        return None, None, reason

    def backend_description(self) -> str:
        if self.download_backend == "hub":
            return "Hub downloader only"
        if self.download_backend == "ranges":
            return (
                f"built-in Python byte ranges ({int(self.config['range_connections'])} connections) "
                "with Hub fallback"
            )

        executable, version, reason = self._probe_aria2()
        if executable:
            fallback = "" if self.download_backend == "aria2" else "; Python ranges and Hub are fallbacks"
            return (
                f"aria2c {version} ({int(self.config['aria2_connections'])} connections, "
                f"{executable}){fallback}"
            )
        if self.download_backend == "aria2":
            return f"aria2c required but unavailable ({reason})"
        return (
            f"built-in Python byte ranges ({int(self.config['range_connections'])} connections); "
            f"aria2c unavailable ({reason}); Hub fallback remains enabled"
        )

    def aria2_available(self) -> bool:
        return self._probe_aria2()[0] is not None

    def _aria2_paths(self, local_name: str) -> Tuple[Path, Path, Path, Path]:
        key = hashlib.sha256(os.path.normcase(local_name).encode("utf-8")).hexdigest()
        stage_dir = self.aria2_root / key
        payload = stage_dir / "payload"
        control = Path(str(payload) + ".aria2")
        manifest = stage_dir / "manifest.json"
        return stage_dir, payload, control, manifest

    def _aria2_manifest_data(self, info: RemoteFileInfo, local_name: str) -> Dict[str, Any]:
        return {
            "version": ARIA2_MANIFEST_VERSION,
            "repo_id": info.repo_id,
            "remote_filename": info.filename,
            "local_filename": local_name,
            "requested_revision": self.revision,
            "commit_hash": info.commit_hash,
            "etag": info.etag,
            "size": info.size,
            "piece_length": str(self.config["aria2_piece_length"]),
        }

    def _discard_aria2_state(self, local_name: str, *, log_cleanup: bool = True):
        stage_dir, payload, control, manifest = self._aria2_paths(local_name)
        reclaimed = 0
        for path in (payload, control, manifest):
            try:
                reclaimed += path.stat().st_size
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                self.log(f"[WARNING] Could not remove aria2 state {path.name}: {exc}")
        try:
            stage_dir.rmdir()
        except OSError:
            pass
        if reclaimed and log_cleanup:
            self.log(f"[CLEANUP] Removed {self.format_bytes(reclaimed)} of aria2 staging data")

    def _cleanup_aria2_state(self, local_name: str):
        self._discard_aria2_state(local_name, log_cleanup=False)

    def _prepare_aria2_state(
        self,
        info: RemoteFileInfo,
        local_name: str,
    ) -> Tuple[Path, Path, Path, Path]:
        stage_dir, payload, control, manifest_path = self._aria2_paths(local_name)
        stage_dir.mkdir(parents=True, exist_ok=True)
        expected = self._aria2_manifest_data(info, local_name)
        manifest: Optional[Dict[str, Any]] = None
        manifest_was_invalid = False
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                manifest = loaded
            else:
                manifest_was_invalid = True
                self.log("[WARNING] aria2 manifest is not a JSON object")
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError) as exc:
            manifest_was_invalid = True
            self.log(f"[WARNING] Ignoring invalid aria2 manifest: {exc}")

        if manifest is not None and any(
            manifest.get(key) != value for key, value in expected.items()
        ):
            self.log(
                "[WARNING] aria2 partial data belongs to a different immutable object; "
                "discarding it before download"
            )
            self._discard_aria2_state(local_name)
            stage_dir, payload, control, manifest_path = self._aria2_paths(local_name)
            if payload.exists() or control.exists() or manifest_path.exists():
                raise ResumeError("Could not remove stale aria2 state safely")
            stage_dir.mkdir(parents=True, exist_ok=True)
            manifest = None

        # A partial payload and .aria2 piece map are safe only when their manifest
        # binds them to the exact repository, commit, ETag, size, and piece length.
        # A full payload without a valid manifest can still be salvaged: remove the
        # untrusted piece map and let the caller independently hash the whole file.
        if manifest is None and (payload.exists() or control.exists()):
            try:
                payload_size = payload.stat().st_size if payload.is_file() else -1
            except OSError:
                payload_size = -1
            if payload_size == info.size:
                reason = "invalid" if manifest_was_invalid else "missing"
                self.log(
                    f"[WARNING] aria2 manifest is {reason}; preserving only the full-sized "
                    "payload for independent verification"
                )
                try:
                    control.unlink()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    raise ResumeError(
                        "Could not remove an untrusted aria2 control file"
                    ) from exc
            else:
                reason = "invalid" if manifest_was_invalid else "missing"
                self.log(
                    f"[WARNING] aria2 partial data has a {reason} content manifest; "
                    "discarding it instead of risking a mixed-file resume"
                )
                self._discard_aria2_state(local_name)
                stage_dir, payload, control, manifest_path = self._aria2_paths(local_name)
                if payload.exists() or control.exists() or manifest_path.exists():
                    raise ResumeError("Could not remove unbound aria2 partial data safely")
                stage_dir.mkdir(parents=True, exist_ok=True)

        if control.exists() and not payload.exists():
            self.log("[WARNING] aria2 control data has no payload; restarting that staged download")
            try:
                control.unlink()
            except OSError as exc:
                raise ResumeError("Could not remove orphaned aria2 control data") from exc

        if payload.exists() and not control.exists():
            try:
                payload_size = payload.stat().st_size
            except OSError:
                payload_size = -1
            if payload_size != info.size:
                self.log(
                    "[WARNING] aria2 payload has no piece map and is incomplete; "
                    "it cannot be resumed safely and will be restarted"
                )
                try:
                    payload.unlink()
                except OSError as exc:
                    raise ResumeError("Could not remove an unsafe aria2 payload") from exc

        self._write_json_atomic(manifest_path, expected)
        if payload.exists() and control.exists():
            self.log(
                "[RESUMING/ARIA2] Found a saved aria2 piece map for the same commit and ETag"
            )
        elif payload.exists():
            self.log("[VERIFYING/ARIA2] Found a complete staged payload without a piece map")
        else:
            self.log(
                f"[DOWNLOADING/ARIA2] Starting up to "
                f"{int(self.config['aria2_connections'])} connections"
            )
        return stage_dir, payload, control, manifest_path

    @staticmethod
    def _aria2_has_resumable_data(payload: Path, control: Path) -> bool:
        """Return True only when aria2 left both a piece map and actual file data."""
        if not payload.is_file() or not control.is_file():
            return False
        try:
            return payload.stat().st_size > 0 and control.stat().st_size > 0
        except OSError:
            return False

    def _aria2_headers(self, info: RemoteFileInfo, download_url: str) -> Dict[str, str]:
        headers = build_hf_headers(
            token=self.token,
            library_name="secourses-model-downloader",
            library_version="3",
        )
        headers["Accept-Encoding"] = "identity"
        pinned_hub_url = hf_hub_url(
            repo_id=info.repo_id,
            filename=info.filename,
            repo_type="model",
            revision=info.commit_hash,
        )
        if self._url_origin(download_url) != self._url_origin(pinned_hub_url):
            for name in list(headers):
                if name.casefold() == "authorization":
                    headers.pop(name, None)
        return {str(name): str(value) for name, value in headers.items()}

    @staticmethod
    def _validate_aria2_input_value(value: str, description: str):
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise IntegrityError(f"Unsafe control character in aria2 {description}")

    def _aria2_input(
        self,
        info: RemoteFileInfo,
        download_url: str,
        headers: Dict[str, str],
    ) -> bytes:
        self._validate_aria2_input_value(download_url, "download URL")
        lines = [download_url, "  out=payload"]
        for name, value in headers.items():
            self._validate_aria2_input_value(name, "header name")
            self._validate_aria2_input_value(value, f"header {name}")
            if ":" in name:
                raise IntegrityError(f"Unsafe aria2 header name: {name!r}")
            lines.append(f"  header={name}: {value}")
        algorithm, digest = info.digest_spec()
        if algorithm == "sha256" and digest:
            lines.append(f"  checksum=sha-256={digest}")
        return ("\n".join(lines) + "\n").encode("utf-8")

    def _aria2_command(self, executable: str, stage_dir: Path) -> List[str]:
        connections = max(1, min(16, int(self.config["aria2_connections"])))
        return [
            executable,
            "--no-conf=true",
            "--no-netrc=true",
            "--input-file=-",
            f"--dir={stage_dir}",
            "--max-concurrent-downloads=1",
            f"--max-connection-per-server={connections}",
            f"--split={connections}",
            f"--min-split-size={self.config['aria2_min_split_size']}",
            f"--piece-length={self.config['aria2_piece_length']}",
            "--continue=true",
            "--always-resume=true",
            "--max-resume-failure-tries=0",
            "--allow-overwrite=true",
            "--auto-file-renaming=false",
            "--allow-piece-length-change=false",
            f"--auto-save-interval={max(1, int(self.config['aria2_auto_save_interval']))}",
            f"--file-allocation={self.config['aria2_file_allocation']}",
            "--enable-mmap=false",
            f"--disk-cache={self.config['aria2_disk_cache']}",
            "--check-integrity=true",
            "--check-certificate=true",
            "--http-accept-gzip=false",
            f"--connect-timeout={max(1, int(self.config['aria2_connect_timeout']))}",
            f"--timeout={max(1, int(self.config['aria2_timeout']))}",
            f"--max-tries={max(1, int(self.config['aria2_max_tries']))}",
            f"--retry-wait={max(0, int(self.config['aria2_retry_wait']))}",
            "--max-file-not-found=1",
            "--summary-interval=1",
            "--show-console-readout=true",
            "--truncate-console-readout=false",
            "--stderr=true",
            "--console-log-level=warn",
            "--download-result=hide",
            "--enable-color=false",
        ]

    @staticmethod
    def _aria2_environment() -> Dict[str, str]:
        environment = dict(os.environ)
        for lowercase, uppercase in (
            ("http_proxy", "HTTP_PROXY"),
            ("https_proxy", "HTTPS_PROXY"),
            ("all_proxy", "ALL_PROXY"),
            ("no_proxy", "NO_PROXY"),
        ):
            if lowercase not in environment and uppercase in environment:
                environment[lowercase] = environment[uppercase]
        return environment

    @staticmethod
    def _format_aria2_progress(line: str) -> Optional[str]:
        """Convert aria2's console readout into a stable, cloud-console-safe line."""
        cleaned = ANSI_ESCAPE_RE.sub("", line).replace("\x00", "").strip()
        match = ARIA2_PROGRESS_RE.search(cleaned)
        if match is None:
            return None

        completed = match.group("completed")
        total = match.group("total")
        percent = (match.group("percent") or "").strip()
        details = match.group("details")
        connections_match = re.search(r"(?:^|\s)CN:(\d+)", details, re.IGNORECASE)
        speed_match = re.search(r"(?:^|\s)DL:([^\s\]]+)", details, re.IGNORECASE)
        eta_match = re.search(r"(?:^|\s)ETA:([^\s\]]+)", details, re.IGNORECASE)

        fields = [f"{completed} / {total}"]
        if percent:
            fields.insert(0, percent)
        if speed_match is not None:
            speed = speed_match.group(1)
            if not speed.casefold().endswith("/s"):
                speed += "/s"
            fields.append(speed)
        if eta_match is not None:
            fields.append(f"ETA {eta_match.group(1)}")
        if connections_match is not None:
            count = int(connections_match.group(1))
            fields.append(f"{count} connection{'s' if count != 1 else ''}")
        return "[ARIA2] " + " | ".join(fields)

    def _handle_aria2_console_line(self, line: str):
        cleaned = ANSI_ESCAPE_RE.sub("", line).replace("\x00", "").strip()
        if not cleaned:
            return

        progress = self._format_aria2_progress(cleaned)
        if progress is not None:
            self.show_progress_line(progress)
            return

        # summary-interval also emits decorative headings and the same file path
        # every second. They add noise but no information beyond the readout.
        if (
            cleaned.startswith("*** Download Progress Summary")
            or cleaned.startswith("FILE:")
            or set(cleaned) <= {"=", "-"}
        ):
            return
        self.log(f"[ARIA2] {cleaned}")

    def _run_aria2_once(
        self,
        executable: str,
        stage_dir: Path,
        input_payload: bytes,
    ) -> int:
        command = self._aria2_command(executable, stage_dir)
        self.finalize_progress_line()
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                env=self._aria2_environment(),
                shell=False,
            )
        except OSError as exc:
            self._aria2_probe_done = False
            raise Aria2UnavailableError(f"Could not start aria2c: {exc}") from exc

        if process.stdin is None or process.stdout is None:
            process.kill()
            process.wait()
            raise Aria2UnavailableError("Could not open aria2c input/output pipes")

        output_queue: "queue.Queue[Optional[bytes]]" = queue.Queue()

        def read_aria2_output():
            buffered = bytearray()
            try:
                while True:
                    value = process.stdout.read(1)
                    if not value:
                        break
                    buffered.extend(value)
                    if value in (b"\r", b"\n") or len(buffered) >= 4096:
                        output_queue.put(bytes(buffered))
                        buffered.clear()
            finally:
                if buffered:
                    output_queue.put(bytes(buffered))
                output_queue.put(None)

        reader = threading.Thread(
            target=read_aria2_output,
            name="aria2-output-reader",
            daemon=True,
        )
        reader.start()

        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        pending_text = ""

        def consume_output(text: str, *, final: bool = False):
            nonlocal pending_text
            pending_text += text
            records = re.split(r"[\r\n]+", pending_text)
            if final:
                pending_text = ""
            else:
                pending_text = records.pop()
            for record in records:
                self._handle_aria2_console_line(record)

        try:
            try:
                remaining = memoryview(input_payload)
                while remaining:
                    written = process.stdin.write(remaining)
                    if not written:
                        raise BrokenPipeError("aria2c closed its input pipe")
                    remaining = remaining[written:]
            except (BrokenPipeError, OSError):
                # Invalid options and very early startup failures can close stdin
                # before Python finishes writing. The captured aria2 output and exit
                # code provide the useful diagnostic, so continue draining them.
                pass
            finally:
                try:
                    process.stdin.close()
                except OSError:
                    pass

            reader_done = False
            while not reader_done:
                try:
                    chunk = output_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if chunk is None:
                    reader_done = True
                else:
                    consume_output(decoder.decode(chunk))
            process.wait()
        except KeyboardInterrupt:
            # Ctrl+C normally reaches aria2 as well. Give it time to flush its .aria2
            # control file before escalating to terminate/kill.
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            reader.join(timeout=2)
            while True:
                try:
                    chunk = output_queue.get_nowait()
                except queue.Empty:
                    break
                if chunk is not None:
                    consume_output(decoder.decode(chunk))
            consume_output(decoder.decode(b"", final=True), final=True)
            try:
                process.stdout.close()
            except OSError:
                pass
            self.finalize_progress_line()
        return int(process.returncode or 0)

    @staticmethod
    def _aria2_exit_message(exit_code: int) -> str:
        messages = {
            1: "unknown aria2 error",
            2: "aria2 timed out",
            3: "resource not found",
            4: "too many resource-not-found responses",
            5: "download speed stayed below the configured limit",
            6: "network failure",
            7: "aria2 stopped with an unfinished download",
            8: "server did not support the required resume request",
            9: "not enough disk space",
            10: "saved aria2 piece length is incompatible",
            11: "the same file is already being downloaded",
            13: "file already existed and could not be reused",
            14: "file rename failed",
            15: "existing file could not be opened",
            16: "file could not be created or truncated",
            17: "file I/O failure",
            18: "directory could not be created",
            19: "DNS resolution failed",
            22: "bad or unexpected HTTP response headers",
            23: "too many redirects",
            24: "HTTP authorization failed or a signed URL expired",
            28: "aria2 rejected an option",
            29: "remote server is temporarily overloaded",
            32: "aria2 checksum validation failed",
        }
        return messages.get(exit_code, f"aria2 exited with code {exit_code}")

    def _download_with_aria2(
        self,
        info: RemoteFileInfo,
        local_name: str,
    ) -> Tuple[Path, Optional[str], Optional[str]]:
        executable, version, reason = self._probe_aria2()
        if not executable:
            raise Aria2UnavailableError(reason or "aria2c is unavailable")

        stage_dir, payload, control, _ = self._prepare_aria2_state(info, local_name)
        if payload.is_file() and not control.exists() and payload.stat().st_size == info.size:
            valid, algorithm, digest = self._verify_path(payload, info, local_name)
            if valid:
                self.log("[ARIA2] Reusing a completed, independently verified staged payload")
                return payload, algorithm, digest
            self._discard_aria2_state(local_name)
            stage_dir, payload, control, _ = self._prepare_aria2_state(info, local_name)

        max_attempts = max(1, int(self.config["max_retries"]))
        retryable_codes = {1, 2, 3, 4, 6, 7, 19, 22, 23, 24, 29}
        for attempt in range(max_attempts):
            download_url = info.location if attempt == 0 else self._refresh_download_location(info)
            headers = self._aria2_headers(info, download_url)
            aria2_input = self._aria2_input(info, download_url, headers)
            action = "Resuming" if control.exists() else "Downloading"
            self.log(
                f"[ARIA2] {action} {local_name} with aria2c {version} "
                f"({int(self.config['aria2_connections'])} connections max)"
            )
            exit_code = self._run_aria2_once(executable, stage_dir, aria2_input)

            if exit_code == 0:
                if not payload.is_file():
                    raise Aria2DownloadError(
                        "aria2 reported success but produced no staged payload",
                        exit_code=exit_code,
                        partial_preserved=False,
                    )
                valid, algorithm, digest = self._verify_path(payload, info, local_name)
                if valid:
                    return payload, algorithm, digest
                self.log(
                    "[WARNING] aria2 completed, but independent verification failed; "
                    "discarding the staged bytes before a clean retry"
                )
                self._discard_aria2_state(local_name)
                if attempt == max_attempts - 1:
                    raise IntegrityError(
                        f"aria2 content failed independent verification for {local_name}"
                    )
                stage_dir, payload, control, _ = self._prepare_aria2_state(info, local_name)
                continue

            partial_preserved = self._aria2_has_resumable_data(payload, control)
            message = self._aria2_exit_message(exit_code)

            # aria2 may create an empty payload/control pair before TLS, DNS, or
            # authorization succeeds. Empty or unpaired state contains no useful
            # bytes, so remove it and let auto mode fall back instead of trapping
            # the user in an aria2-only retry loop.
            if not partial_preserved and (payload.exists() or control.exists()):
                self.log(
                    "[ARIA2] No resumable bytes were produced; removing empty staging state"
                )
                self._discard_aria2_state(local_name)
                if attempt < max_attempts - 1:
                    stage_dir, payload, control, _ = self._prepare_aria2_state(
                        info, local_name
                    )

            if exit_code in (8, 10, 32):
                # Exit 8 means the server cannot resume the saved payload. Keeping it
                # would create a permanent failure loop. Exit 10/32 likewise identifies
                # state that cannot be trusted for the next attempt.
                self.log(f"[WARNING] {message}; restarting the aria2 stage cleanly")
                self._discard_aria2_state(local_name)
                partial_preserved = False
                if attempt < max_attempts - 1:
                    stage_dir, payload, control, _ = self._prepare_aria2_state(info, local_name)
                    continue

            if exit_code in retryable_codes and attempt < max_attempts - 1:
                error = Aria2DownloadError(
                    message,
                    exit_code=exit_code,
                    partial_preserved=partial_preserved,
                )
                delay = self._retry_delay(attempt, error)
                self.log(
                    f"[RETRY/ARIA2] {message}; refreshing the signed URL and retrying in "
                    f"{delay:.1f}s [{attempt + 2}/{max_attempts}]"
                )
                time.sleep(delay)
                continue

            raise Aria2DownloadError(
                message,
                exit_code=exit_code,
                partial_preserved=partial_preserved,
            )

        raise Aria2DownloadError(
            f"aria2 retry loop ended unexpectedly for {local_name}",
            partial_preserved=self._aria2_has_resumable_data(payload, control),
        )

    def _repo_stage_dir(self, repo_id: str) -> Path:
        repo_key = hashlib.sha256(repo_id.encode("utf-8")).hexdigest()[:20]
        path = self.stage_root / repo_key
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _download_to_stage(
        self,
        info: RemoteFileInfo,
        display_name: str,
    ) -> Tuple[Path, Optional[str], Optional[str]]:
        max_retries = max(1, int(self.config["max_retries"]))
        force_download = self.force
        stage_dir = self._repo_stage_dir(info.repo_id)

        for attempt in range(max_retries):
            try:
                staged = Path(
                    hf_hub_download(
                        repo_id=info.repo_id,
                        filename=info.filename,
                        repo_type="model",
                        revision=info.commit_hash,
                        local_dir=str(stage_dir),
                        token=self.token,
                        etag_timeout=float(self.config["metadata_timeout"]),
                        force_download=force_download,
                        local_files_only=False,
                        library_name="secourses-model-downloader",
                        library_version="3",
                    )
                )
                valid, algorithm, digest = self._verify_path(staged, info, display_name)
                if not valid:
                    try:
                        staged.unlink()
                    except FileNotFoundError:
                        pass
                    force_download = True
                    raise IntegrityError(f"Downloaded content failed verification for {display_name}")
                return staged, algorithm, digest
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                if not isinstance(exc, IntegrityError):
                    force_download = False
                if attempt == max_retries - 1 or not self._is_retryable(exc):
                    raise
                delay = self._retry_delay(attempt, exc)
                self.log(
                    f"[RETRY] Download failed for {display_name} ({type(exc).__name__}); "
                    f"resuming in {delay:.1f}s [{attempt + 2}/{max_retries}]"
                )
                time.sleep(delay)
        raise RuntimeError(f"Download retry loop ended unexpectedly for {display_name}")

    def _atomic_replace(self, staged: Path, destination: Path):
        destination.parent.mkdir(parents=True, exist_ok=True)
        attempts = max(1, int(self.config["replace_retries"]))
        for attempt in range(attempts):
            try:
                os.replace(staged, destination)
                return
            except PermissionError:
                if attempt == attempts - 1:
                    raise
                delay = min(0.25 * (2 ** attempt), 2.0)
                self.log(
                    f"[RETRY] {destination.name} is temporarily locked; "
                    f"retrying replacement in {delay:.2f}s"
                )
                time.sleep(delay)

    def _range_layout(self, info: RemoteFileInfo) -> List[Tuple[int, int, int]]:
        connections = max(1, int(self.config["range_connections"]))
        connections = min(connections, max(1, info.size))
        base_size = info.size // connections
        chunks = []
        for index in range(connections):
            start = index * base_size
            end = info.size - 1 if index == connections - 1 else (index + 1) * base_size - 1
            chunks.append((index, start, end))
        return chunks

    def _legacy_chunk_paths(self, destination: Path, count: Optional[int] = None) -> List[Path]:
        count = count or max(1, int(self.config["range_connections"]))
        return [Path(f"{destination}.part{index}") for index in range(count)]

    @staticmethod
    def _part_manifest_path(destination: Path) -> Path:
        return Path(f"{destination}.parts.json")

    def _part_manifest_data(self, info: RemoteFileInfo, local_name: str) -> Dict[str, Any]:
        return {
            "version": 1,
            "repo_id": info.repo_id,
            "remote_filename": info.filename,
            "local_filename": local_name,
            "requested_revision": self.revision,
            "commit_hash": info.commit_hash,
            "etag": info.etag,
            "size": info.size,
            "connections": len(self._range_layout(info)),
        }

    def _read_part_manifest(self, destination: Path) -> Optional[Dict[str, Any]]:
        path = self._part_manifest_path(destination)
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else None
        except FileNotFoundError:
            return None
        except (OSError, ValueError, TypeError) as exc:
            self.log(f"[WARNING] Ignoring invalid resume manifest {path.name}: {exc}")
            return None

    def _discard_range_parts(self, destination: Path):
        for path in self._legacy_chunk_paths(destination):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        try:
            self._part_manifest_path(destination).unlink()
        except FileNotFoundError:
            pass

    def _prepare_range_resume(
        self,
        info: RemoteFileInfo,
        local_name: str,
        destination: Path,
    ) -> bool:
        layout = self._range_layout(info)
        chunk_paths = self._legacy_chunk_paths(destination, len(layout))
        has_saved_chunks = any(path.is_file() for path in chunk_paths)

        expected_manifest = self._part_manifest_data(info, local_name)
        manifest = self._read_part_manifest(destination)
        if manifest is not None and any(
            manifest.get(key) != value for key, value in expected_manifest.items()
        ):
            self.log(
                "[WARNING] Partial chunks belong to a different repository revision; "
                "they cannot be mixed with the requested content and will be restarted."
            )
            self._discard_range_parts(destination)
            manifest = None
            has_saved_chunks = False

        resumed_bytes = 0
        for (index, start, end), path in zip(layout, chunk_paths):
            expected_size = end - start + 1
            if not path.exists():
                continue
            actual_size = path.stat().st_size
            if actual_size > expected_size:
                self.log(f"[WARNING] Chunk {index + 1} is oversized; restarting only that chunk")
                path.unlink()
                continue
            resumed_bytes += actual_size

        self._write_json_atomic(self._part_manifest_path(destination), expected_manifest)
        if resumed_bytes:
            source = "legacy chunks" if manifest is None and has_saved_chunks else "saved chunks"
            self.log(
                f"[RESUMING] Reusing {self.format_bytes(resumed_bytes)} from {source}; "
                "each connection continues from its exact byte offset"
            )
        else:
            self.log(
                f"[DOWNLOADING] Starting {len(layout)} independently resumable byte ranges"
            )
        return True

    def _download_range_chunk(
        self,
        info: RemoteFileInfo,
        url: str,
        base_headers: Dict[str, str],
        start: int,
        end: int,
        chunk_path: Path,
        chunk_index: int,
        chunk_count: int,
        cancel_event: threading.Event,
        progress_callback,
    ) -> bool:
        expected_size = end - start + 1
        max_retries = max(1, int(self.config["max_retries"]))

        for attempt in range(max_retries):
            if cancel_event.is_set():
                raise ResumeError("Range resume was cancelled")
            try:
                existing_size = chunk_path.stat().st_size if chunk_path.exists() else 0
                if existing_size > expected_size:
                    chunk_path.unlink()
                    existing_size = 0
                if existing_size == expected_size:
                    progress_callback(chunk_index, existing_size)
                    return True

                request_start = start + existing_size
                headers = dict(base_headers)
                headers["Range"] = f"bytes={request_start}-{end}"
                headers["Accept-Encoding"] = "identity"
                timeout = (
                    float(self.config["range_connect_timeout"]),
                    float(self.config["range_read_timeout"]),
                )

                with requests.Session() as session:
                    with session.get(
                        url,
                        headers=headers,
                        stream=True,
                        allow_redirects=True,
                        timeout=timeout,
                    ) as response:
                        if response.status_code == 200:
                            raise RangeNotSupportedError(
                                "The server ignored the Range header and returned the full file"
                            )
                        if response.status_code != 206:
                            response.raise_for_status()
                            raise ResumeError(f"Unexpected HTTP status {response.status_code}")

                        content_range = response.headers.get("Content-Range", "")
                        match = CONTENT_RANGE_RE.fullmatch(content_range.strip())
                        if match is None:
                            raise IntegrityError(f"Missing or invalid Content-Range: {content_range!r}")
                        returned_start, returned_end, returned_total = map(int, match.groups())
                        if (returned_start, returned_end, returned_total) != (
                            request_start,
                            end,
                            info.size,
                        ):
                            raise IntegrityError(
                                "Range response does not match the requested bytes: "
                                f"{content_range!r}"
                            )

                        encoding = response.headers.get("Content-Encoding", "identity").lower()
                        if encoding not in ("", "identity"):
                            raise IntegrityError(f"Unexpected Content-Encoding for range: {encoding}")
                        response_size = end - request_start + 1
                        content_length = response.headers.get("Content-Length")
                        if content_length is not None and int(content_length) != response_size:
                            raise IntegrityError(
                                f"Range Content-Length is {content_length}, expected {response_size}"
                            )

                        mode = "ab" if existing_size else "wb"
                        downloaded = existing_size
                        with chunk_path.open(mode) as output:
                            for data in response.iter_content(
                                chunk_size=int(self.config["range_read_size"])
                            ):
                                if cancel_event.is_set():
                                    raise ResumeError("Range resume was cancelled")
                                if not data:
                                    continue
                                if downloaded + len(data) > expected_size:
                                    raise IntegrityError("Range response exceeded its chunk boundary")
                                output.write(data)
                                downloaded += len(data)
                                progress_callback(chunk_index, downloaded)
                            output.flush()

                final_size = chunk_path.stat().st_size
                if final_size != expected_size:
                    raise requests.exceptions.ChunkedEncodingError(
                        f"Chunk ended early at {final_size}/{expected_size} bytes"
                    )
                progress_callback(chunk_index, final_size)
                return True
            except KeyboardInterrupt:
                raise
            except RangeNotSupportedError:
                raise
            except Exception as exc:
                final_size = chunk_path.stat().st_size if chunk_path.exists() else 0
                progress_callback(chunk_index, min(final_size, expected_size))
                if cancel_event.is_set():
                    raise ResumeError("Range resume was cancelled") from exc
                if attempt == max_retries - 1 or not self._is_retryable(exc):
                    raise ResumeError(
                        f"Chunk {chunk_index + 1}/{chunk_count} failed at exact offset "
                        f"{start + final_size}: {exc}"
                    ) from exc
                delay = self._retry_delay(attempt, exc)
                self.log(
                    f"[RETRY] Chunk {chunk_index + 1}/{chunk_count} stopped at "
                    f"{self.format_bytes(final_size)}/{self.format_bytes(expected_size)}; "
                    f"continuing in {delay:.1f}s"
                )
                time.sleep(delay)
        return False

    def _merge_complete_range_parts(
        self,
        info: RemoteFileInfo,
        local_name: str,
        destination: Path,
    ) -> Optional[Tuple[Path, str, str]]:
        algorithm, expected_digest = info.digest_spec()
        if algorithm is None or expected_digest is None:
            raise ResumeError("Hub metadata does not provide a digest for resumed chunks")

        layout = self._range_layout(info)
        chunk_paths = self._legacy_chunk_paths(destination, len(layout))
        for (_, start, end), path in zip(layout, chunk_paths):
            if not path.is_file() or path.stat().st_size != end - start + 1:
                raise ResumeError(f"Chunk {path.name} is incomplete before merge")

        recovery_dir = self.cache_root / "range_merge"
        recovery_dir.mkdir(parents=True, exist_ok=True)
        recovery_key = hashlib.sha256(local_name.encode("utf-8")).hexdigest()
        recovered_path = recovery_dir / f"{recovery_key}.{info.etag}.incomplete"
        if algorithm == "git-sha1":
            digest = hashlib.sha1()
            digest.update(f"blob {info.size}\0".encode("ascii"))
        else:
            digest = hashlib.new(algorithm)

        self.log("[MERGING] Combining resumed chunks and verifying their digest")
        bytes_copied = 0
        start_time = time.monotonic()
        last_update = 0.0
        try:
            with recovered_path.open("wb") as output:
                for chunk_path in chunk_paths:
                    with chunk_path.open("rb") as source:
                        while True:
                            data = source.read(int(self.config["hash_chunk_size"]))
                            if not data:
                                break
                            output.write(data)
                            digest.update(data)
                            bytes_copied += len(data)
                            now = time.monotonic()
                            if now - last_update >= 0.25:
                                speed = bytes_copied / max(0.001, now - start_time)
                                self.show_progress_line(
                                    f"[MERGING] {local_name}: "
                                    f"{bytes_copied * 100.0 / info.size:.1f}% "
                                    f"({self.format_bytes(bytes_copied)}/{self.format_bytes(info.size)}) "
                                    f"{self.format_bytes(speed)}/s"
                                )
                                last_update = now
                output.flush()
                os.fsync(output.fileno())
        except KeyboardInterrupt:
            self.finalize_progress_line()
            raise
        except OSError as exc:
            self.finalize_progress_line()
            try:
                recovered_path.unlink()
            except OSError:
                pass
            raise ResumeError(f"Could not merge resumed chunks: {exc}") from exc

        actual_digest = digest.hexdigest().lower()
        if bytes_copied != info.size or actual_digest != expected_digest:
            self.finalize_progress_line("[ERROR] Resumed chunks failed final integrity verification")
            try:
                recovered_path.unlink()
            except OSError:
                pass
            return None

        self.finalize_progress_line(f"[VERIFIED] {local_name}: {actual_digest[:16]}...")
        return recovered_path, algorithm, actual_digest

    def _resume_range_parts(
        self,
        info: RemoteFileInfo,
        local_name: str,
        destination: Path,
    ) -> Optional[Tuple[Path, str, str]]:
        if info.digest_spec()[0] is None:
            return None
        has_existing_parts = any(
            path.is_file() for path in self._legacy_chunk_paths(destination)
        )
        if not self.config.get("prefer_parallel_ranges", True) and not has_existing_parts:
            return None
        if not self._prepare_range_resume(info, local_name, destination):
            return None

        layout = self._range_layout(info)
        chunk_paths = self._legacy_chunk_paths(destination, len(layout))
        progress = {}
        progress_lock = threading.Lock()
        cancel_event = threading.Event()
        for (index, start, end), path in zip(layout, chunk_paths):
            expected_size = end - start + 1
            progress[index] = min(path.stat().st_size, expected_size) if path.exists() else 0

        def update_progress(index: int, downloaded: int):
            with progress_lock:
                progress[index] = downloaded

        url = hf_hub_url(
            repo_id=info.repo_id,
            filename=info.filename,
            repo_type="model",
            revision=info.commit_hash,
        )
        headers = build_hf_headers(
            token=self.token,
            library_name="secourses-model-downloader",
            library_version="3",
        )
        initial_bytes = sum(progress.values())
        progress_action = "RESUMING" if initial_bytes else "DOWNLOADING"
        started = time.monotonic()
        last_update = 0.0
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(layout))
        futures = []
        try:
            for (index, start, end), path in zip(layout, chunk_paths):
                if progress[index] == end - start + 1:
                    continue
                futures.append(
                    executor.submit(
                        self._download_range_chunk,
                        info,
                        url,
                        headers,
                        start,
                        end,
                        path,
                        index,
                        len(layout),
                        cancel_event,
                        update_progress,
                    )
                )

            pending = set(futures)
            while pending:
                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=0.5,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    future.result()
                now = time.monotonic()
                if now - last_update >= 0.5:
                    with progress_lock:
                        current = sum(progress.values())
                    speed = (current - initial_bytes) / max(0.001, now - started)
                    self.show_progress_line(
                        f"[{progress_action}] {local_name}: {current * 100.0 / info.size:.1f}% "
                        f"({self.format_bytes(current)}/{self.format_bytes(info.size)}) "
                        f"{self.format_bytes(speed)}/s"
                    )
                    last_update = now
        except BaseException:
            cancel_event.set()
            for future in futures:
                future.cancel()
            self.finalize_progress_line()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        completed_action = "RESUMED" if initial_bytes else "DOWNLOADED"
        self.finalize_progress_line(
            f"[{completed_action}] {local_name}: all {len(layout)} byte ranges are complete"
        )
        merged = self._merge_complete_range_parts(info, local_name, destination)
        if merged is None:
            self.log(
                "[WARNING] The saved range bytes did not belong to the current file. "
                "Only the invalid chunks will be discarded; a clean download will follow."
            )
            self._discard_range_parts(destination)
        return merged

    def _cleanup_range_parts(self, destination: Path):
        reclaimed = 0
        pattern = re.compile(re.escape(destination.name) + r"\.part\d+$")
        candidates = list(destination.parent.glob(destination.name + ".part*"))
        candidates.append(Path(str(destination) + ".tmp"))
        for candidate in candidates:
            if candidate.name != destination.name + ".tmp" and not pattern.fullmatch(candidate.name):
                continue
            try:
                reclaimed += candidate.stat().st_size
                candidate.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                self.log(f"[WARNING] Could not remove old partial file {candidate.name}: {exc}")
        manifest_path = self._part_manifest_path(destination)
        try:
            reclaimed += manifest_path.stat().st_size
            manifest_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            self.log(f"[WARNING] Could not remove resume manifest {manifest_path.name}: {exc}")
        if reclaimed:
            self.log(f"[CLEANUP] Removed {self.format_bytes(reclaimed)} of range-part data")

    def _friendly_error(self, exc: BaseException) -> str:
        status = self._status_code(exc)
        if isinstance(exc, GatedRepoError):
            return "The repository is gated. Accept its terms and run `hf auth login`."
        if isinstance(exc, RepositoryNotFoundError) or status == 401:
            return "Repository not found or authentication failed. Run `hf auth login` if it is private."
        if isinstance(exc, RevisionNotFoundError):
            return f"Revision {self.revision!r} does not exist."
        if isinstance(exc, EntryNotFoundError):
            return "The configured file does not exist in the repository."
        if isinstance(exc, requests.exceptions.SSLError):
            return "TLS verification failed. Check the system clock, CA certificates, proxy, or firewall."
        if isinstance(exc, requests.exceptions.Timeout):
            return "The connection timed out repeatedly. Re-run to resume the partial download."
        if isinstance(exc, requests.exceptions.ConnectionError):
            return "Could not reach Hugging Face. Check the connection, proxy, DNS, or firewall."
        if status == 429:
            return "Hugging Face rate-limited the requests. Wait briefly, then re-run to resume."
        if status is not None and status >= 500:
            return f"Hugging Face returned HTTP {status}. Re-run later to resume."
        if isinstance(exc, Aria2UnavailableError):
            return (
                f"aria2c is unavailable: {exc}. Install aria2c, put aria2c.exe beside "
                "the script on Windows, pass --aria2-path, or use --backend ranges."
            )
        if isinstance(exc, Aria2DownloadError):
            suffix = " Partial aria2 data was preserved for the next run." if exc.partial_preserved else ""
            return f"aria2c download failed: {exc}.{suffix}"
        if isinstance(exc, IntegrityError):
            return f"Integrity verification failed after all retries: {exc}"
        if isinstance(exc, PermissionError):
            return "The destination is locked or not writable. Close programs using the model file and retry."
        if isinstance(exc, OSError) and getattr(exc, "errno", None) == 28:
            return "Not enough free disk space for the download."
        return str(exc) or type(exc).__name__

    def ensure_file(
        self,
        repo_id: str,
        remote_filename: str,
        local_filename: str,
    ) -> DownloadResult:
        try:
            remote_filename = self._validate_remote_name(remote_filename)
            local_name, destination = self.resolve_local_path(local_filename)
        except ValueError as exc:
            return DownloadResult("failed", self.target_dir, error=str(exc))

        try:
            with WeakFileLock(self._destination_lock_path(local_name)):
                return self._ensure_file_locked(
                    repo_id,
                    remote_filename,
                    local_name,
                    destination,
                )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            return DownloadResult("failed", destination, error=self._friendly_error(exc))

    def _ensure_file_locked(
        self,
        repo_id: str,
        remote_filename: str,
        local_name: str,
        destination: Path,
    ) -> DownloadResult:
        if self.offline:
            if not self.force and self._offline_existing_is_trusted(
                repo_id, remote_filename, local_name, destination
            ):
                self.log(f"[SKIP/OFFLINE] Using previously verified {local_name}")
                return DownloadResult("skipped", destination, destination.stat().st_size)
            return DownloadResult(
                "failed",
                destination,
                error="Offline mode has no matching previously verified local file.",
            )

        try:
            info = self.get_remote_info(repo_id, remote_filename)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if self._is_retryable(exc) and not self.force and self._offline_existing_is_trusted(
                repo_id, remote_filename, local_name, destination
            ):
                self.log(
                    f"[WARNING] Hub is unavailable; using previously verified {local_name} "
                    f"for revision {self.revision!r}"
                )
                return DownloadResult("skipped", destination, destination.stat().st_size)
            return DownloadResult("failed", destination, error=self._friendly_error(exc))

        self.log(
            f"[REMOTE] {remote_filename} | {self.format_bytes(info.size)} | "
            f"commit {info.commit_hash[:12]}"
        )
        if self._existing_is_current(info, local_name, destination):
            self._cleanup_range_parts(destination)
            self._cleanup_aria2_state(local_name)
            return DownloadResult("skipped", destination, info.size)

        self.log("[LOCAL] Checking configured model folders for a verified copy...")
        local_result = self._try_local_reuse(info, local_name, destination)
        if local_result is not None:
            self._cleanup_aria2_state(local_name)
            return local_result

        has_legacy_ranges = any(
            path.is_file() for path in destination.parent.glob(destination.name + ".part*")
        ) or self._part_manifest_path(destination).is_file()
        range_fallback_allowed = self.download_backend in ("auto", "ranges")

        # Existing .partN files are finished with the original engine first so an
        # upgrade never throws away already downloaded bytes.
        if has_legacy_ranges and self.download_backend != "hub":
            self.log(
                "[RESUME] Existing Python range chunks detected; finishing them before "
                "starting any new backend"
            )
            try:
                resumed = self._resume_range_parts(info, local_name, destination)
            except RangeNotSupportedError as exc:
                self.log(f"[WARNING] Byte-range resume is unavailable: {exc}")
                self.log("[FALLBACK] Keeping the old chunks while the Hub downloader takes over")
                range_fallback_allowed = False
                resumed = None
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                return DownloadResult(
                    "failed",
                    destination,
                    info.size,
                    f"Resume stopped safely with partial bytes preserved: {self._friendly_error(exc)}",
                )

            if resumed is not None:
                staged, algorithm, digest = resumed
                try:
                    self._atomic_replace(staged, destination)
                    self._mark_verified(info, local_name, destination, algorithm, digest)
                    self._cleanup_range_parts(destination)
                    self._cleanup_aria2_state(local_name)
                    self._register_local_file(destination)
                    self.log(f"[OK] {local_name} ready with exact byte-range resume enabled")
                    return DownloadResult("downloaded", destination, info.size)
                except OSError as exc:
                    return DownloadResult("failed", destination, info.size, self._friendly_error(exc))

        if self.download_backend in ("auto", "aria2") and not has_legacy_ranges:
            try:
                staged, algorithm, digest = self._download_with_aria2(info, local_name)
                self._atomic_replace(staged, destination)
                self._mark_verified(info, local_name, destination, algorithm, digest)
                self._cleanup_aria2_state(local_name)
                self._cleanup_range_parts(destination)
                self._register_local_file(destination)
                self.log(
                    f"[OK] {local_name} ready via aria2c with independent final verification"
                )
                return DownloadResult("downloaded", destination, info.size)
            except KeyboardInterrupt:
                raise
            except Aria2UnavailableError as exc:
                if self.download_backend == "aria2":
                    return DownloadResult("failed", destination, info.size, self._friendly_error(exc))
                self.log(f"[FALLBACK] aria2c unavailable: {exc}")
            except Aria2DownloadError as exc:
                no_fallback_codes = {9, 11, 13, 14, 15, 16, 17, 18, 28}
                if (
                    self.download_backend == "aria2"
                    or exc.partial_preserved
                    or exc.exit_code in no_fallback_codes
                ):
                    return DownloadResult(
                        "failed",
                        destination,
                        info.size,
                        self._friendly_error(exc),
                    )
                self.log(f"[FALLBACK] aria2c did not create resumable data: {exc}")
            except Exception as exc:
                return DownloadResult("failed", destination, info.size, self._friendly_error(exc))

        if range_fallback_allowed and not has_legacy_ranges:
            try:
                resumed = self._resume_range_parts(info, local_name, destination)
            except RangeNotSupportedError as exc:
                self.log(f"[WARNING] Byte-range resume is unavailable: {exc}")
                self.log("[FALLBACK] The maintained Hub downloader will take over")
                resumed = None
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                return DownloadResult(
                    "failed",
                    destination,
                    info.size,
                    f"Resume stopped safely with partial bytes preserved: {self._friendly_error(exc)}",
                )

            if resumed is not None:
                staged, algorithm, digest = resumed
                try:
                    self._atomic_replace(staged, destination)
                    self._mark_verified(info, local_name, destination, algorithm, digest)
                    self._cleanup_range_parts(destination)
                    self._cleanup_aria2_state(local_name)
                    self._register_local_file(destination)
                    self.log(f"[OK] {local_name} ready with exact byte-range resume enabled")
                    return DownloadResult("downloaded", destination, info.size)
                except OSError as exc:
                    return DownloadResult("failed", destination, info.size, self._friendly_error(exc))

        self.log(f"[DOWNLOAD/HUB] {remote_filename} -> {local_name}")
        try:
            staged, algorithm, digest = self._download_to_stage(info, local_name)
            self._atomic_replace(staged, destination)
            self._mark_verified(info, local_name, destination, algorithm, digest)
            self._cleanup_range_parts(destination)
            self._cleanup_aria2_state(local_name)
            self._register_local_file(destination)
            self.log(f"[OK] {local_name} ready ({self.format_bytes(info.size)})")
            return DownloadResult("downloaded", destination, info.size)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            return DownloadResult("failed", destination, info.size, self._friendly_error(exc))

    def download_file(self, repo_id: str, filename: str, local_dir: str) -> bool:
        if Path(local_dir).expanduser().resolve() != self.target_dir:
            raise ValueError("RobustDownloader target_dir must match local_dir")
        return self.ensure_file(repo_id, filename, filename).status != "failed"

def parse_model_file_specs(file_specs) -> List[Tuple[str, str]]:
    """
    Normalize model config file specs to a list of (remote_path, local_filename).

    Supported formats:
      - "filename.ext" (remote == local)
      - {"remote": "path/in/repo.ext", "local": "local_name.ext", "repo_id": "optional/repo"}
      - ("remote_path", "local_name")
    """
    items: List[Tuple[str, str]] = []
    for spec in file_specs or []:
        if isinstance(spec, str):
            items.append((spec, spec))
            continue

        if isinstance(spec, dict):
            remote = spec.get("remote")
            local = spec.get("local")
            if not isinstance(remote, str) or not remote:
                raise ValueError(f"Invalid file spec (missing 'remote'): {spec}")
            if not local:
                local = PurePosixPath(remote).name
            if not isinstance(local, str):
                raise ValueError(f"Invalid file spec ('local' must be a string): {spec}")
            items.append((remote, local))
            continue

        if isinstance(spec, (list, tuple)) and len(spec) == 2:
            remote, local = spec
            if not isinstance(remote, str) or not isinstance(local, str):
                raise ValueError(f"Invalid file spec (paths must be strings): {spec}")
            items.append((remote, local))
            continue

        raise ValueError(f"Unsupported file spec format: {spec}")

    seen = set()
    local_owners: Dict[str, str] = {}
    deduped: List[Tuple[str, str]] = []
    for remote, local in items:
        remote = RobustDownloader._validate_remote_name(remote)
        local = RobustDownloader._normalize_local_name(local)
        key = (remote, local)
        if key in seen:
            continue
        collision_key = local.casefold()
        previous_remote = local_owners.get(collision_key)
        if previous_remote is not None and previous_remote != remote:
            raise ValueError(
                f"Multiple remote files map to the same local path {local!r}: "
                f"{previous_remote!r} and {remote!r}"
            )
        seen.add(key)
        local_owners[collision_key] = remote
        deduped.append((remote, local))
    return deduped


def parse_model_download_specs(model_config: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """Return normalized (repository, remote path, local path) download items."""
    default_repo_id = model_config.get("repo_id")
    items: List[Tuple[str, str, str]] = []

    for spec in model_config.get("files", []) or []:
        repo_id = spec.get("repo_id", default_repo_id) if isinstance(spec, dict) else default_repo_id
        if not isinstance(repo_id, str) or not repo_id.strip():
            raise ValueError(f"File spec does not define a Hugging Face repository: {spec}")
        repo_id = repo_id.strip()
        for remote, local in parse_model_file_specs([spec]):
            items.append((repo_id, remote, local))

    seen = set()
    local_owners: Dict[str, Tuple[str, str]] = {}
    deduped: List[Tuple[str, str, str]] = []
    for repo_id, remote, local in items:
        key = (repo_id, remote, local)
        if key in seen:
            continue
        collision_key = local.casefold()
        source = (repo_id, remote)
        previous_source = local_owners.get(collision_key)
        if previous_source is not None and previous_source != source:
            raise ValueError(
                f"Multiple remote files map to the same local path {local!r}: "
                f"{previous_source[0]!r}/{previous_source[1]!r} and {repo_id!r}/{remote!r}"
            )
        seen.add(key)
        local_owners[collision_key] = source
        deduped.append(key)
    return deduped


def download_file_with_rename(downloader: RobustDownloader, repo_id: str, remote_filename: str,
                              local_dir: str, local_filename: str) -> bool:
    """Compatibility wrapper for callers that used the old helper."""
    if Path(local_dir).expanduser().resolve() != downloader.target_dir:
        downloader.log("[ERROR] Downloader target directory does not match local_dir")
        return False
    result = downloader.ensure_file(repo_id, remote_filename, local_filename)
    if result.status == "failed":
        downloader.log(f"[FAILED] {local_filename}: {result.error}")
        return False
    return True

def _get_terminal_width(fallback: int = 140) -> int:
    try:
        return shutil.get_terminal_size(fallback=(fallback, 20)).columns
    except Exception:
        return fallback

def render_available_models_table() -> str:
    entries = []
    for i, model_id in enumerate(MODEL_MENU, 1):
        cfg = MODEL_CONFIGS.get(model_id, {})
        files = [local for _, local in parse_model_file_specs(cfg.get("files", []))]
        entries.append(
            {
                "choice": str(i),
                "model_id": model_id,
                "name": cfg.get("name", model_id),
                "default_dir": cfg.get("default_dir", ""),
                "files": files,
            }
        )

    headers = ["Choice", "Training Bundle", "Downloaded Models"]

    choice_w = max(len(headers[0]), max(len(e["choice"]) for e in entries))
    max_bundle_len = max(len(e["name"]) for e in entries)

    term_w = _get_terminal_width()
    min_bundle_w = max(len(headers[1]), 22)
    min_models_w = max(len(headers[2]), 30)

    available = term_w - choice_w - 10
    if available < (min_bundle_w + min_models_w):
        # Ensure we still render a table even in very narrow terminals
        min_bundle_w = max(14, min_bundle_w)
        min_models_w = max(18, min_models_w)

    bundle_w = min(max_bundle_len, max(min_bundle_w, int(available * 0.35)))
    models_w = max(min_models_w, available - bundle_w)

    if bundle_w + models_w > available:
        models_w = max(min_models_w, available - bundle_w)
    if models_w < min_models_w:
        models_w = min_models_w
        bundle_w = max(min_bundle_w, available - models_w)
    if bundle_w < min_bundle_w:
        bundle_w = min_bundle_w
        models_w = max(min_models_w, available - bundle_w)

    def wrap_cell(text: str, width: int) -> List[str]:
        if width <= 0:
            return [""]
        if not text:
            return [""]
        return textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=False) or [""]

    def border_line() -> str:
        return "+" + "-" * (choice_w + 2) + "+" + "-" * (bundle_w + 2) + "+" + "-" * (models_w + 2) + "+"

    def format_row(choice: str, bundle: str, models: str) -> List[str]:
        bundle_lines = wrap_cell(bundle, bundle_w)
        model_lines = wrap_cell(models, models_w)
        height = max(len(bundle_lines), len(model_lines), 1)
        lines = []
        for i in range(height):
            ch = choice if i == 0 else ""
            b = bundle_lines[i] if i < len(bundle_lines) else ""
            m = model_lines[i] if i < len(model_lines) else ""
            lines.append(f"| {ch:<{choice_w}} | {b:<{bundle_w}} | {m:<{models_w}} |")
        return lines

    lines = [border_line()]
    lines.extend(format_row(headers[0], headers[1], headers[2]))
    lines.append(border_line())

    for e in entries:
        models_text = ", ".join(e["files"]) if e["files"] else "-"
        lines.extend(format_row(e["choice"], e["name"], models_text))
        lines.append(border_line())

    return "\n".join(lines).rstrip()

def parse_model_choices(choice: str) -> List[str]:
    """Parse comma-separated menu numbers and inclusive ranges in entered order."""
    if not isinstance(choice, str):
        raise ValueError("Model choices must be entered as text.")

    max_choice = len(MODEL_MENU)
    selected: List[str] = []
    invalid: List[str] = []

    for raw_token in choice.split(","):
        token = raw_token.strip()
        if not token:
            continue

        match = MODEL_CHOICE_TOKEN_RE.fullmatch(token)
        if match is None:
            invalid.append(token)
            continue

        try:
            start = int(match.group(1), 10)
            end = int(match.group(2), 10) if match.group(2) is not None else start
        except (ValueError, OverflowError):
            invalid.append(token)
            continue

        if not (1 <= start <= max_choice and 1 <= end <= max_choice) or start > end:
            invalid.append(token)
            continue

        selected.extend(MODEL_MENU[menu_number - 1] for menu_number in range(start, end + 1))

    if invalid:
        invalid_text = ", ".join(repr(token) for token in invalid)
        raise ValueError(
            f"Invalid choice(s): {invalid_text}. Enter numbers or ascending ranges from "
            f"1 to {max_choice}, separated by commas (for example: 1,3-5)."
        )
    if not selected:
        raise ValueError(
            f"No model was selected. Enter numbers or ascending ranges from 1 to "
            f"{max_choice}, separated by commas (for example: 1,3-5)."
        )
    return selected


def get_model_choice() -> List[str]:
    """Get one or more model choices from the user."""
    print("\nAvailable models:")
    print(render_available_models_table())
    print()

    max_choice = len(MODEL_MENU)

    while True:
        try:
            choice = input(
                f"Please select model(s) (1-{max_choice}; numbers/ranges, comma-separated): "
            )
            try:
                return parse_model_choices(choice)
            except ValueError as exc:
                print(exc)
        except KeyboardInterrupt:
            raise

def normalize_path(path_str: str) -> str:
    """Normalize path to work on both Windows and Linux"""
    if not path_str:
        return path_str
    
    # Convert to Path object and resolve
    path = Path(path_str).expanduser()
    
    # If it's not absolute, make it relative to current directory
    if not path.is_absolute():
        path = Path.cwd() / path
    
    # Normalize and return as string
    return str(path.resolve())


def get_default_model_directories() -> List[str]:
    """Return every configured default model folder once, as an absolute path."""
    directories: List[str] = []
    seen: Set[str] = set()
    for config in MODEL_CONFIGS.values():
        default_dir = config.get("default_dir")
        if not isinstance(default_dir, str) or not default_dir.strip():
            continue
        directory = normalize_path(default_dir)
        key = os.path.normcase(directory)
        if key in seen:
            continue
        seen.add(key)
        directories.append(directory)
    return directories

def _download_model_bundle(
    model_id: str,
    download_dir: Optional[str] = None,
    revision: str = "main",
    force: bool = False,
    offline: bool = False,
    max_retries: Optional[int] = None,
    backend: str = "auto",
    aria2_path: Optional[str] = None,
    connections: Optional[int] = None,
) -> int:
    """Ensure every file in a configured training bundle is present and verified."""
    revision = str(revision or "main").strip() or "main"
    if model_id not in MODEL_CONFIGS:
        print(f"Error: Invalid model ID '{model_id}'. Available options: {list(MODEL_CONFIGS.keys())}")
        return 2

    model_config = MODEL_CONFIGS[model_id]
    target_dir = normalize_path(download_dir or model_config["default_dir"])

    try:
        download_items = parse_model_download_specs(model_config)
    except ValueError as exc:
        print(f"Error: Invalid model configuration: {exc}")
        return 2

    if not download_items:
        print("Error: This model bundle does not contain any files.")
        return 2

    config = dict(DOWNLOAD_CONFIG)
    if max_retries is not None:
        config["max_retries"] = max_retries
        config["aria2_max_tries"] = max_retries
    config["download_backend"] = backend
    if aria2_path:
        config["aria2_path"] = aria2_path
    if connections is not None:
        config["range_connections"] = connections
        config["aria2_connections"] = connections

    print(f"\nModel: {model_config['name']}")
    print(f"Description: {model_config['description']}")
    repo_ids = list(dict.fromkeys(repo_id for repo_id, _, _ in download_items))
    if len(repo_ids) == 1:
        print(f"\nRepository: {repo_ids[0]}")
    else:
        print("\nRepositories:")
        for repo_id in repo_ids:
            print(f"  - {repo_id}")
    print(f"Revision: {revision}")
    print(f"Target: {target_dir}")
    print(f"Files: {len(download_items)}")
    if offline:
        print("Mode: offline (only previously verified files are accepted)")
    elif force:
        print("Mode: force refresh")

    try:
        downloader = RobustDownloader(
            config,
            target_dir=target_dir,
            revision=revision,
            offline=offline,
            force=force,
            reuse_dirs=get_default_model_directories(),
        )
    except (OSError, ValueError) as exc:
        print(f"Error: Cannot prepare downloader: {exc}")
        return 1

    if not offline:
        print(f"Downloader: {downloader.backend_description()}")
        if backend == "aria2" and not downloader.aria2_available():
            print(
                "Error: --backend aria2 requires a working aria2c executable. "
                "Install it, put aria2c.exe beside this script on Windows, or pass --aria2-path."
            )
            return 1
        print("\n[PREFLIGHT] Resolving every file to an immutable repository commit...")
        preflight_infos: List[RemoteFileInfo] = []
        permanent_errors: List[Tuple[str, str]] = []
        for repo_id, remote_filename, local_filename in download_items:
            try:
                preflight_infos.append(downloader.get_remote_info(repo_id, remote_filename))
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                message = downloader._friendly_error(exc)
                if downloader._is_retryable(exc):
                    print(f"[WARNING] Preflight unavailable for {local_filename}: {message}")
                else:
                    permanent_errors.append((local_filename, message))

        if permanent_errors:
            for local_filename, message in permanent_errors:
                print(f"[FAILED] {local_filename}: {message}")
            print("\nBundle preflight failed; no model downloads were started.")
            return 1

        if len(preflight_infos) == len(download_items):
            total_size = sum(info.size for info in preflight_infos)
            largest_size = max(info.size for info in preflight_infos)
            committed_size = 0
            clean_target_peak = 0
            for info in preflight_infos:
                clean_target_peak = max(clean_target_peak, committed_size + 2 * info.size)
                committed_size += info.size
            resolved_repositories = list(
                dict.fromkeys((info.repo_id, info.commit_hash) for info in preflight_infos)
            )
            if len(resolved_repositories) == 1:
                print(f"Resolved commit: {resolved_repositories[0][1]}")
            else:
                print("Resolved repository commits:")
                for repo_id, commit_hash in resolved_repositories:
                    print(f"  {repo_id}: {commit_hash}")
            print(f"Bundle size:     {downloader.format_bytes(total_size)}")
            print(
                f"Peak disk estimate for an empty target: "
                f"{downloader.format_bytes(clean_target_peak)}"
            )
            try:
                free_space = shutil.disk_usage(target_dir).free
                print(f"Free disk:       {downloader.format_bytes(free_space)}")
                if free_space < largest_size:
                    print(
                        f"[WARNING] Free disk is below the largest bundle file "
                        f"({downloader.format_bytes(largest_size)}). A missing file may not fit."
                    )
                elif free_space < clean_target_peak:
                    print(
                        "[WARNING] A completely empty target may exceed current free space during "
                        "the verified range-part merge. Existing verified files or saved parts reduce "
                        "the actual requirement."
                    )
            except OSError as exc:
                print(f"[WARNING] Could not determine free disk space: {exc}")

    results: List[DownloadResult] = []
    for index, (repo_id, remote_filename, local_filename) in enumerate(download_items, 1):
        print(f"\n[{index}/{len(download_items)}] {local_filename}")
        result = downloader.ensure_file(repo_id, remote_filename, local_filename)
        results.append(result)
        if result.status == "failed":
            print(f"[FAILED] {local_filename}: {result.error}")

    skipped = [result for result in results if result.status == "skipped"]
    copied = [result for result in results if result.status == "copied"]
    downloaded = [result for result in results if result.status == "downloaded"]
    failed = [result for result in results if result.status == "failed"]

    print(f"\n{'='*50}")
    print("Summary:")
    print(f"  Already verified: {len(skipped)}")
    print(f"  Copied locally:   {len(copied)}")
    print(f"  Downloaded now:   {len(downloaded)}")
    print(f"  Failed:           {len(failed)}")
    print(f"  Ready size:       {downloader.format_bytes(sum(r.size for r in results if r.status != 'failed'))}")
    print(f"Location: {target_dir}")

    if failed:
        print("\nBundle is incomplete. Re-run the same command to resume partial downloads.")
        return 1
    print("\nAll model files are present and verified.")
    return 0


def download_models(
    model_id: Optional[str] = None,
    download_dir: Optional[str] = None,
    revision: str = "main",
    force: bool = False,
    offline: bool = False,
    max_retries: Optional[int] = None,
    backend: str = "auto",
    aria2_path: Optional[str] = None,
    connections: Optional[int] = None,
) -> int:
    """Download one bundle or an interactively selected queue of bundles in order."""
    model_ids = [model_id] if model_id else get_model_choice()
    results: List[Tuple[str, int]] = []

    for index, selected_model_id in enumerate(model_ids, 1):
        if len(model_ids) > 1:
            model_name = MODEL_CONFIGS.get(selected_model_id, {}).get(
                "name", selected_model_id
            )
            print(f"\n{'#' * 70}")
            print(f"Bundle {index}/{len(model_ids)}: {model_name}")
            print(f"{'#' * 70}")

        exit_code = _download_model_bundle(
            selected_model_id,
            download_dir,
            revision=revision,
            force=force,
            offline=offline,
            max_retries=max_retries,
            backend=backend,
            aria2_path=aria2_path,
            connections=connections,
        )
        results.append((selected_model_id, exit_code))

    if len(results) > 1:
        print(f"\n{'=' * 70}")
        print("Multiple bundle summary:")
        for selected_model_id, exit_code in results:
            model_name = MODEL_CONFIGS.get(selected_model_id, {}).get(
                "name", selected_model_id
            )
            status = "Completed" if exit_code == 0 else f"Failed (exit code {exit_code})"
            print(f"  {model_name}: {status}")

    if any(exit_code == 2 for _, exit_code in results):
        return 2
    if any(exit_code != 0 for _, exit_code in results):
        return 1
    return 0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonempty_revision(value: str) -> str:
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("must not be empty")
    return value


def connection_count(value: str) -> int:
    parsed = positive_int(value)
    if parsed > 16:
        raise argparse.ArgumentTypeError("must be between 1 and 16")
    return parsed


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reliably download and verify SECourses training model bundles"
    )
    parser.add_argument(
        '--model',
        choices=MODEL_MENU,
        help="Training bundle ID (omit for an interactive prompt)",
    )
    parser.add_argument('--dir', type=str,
                       help='Download directory (supports both full and relative paths for Windows and Linux)')
    parser.add_argument('--revision', type=nonempty_revision, default='main',
                        help='Hugging Face branch, tag, or commit to download (default: main)')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--force', action='store_true',
                      help='Refresh files even when the verified local copy is current')
    mode.add_argument('--offline', action='store_true',
                      help='Use only files previously verified by this downloader')
    parser.add_argument('--retries', type=positive_int, default=DOWNLOAD_CONFIG['max_retries'],
                        help=f"Outer retry attempts (default: {DOWNLOAD_CONFIG['max_retries']})")
    parser.add_argument(
        '--backend', '--downloader',
        choices=DOWNLOAD_BACKENDS,
        default=DOWNLOAD_CONFIG['download_backend'],
        help=(
            'Transfer backend: auto prefers aria2c, then Python ranges, then Hub; '
            'aria2 requires aria2c; ranges skips aria2c; hub uses huggingface_hub only'
        ),
    )
    parser.add_argument(
        '--aria2-path',
        help=(
            'Path to aria2c/aria2c.exe. Otherwise ARIA2C, a copy beside the script, '
            'and PATH are checked in that order'
        ),
    )
    parser.add_argument(
        '--connections',
        type=connection_count,
        default=DOWNLOAD_CONFIG['range_connections'],
        help=f"Connections per file for aria2/Python ranges (1-16; default: {DOWNLOAD_CONFIG['range_connections']})",
    )
    parser.add_argument('--list', action='store_true',
                       help='Print available models table and exit')

    args = parser.parse_args(argv)

    if args.list:
        print(render_available_models_table())
        return 0

    try:
        return download_models(
            args.model,
            args.dir,
            revision=args.revision,
            force=args.force,
            offline=args.offline,
            max_retries=args.retries,
            backend=args.backend,
            aria2_path=args.aria2_path,
            connections=args.connections,
        )
    except KeyboardInterrupt:
        print("\nDownload cancelled. Partial data was kept and will resume next time.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
