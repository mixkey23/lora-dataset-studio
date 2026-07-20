"""The 'engine' axis (Wave 2): qwen_image is the only family with a choice of
local training engine — ai-toolkit (default, blanket-required regardless of
engine) or musubi-tuner. These tests exercise launch_training/enqueue_training's
engine branch with the actual subprocess boundary mocked (no real
musubi-tuner/GPU in this sandbox)."""
from unittest.mock import patch

import pytest


def _configure_aitoolkit(tmp_path, app):
    """ai-toolkit stays a blanket requirement regardless of engine — every
    test in this file needs it installed, qwen_image arch included so the
    ai-toolkit-engine guard never trips on tests that don't care about it."""
    from app import config as cfg
    root = tmp_path / 'aitoolkit'
    (root / 'venv' / 'bin').mkdir(parents=True)
    (root / 'venv' / 'bin' / 'python').write_text('fake')
    (root / 'run.py').write_text('fake')
    ext = root / 'extensions_built_in' / 'diffusion_models' / 'qwen_image'
    ext.mkdir(parents=True)
    (ext / 'qwen_image_model.py').write_text(
        'class QwenImageModel:\n    arch = "qwen_image"\n', encoding='utf-8')
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(root)}})
    return root


def _configure_musubi(tmp_path, app, with_weights=True):
    from app import config as cfg
    root = tmp_path / 'musubi-tuner'
    (root / '.venv' / 'bin').mkdir(parents=True)
    (root / '.venv' / 'bin' / 'python').write_text('fake')
    scripts = root / 'src' / 'musubi_tuner'
    scripts.mkdir(parents=True)
    (scripts / 'qwen_image_cache_latents.py').write_text('fake')
    (scripts / 'qwen_image_cache_text_encoder_outputs.py').write_text('fake')
    (scripts / 'qwen_image_train_network.py').write_text('fake')
    section = {'dir': str(root)}
    if with_weights:
        weights_dir = tmp_path / 'weights'
        weights_dir.mkdir()
        dit, vae, te = (weights_dir / n for n in
                        ('dit.safetensors', 'vae.safetensors', 'te.safetensors'))
        for p in (dit, vae, te):
            p.write_text('fake')
        section.update({'qwen_image_dit': str(dit), 'qwen_image_vae': str(vae),
                        'qwen_image_text_encoder': str(te)})
    with app.app_context():
        cfg.save_config({'musubi_tuner': section})
    return root


def _mock_disk_and_kept(monkeypatch, lt, tmp_path, n_kept=5):
    monkeypatch.setattr(lt.shutil, 'disk_usage',
                        lambda p: type('u', (), {'free': 500e9})())
    monkeypatch.setattr(lt, 'assert_trainable', lambda *_a, **_kw: None)
    monkeypatch.setattr(lt, 'export_dataset_to_aitoolkit',
                        lambda *_a, **_kw: str(tmp_path / 'ds_export'))


# --- _train_engine / _valid_engines_for ------------------------------------

def test_valid_engines_for_qwen_image_vs_other_families():
    from app.services import lora_training as lt
    assert lt._valid_engines_for('qwen_image') == ('aitoolkit', 'musubi')
    assert lt._valid_engines_for('zimage') == ('aitoolkit',)
    assert lt._valid_engines_for('sdxl') == ('aitoolkit',)


# --- launch_training: engine validation guards ------------------------------

def test_launch_refuses_musubi_engine_for_non_qwen_image_family(app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app)
    monkeypatch.setattr(lt.shutil, 'disk_usage',
                        lambda p: type('u', (), {'free': 500e9})())
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'ZI', 'zchar_zi', train_type='zimage')
        with pytest.raises(ValueError, match=r"engine isn't available"):
            lt.launch_training(LOCAL_USER, ds.id, check_captions=False, engine='musubi')


def test_launch_refuses_musubi_when_not_installed(app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app)
    monkeypatch.setattr(lt.shutil, 'disk_usage',
                        lambda p: type('u', (), {'free': 500e9})())
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QI', 'zchar_qi', train_type='qwen_image')
        with pytest.raises(ValueError, match=r'musubi-tuner is not configured'):
            lt.launch_training(LOCAL_USER, ds.id, check_captions=False, engine='musubi')


def test_launch_refuses_musubi_when_weight_missing(app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app)
    _configure_musubi(tmp_path, app, with_weights=False)
    monkeypatch.setattr(lt.shutil, 'disk_usage',
                        lambda p: type('u', (), {'free': 500e9})())
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QI2', 'zchar_qi2', train_type='qwen_image')
        with pytest.raises(ValueError, match=r'missing Qwen-Image weight path'):
            lt.launch_training(LOCAL_USER, ds.id, check_captions=False, engine='musubi')


# --- launch_training: full success path (mocked subprocess boundary) -------

def test_launch_musubi_success_calls_write_precache_spawn_in_order(app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.services import musubi_tuner as mt
    from app.job_queue import queue_manager
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app)
    _configure_musubi(tmp_path, app, with_weights=True)
    _mock_disk_and_kept(monkeypatch, lt, tmp_path)

    calls = []

    def _fake_write_toml(dataset_folder, cache_dir, resolution, caption_ext='txt'):
        calls.append('write_dataset_toml')
        return str(tmp_path / 'ds_musubi.toml')

    def _fake_run_precache(toml_path, model_version, log_path):
        calls.append('run_precache')

    def _fake_build_argv(**kwargs):
        calls.append('build_train_argv')
        return ['-m', 'accelerate.commands.launch']

    class _FakeProc:
        pid = 424242

    def _fake_spawn(argv, cwd, env, log_path):
        calls.append('spawn_training')
        return _FakeProc()

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QI3', 'zchar_qi3', train_type='qwen_image')
        with patch.object(mt, 'write_dataset_toml', side_effect=_fake_write_toml), \
             patch.object(mt, 'run_precache', side_effect=_fake_run_precache), \
             patch.object(mt, 'build_train_argv', side_effect=_fake_build_argv), \
             patch.object(mt, 'spawn_training', side_effect=_fake_spawn):
            result = lt.launch_training(LOCAL_USER, ds.id, check_captions=False, engine='musubi')

        assert calls == ['write_dataset_toml', 'run_precache', 'build_train_argv', 'spawn_training']
        assert result['engine'] == 'musubi'
        assert result['pid'] == 424242
        # Same identity/PID bookkeeping as an ai-toolkit launch.
        assert queue_manager._get_system_state('training_in_progress', False) is True
        assert queue_manager._get_system_state('training_pid', None) == 424242
        assert ds.train_engine == 'musubi'


# --- enqueue_training: same guards, plus the resume refusal ----------------

def test_enqueue_refuses_musubi_resume(app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app)
    _configure_musubi(tmp_path, app, with_weights=True)
    monkeypatch.setattr(lt, 'assert_trainable', lambda *_a, **_kw: None)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QI4', 'zchar_qi4', train_type='qwen_image')
        with pytest.raises(ValueError, match=r"continuing/resuming a run isn't available"):
            lt.enqueue_training(LOCAL_USER, ds.id, extra_steps=100, engine='musubi')


def test_enqueue_refuses_musubi_when_not_installed(app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app)
    monkeypatch.setattr(lt, 'assert_trainable', lambda *_a, **_kw: None)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QI5', 'zchar_qi5', train_type='qwen_image')
        with pytest.raises(ValueError, match=r'musubi-tuner is not configured'):
            lt.enqueue_training(LOCAL_USER, ds.id, engine='musubi')


def test_enqueue_musubi_persists_engine_on_success(app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app)
    _configure_musubi(tmp_path, app, with_weights=True)
    monkeypatch.setattr(lt, 'assert_trainable', lambda *_a, **_kw: None)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QI6', 'zchar_qi6', train_type='qwen_image')
        result = lt.enqueue_training(LOCAL_USER, ds.id, engine='musubi')
        assert result['queued'] is True
        assert ds.train_engine == 'musubi'
