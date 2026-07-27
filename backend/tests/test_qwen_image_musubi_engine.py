"""The 'engine' axis (Wave 2): qwen_image is the only family with a choice of
local training engine — ai-toolkit (default, blanket-required regardless of
engine) or musubi-tuner. These tests exercise launch_training/enqueue_training's
engine branch with the actual subprocess boundary mocked (no real
musubi-tuner/GPU in this sandbox)."""
import os
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


# --- _default_engine_for / _train_engine family-aware default --------------

def test_default_engine_for_qwen_image_is_musubi_others_aitoolkit():
    from app.services import lora_training as lt
    assert lt._default_engine_for('qwen_image') == 'musubi'
    assert lt._default_engine_for('zimage') == 'aitoolkit'
    assert lt._default_engine_for('sdxl') == 'aitoolkit'
    assert lt._default_engine_for(None) == 'aitoolkit'


def test_train_engine_defaults_to_musubi_for_unset_qwen_image_dataset(app, tmp_path):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QE1', 'zchar_qe1', train_type='qwen_image')
        assert lt._train_engine(ds) == 'musubi'
        # Every other family still defaults to aitoolkit.
        ds2 = svc.create_dataset(LOCAL_USER, 'QE2', 'zchar_qe2', train_type='zimage')
        assert lt._train_engine(ds2) == 'aitoolkit'


def test_train_engine_explicit_persisted_choice_wins_over_default(app, tmp_path):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QE3', 'zchar_qe3', train_type='qwen_image')
        ds.train_engine = 'aitoolkit'
        svc.db.session.commit()
        assert lt._train_engine(ds) == 'aitoolkit'   # persisted choice, not the default
        assert lt._train_engine(ds, engine='musubi') == 'musubi'   # override wins over persisted


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


def test_launch_refuses_musubi_weight_error_names_the_configured_path(app, tmp_path, monkeypatch):
    """The error must show WHAT path was read, not just the bare key name —
    "text_encoder" alone gives a user nothing to check their Settings against."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app import config as cfg
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app)
    _configure_musubi(tmp_path, app, with_weights=True)
    monkeypatch.setattr(lt.shutil, 'disk_usage',
                        lambda p: type('u', (), {'free': 500e9})())
    bogus = str(tmp_path / 'weights' / 'not_really_there.safetensors')
    with app.app_context():
        cfg.save_config({'musubi_tuner': {'qwen_image_text_encoder': bogus}})
        ds = svc.create_dataset(LOCAL_USER, 'QI2B', 'zchar_qi2b', train_type='qwen_image')
        with pytest.raises(ValueError, match=r'text_encoder .*not found') as exc:
            lt.launch_training(LOCAL_USER, ds.id, check_captions=False, engine='musubi')
    assert bogus in str(exc.value)


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

    def _fake_write_toml(dataset_folder, cache_dir, resolution, caption_ext='txt', control_dir=None):
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


def test_relaunch_archives_previous_attempts_log(app, tmp_path, monkeypatch):
    """Repo owner report: after an OOM, retrying (fresh=False, same base+
    variant -> same run dir) used to overwrite/concatenate the previous
    attempt's training.log with no way to read it back once the retry had
    started writing. launch_training must archive whatever log is already
    there BEFORE the new one starts, on every launch, not just a fresh one."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.services import musubi_tuner as mt
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app)
    _configure_musubi(tmp_path, app, with_weights=True)
    _mock_disk_and_kept(monkeypatch, lt, tmp_path)

    class _FakeProc:
        pid = 1

    with app.app_context(), \
         patch.object(mt, 'write_dataset_toml', return_value=str(tmp_path / 'ds.toml')), \
         patch.object(mt, 'run_precache'), \
         patch.object(mt, 'build_train_argv', return_value=[]), \
         patch.object(mt, 'spawn_training', return_value=_FakeProc()):
        ds = svc.create_dataset(LOCAL_USER, 'QI3B', 'zchar_qi3b', train_type='qwen_image')
        lt.launch_training(LOCAL_USER, ds.id, check_captions=False, engine='musubi')
        run_dir = lt._run_dir(LOCAL_USER, ds.id)
        # Simulate the OOM'd first attempt having written some output, then
        # actually died (the training_in_progress flag from the fake PID
        # above would otherwise make the retry refuse as "already running").
        with open(os.path.join(run_dir, 'training.log'), 'w', encoding='utf-8') as fh:
            fh.write('attempt 1 - OOM\n')
        from app.job_queue import queue_manager
        queue_manager._set_system_state('training_in_progress', False)

        lt.launch_training(LOCAL_USER, ds.id, check_captions=False, engine='musubi')

        # The mocked run_precache/spawn_training never actually write to
        # log_path, so no fresh training.log exists here — only the archived
        # attempt matters for this test (a real launch's own log content is
        # covered by test_run_log.py).
        logs = lt.list_run_logs(LOCAL_USER, ds.id)
        names = [l['filename'] for l in logs]
        assert len(names) == 1
        assert names[0] != 'training.log'
        with open(os.path.join(run_dir, names[0]), encoding='utf-8') as fh:
            assert fh.read() == 'attempt 1 - OOM\n'


# --- launch_training: GPU profile wiring ------------------------------------

def test_launch_musubi_default_profile_matches_pre_profile_behaviour(app, tmp_path, monkeypatch):
    """No musubi_profile set -> today's exact behaviour: rank from the shared
    qwen_image family setting (32 default), no fp8, no block-swap."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.services import musubi_tuner as mt
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app)
    _configure_musubi(tmp_path, app, with_weights=True)
    _mock_disk_and_kept(monkeypatch, lt, tmp_path)

    captured = {}

    def _fake_build_argv(**kwargs):
        captured.update(kwargs)
        return ['-m', 'accelerate.commands.launch']

    class _FakeProc:
        pid = 1

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QIP1', 'zchar_qip1', train_type='qwen_image')
        with patch.object(mt, 'write_dataset_toml', return_value=str(tmp_path / 'x.toml')), \
             patch.object(mt, 'run_precache', return_value=None), \
             patch.object(mt, 'build_train_argv', side_effect=_fake_build_argv), \
             patch.object(mt, 'spawn_training', return_value=_FakeProc()):
            lt.launch_training(LOCAL_USER, ds.id, check_captions=False, engine='musubi')

    assert captured['rank'] == 32   # unchanged qwen_image family default
    assert captured['fp8_base'] is False
    assert captured['fp8_scaled'] is False
    assert captured['blocks_to_swap'] == 0


def test_launch_musubi_profile_rtx5090_overrides_rank_and_sets_fp8(app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.services import musubi_tuner as mt
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app)
    _configure_musubi(tmp_path, app, with_weights=True)
    _mock_disk_and_kept(monkeypatch, lt, tmp_path)

    captured = {}

    def _fake_build_argv(**kwargs):
        captured.update(kwargs)
        return ['-m', 'accelerate.commands.launch']

    class _FakeProc:
        pid = 1

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QIP2', 'zchar_qip2', train_type='qwen_image')
        lt.update_train_settings(LOCAL_USER, ds.id, {'musubi_profile': 'rtx5090'})
        with patch.object(mt, 'write_dataset_toml', return_value=str(tmp_path / 'x.toml')), \
             patch.object(mt, 'run_precache', return_value=None), \
             patch.object(mt, 'build_train_argv', side_effect=_fake_build_argv), \
             patch.object(mt, 'spawn_training', return_value=_FakeProc()):
            lt.launch_training(LOCAL_USER, ds.id, check_captions=False, engine='musubi')

    p = mt.MUSUBI_PROFILES['rtx5090']
    assert captured['rank'] == p['rank']
    assert captured['fp8_base'] is True
    assert captured['fp8_scaled'] is True
    assert captured['blocks_to_swap'] == p['blocks_to_swap']
    # Full-recipe fields (not just VRAM knobs) also come from the profile.
    assert captured['network_alpha'] == p['alpha']
    assert captured['weighting_scheme'] == p['weighting_scheme']
    assert captured['optimizer'] == p['optimizer']
    assert captured['learning_rate'] == p['learning_rate']
    assert captured['optimizer_args'] == p['optimizer_args']


def test_launch_musubi_profile_vram5_overrides_resolution_in_dataset_toml(app, tmp_path, monkeypatch):
    """A profile's resolution (not just rank/fp8/swap) reaches write_dataset_toml."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.services import musubi_tuner as mt
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app)
    _configure_musubi(tmp_path, app, with_weights=True)
    _mock_disk_and_kept(monkeypatch, lt, tmp_path)

    captured_toml = {}

    def _fake_write_toml(dataset_folder, cache_dir, resolution, caption_ext='txt', control_dir=None):
        captured_toml['resolution'] = resolution
        return str(tmp_path / 'x.toml')

    class _FakeProc:
        pid = 1

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QIP5', 'zchar_qip5', train_type='qwen_image')
        lt.update_train_settings(LOCAL_USER, ds.id, {'musubi_profile': 'vram5'})
        with patch.object(mt, 'write_dataset_toml', side_effect=_fake_write_toml), \
             patch.object(mt, 'run_precache', return_value=None), \
             patch.object(mt, 'build_train_argv', return_value=['-m', 'accelerate.commands.launch']), \
             patch.object(mt, 'spawn_training', return_value=_FakeProc()):
            lt.launch_training(LOCAL_USER, ds.id, check_captions=False, engine='musubi')

    assert captured_toml['resolution'] == mt.MUSUBI_PROFILES['vram5']['resolution']


def test_launch_musubi_sample_resolution_capped_at_1024_for_rtx5090(app, tmp_path, monkeypatch):
    """Bug reported 2026-07-26: sampling at the full 1328 training resolution
    added a VAE decode-to-pixels pass that OOM'd a validated rtx5090 profile
    (fp8 + blocks_to_swap=20 already left it at ~30.6/31.4 GB during plain
    training). The preview must be capped at 1024 WITHOUT touching the
    profile's own training resolution (still fed to write_dataset_toml)."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.services import musubi_tuner as mt
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app)
    _configure_musubi(tmp_path, app, with_weights=True)
    _mock_disk_and_kept(monkeypatch, lt, tmp_path)

    captured_toml, captured_samples = {}, {}

    def _fake_write_toml(dataset_folder, cache_dir, resolution, caption_ext='txt', control_dir=None):
        captured_toml['resolution'] = resolution
        return str(tmp_path / 'x.toml')

    def _fake_write_samples(prompts, path, width=1024, height=1024, steps=20):
        captured_samples.update(width=width, height=height)
        return path

    class _FakeProc:
        pid = 1

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QIP6', 'zchar_qip6', train_type='qwen_image')
        lt.update_train_settings(LOCAL_USER, ds.id, {'musubi_profile': 'rtx5090'})
        with patch.object(mt, 'write_dataset_toml', side_effect=_fake_write_toml), \
             patch.object(mt, 'run_precache', return_value=None), \
             patch.object(mt, 'write_sample_prompts', side_effect=_fake_write_samples), \
             patch.object(mt, 'build_train_argv', return_value=['-m', 'accelerate.commands.launch']), \
             patch.object(mt, 'spawn_training', return_value=_FakeProc()):
            lt.launch_training(LOCAL_USER, ds.id, check_captions=False, engine='musubi')

    assert mt.MUSUBI_PROFILES['rtx5090']['resolution'] == 1328   # training resolution untouched...
    assert captured_toml['resolution'] == 1328                  # ...still reaches the dataset TOML
    assert captured_samples == {'width': 1024, 'height': 1024}  # ...but the preview is capped


def test_launch_musubi_sample_resolution_not_bumped_up_for_vram5(app, tmp_path, monkeypatch):
    """A profile already BELOW 1024 (vram5: 768) must not have its preview
    bumped UP to 1024 - the cap is a ceiling, not a fixed size."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.services import musubi_tuner as mt
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app)
    _configure_musubi(tmp_path, app, with_weights=True)
    _mock_disk_and_kept(monkeypatch, lt, tmp_path)

    captured_samples = {}

    def _fake_write_samples(prompts, path, width=1024, height=1024, steps=20):
        captured_samples.update(width=width, height=height)
        return path

    class _FakeProc:
        pid = 1

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QIP7', 'zchar_qip7', train_type='qwen_image')
        lt.update_train_settings(LOCAL_USER, ds.id, {'musubi_profile': 'vram5'})
        with patch.object(mt, 'write_dataset_toml', return_value=str(tmp_path / 'x.toml')), \
             patch.object(mt, 'run_precache', return_value=None), \
             patch.object(mt, 'write_sample_prompts', side_effect=_fake_write_samples), \
             patch.object(mt, 'build_train_argv', return_value=['-m', 'accelerate.commands.launch']), \
             patch.object(mt, 'spawn_training', return_value=_FakeProc()):
            lt.launch_training(LOCAL_USER, ds.id, check_captions=False, engine='musubi')

    assert mt.MUSUBI_PROFILES['vram5']['resolution'] == 768
    assert captured_samples == {'width': 768, 'height': 768}


def test_effective_train_settings_exposes_profile_optimizer_for_ui_override_note(app, tmp_path):
    """The panel disables Rank/Optimizer and shows the profile's effective value
    instead when a GPU profile != Auto is active (repo owner's request,
    2026-07-26) - it needs `optimizer` alongside `rank`/`label`/`note` per
    profile, not just `rank` alone."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.services import musubi_tuner as mt
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QIP4', 'zchar_qip4', train_type='qwen_image')
        snap = lt.effective_train_settings(ds)
        for name, profile in mt.MUSUBI_PROFILES.items():
            assert snap['musubi_profiles'][name]['optimizer'] == profile['optimizer']
            assert snap['musubi_profiles'][name]['rank'] == profile['rank']


# --- Training panel's own live sample gallery must find musubi's samples ----
# Bug found 2026-07-27 (repo owner asked where training samples show up in the
# UI): musubi-tuner writes previews to `<run>/sample/` (singular) while
# ai-toolkit writes `<run>/samples/` (plural) - cloud_training.py's Runs-hub
# thumbnail was already engine-aware about this, but the Training panel's OWN
# live gallery (training_progress -> list_training_samples -> _samples_dir)
# stayed hardcoded to the plural name, so a musubi run's live previews never
# showed up there even though musubi wrote them correctly to disk.

def test_samples_dir_is_singular_for_musubi_plural_for_aitoolkit(app, tmp_path):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app)
    with app.app_context():
        musubi_ds = svc.create_dataset(LOCAL_USER, 'QIS1', 'zchar_qis1', train_type='qwen_image')
        musubi_ds.train_engine = 'musubi'
        svc.db.session.commit()
        assert lt._samples_dir(LOCAL_USER, musubi_ds.id).endswith(os.sep + 'sample')

        aitoolkit_ds = svc.create_dataset(LOCAL_USER, 'QIS2', 'zchar_qis2', train_type='qwen_image')
        aitoolkit_ds.train_engine = 'aitoolkit'
        svc.db.session.commit()
        assert lt._samples_dir(LOCAL_USER, aitoolkit_ds.id).endswith(os.sep + 'samples')


def test_list_training_samples_finds_musubi_previews(app, tmp_path):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QIS3', 'zchar_qis3', train_type='qwen_image')
        ds.train_engine = 'musubi'
        svc.db.session.commit()
        sample_dir = os.path.join(lt._samples_dir(LOCAL_USER, ds.id))
        os.makedirs(sample_dir)
        with open(os.path.join(sample_dir, '20260727-000000__000010_0.jpg'), 'wb') as fh:
            fh.write(b'fake')

        samples = lt.list_training_samples(LOCAL_USER, ds.id)
        assert len(samples) == 1
        assert samples[0]['step'] == 10


def test_update_train_settings_musubi_profile_validation():
    """Covered without app fixtures elsewhere in the suite; here we only check
    the choices constant lines up with what update_train_settings accepts."""
    from app.services import musubi_tuner as mt
    assert 'vram48' in mt.MUSUBI_PROFILE_CHOICES
    assert 'bogus' not in mt.MUSUBI_PROFILE_CHOICES


def test_update_train_settings_rejects_unknown_musubi_profile(app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app)
    monkeypatch.setattr(lt.shutil, 'disk_usage',
                        lambda p: type('u', (), {'free': 500e9})())
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QIP3', 'zchar_qip3', train_type='qwen_image')
        with pytest.raises(ValueError, match=r'musubi_profile must be one of'):
            lt.update_train_settings(LOCAL_USER, ds.id, {'musubi_profile': 'bogus'})


def test_update_train_settings_musubi_profile_auto_clears_key(app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app)
    monkeypatch.setattr(lt.shutil, 'disk_usage',
                        lambda p: type('u', (), {'free': 500e9})())
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QIP4', 'zchar_qip4', train_type='qwen_image')
        lt.update_train_settings(LOCAL_USER, ds.id, {'musubi_profile': 'vram48'})
        eff = lt.update_train_settings(LOCAL_USER, ds.id, {'musubi_profile': 'auto'})
    assert eff['musubi_profile'] is None


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


# --- list_checkpoints() must find musubi's own output layout ---------------
# Repo owner report: after a real musubi run finished, the guided-flow "Train"
# step never went green. Root cause (two separate bugs, both fixed together):
# (1) musubi writes checkpoints directly into --output_dir (no ai-toolkit-style
#     lora_<trigger>/ subfolder of its own) — _run_dir() was unconditionally
#     appending that subfolder regardless of engine.
# (2) musubi/kohya's own epoch-numbered filenames use a HYPHEN
#     (`lora_x-000001.safetensors`), not the underscore ai-toolkit's own
#     step-numbered saves use (`lora_x_0000500.safetensors`) — the old
#     _CK_RE only matched the underscore form.

def test_run_dir_musubi_has_no_extra_subfolder(app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QID1', 'zchar_qid1', train_type='qwen_image')
        ds.train_engine = 'musubi'
        svc.db.session.commit()
        musubi_run = lt._run_dir(LOCAL_USER, ds.id)
        assert not musubi_run.endswith('lora_zchar_qid1')
        assert musubi_run == str(lt._output_dir() / lt._run_name(ds))

        ds.train_engine = 'aitoolkit'
        svc.db.session.commit()
        aitoolkit_run = lt._run_dir(LOCAL_USER, ds.id)
        assert aitoolkit_run.endswith(os.sep + 'lora_zchar_qid1')


def test_list_checkpoints_finds_musubi_hyphenated_epoch_files(app, tmp_path):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QID2', 'zchar_qid2', train_type='qwen_image')
        ds.train_engine = 'musubi'
        svc.db.session.commit()
        run_dir = lt._output_dir() / lt._run_name(ds)   # no lora_<trigger>/ subfolder
        run_dir.mkdir(parents=True)
        (run_dir / 'lora_zchar_qid2-000001.safetensors').write_bytes(b'fake')
        (run_dir / 'lora_zchar_qid2-000002.safetensors').write_bytes(b'fake')

        cks = lt.list_checkpoints(LOCAL_USER, ds.id)
        assert [c['filename'] for c in cks] == [
            'lora_zchar_qid2-000001.safetensors', 'lora_zchar_qid2-000002.safetensors']
        assert [c['step'] for c in cks] == [1, 2]


def test_list_checkpoints_finds_musubi_final_bare_name_file(app, tmp_path):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QID3', 'zchar_qid3', train_type='qwen_image')
        ds.train_engine = 'musubi'
        svc.db.session.commit()
        run_dir = lt._output_dir() / lt._run_name(ds)
        run_dir.mkdir(parents=True)
        (run_dir / 'lora_zchar_qid3-000001.safetensors').write_bytes(b'fake')
        (run_dir / 'lora_zchar_qid3.safetensors').write_bytes(b'fake')   # final, no suffix

        cks = lt.list_checkpoints(LOCAL_USER, ds.id)
        finals = [c for c in cks if c.get('final')]
        assert len(finals) == 1
        assert finals[0]['filename'] == 'lora_zchar_qid3.safetensors'
