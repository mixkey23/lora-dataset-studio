"""musubi_tuner.py — the Qwen-Image-only mechanics module for the second
local training engine. Pure-function/subprocess-boundary tests: no real
musubi-tuner install or GPU is available in this sandbox, so subprocess.run
is mocked and only the argv/TOML/state-checking logic is verified."""
from unittest.mock import patch

import pytest


def _configure_musubi(tmp_path, app, with_weights=True):
    """Fake musubi-tuner install: .venv/bin/python + the 3 Qwen-Image
    scripts under src/musubi_tuner/. Optionally also configures the 3
    Qwen-Image weight paths as real (empty) files on disk."""
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
        dit = weights_dir / 'dit.safetensors'
        vae = weights_dir / 'vae.safetensors'
        te = weights_dir / 'te.safetensors'
        for p in (dit, vae, te):
            p.write_text('fake')
        section.update({'qwen_image_dit': str(dit), 'qwen_image_vae': str(vae),
                        'qwen_image_text_encoder': str(te)})
    with app.app_context():
        cfg.save_config({'musubi_tuner': section})
    return root


# --- is_installed() ---------------------------------------------------------

def test_is_installed_true_when_venv_and_scripts_present(app, tmp_path):
    from app.services import musubi_tuner as mt
    _configure_musubi(tmp_path, app)
    with app.app_context():
        assert mt.is_installed() is True


def test_is_installed_false_when_unconfigured(app):
    from app.services import musubi_tuner as mt
    with app.app_context():
        assert mt.is_installed() is False


def test_is_installed_false_when_a_script_is_missing(app, tmp_path):
    from app.services import musubi_tuner as mt
    root = _configure_musubi(tmp_path, app)
    (root / 'src' / 'musubi_tuner' / 'qwen_image_train_network.py').unlink()
    with app.app_context():
        assert mt.is_installed() is False


# --- qwen_image_weights() / missing_qwen_image_weights() -------------------

def test_missing_qwen_image_weights_empty_when_all_configured(app, tmp_path):
    from app.services import musubi_tuner as mt
    _configure_musubi(tmp_path, app, with_weights=True)
    with app.app_context():
        assert mt.missing_qwen_image_weights() == []


def test_missing_qwen_image_weights_lists_each_unset_or_missing_key(app, tmp_path):
    from app.services import musubi_tuner as mt
    from app import config as cfg
    _configure_musubi(tmp_path, app, with_weights=False)
    with app.app_context():
        missing = mt.missing_qwen_image_weights()
    assert set(missing) == {'dit', 'vae', 'text_encoder'}

    # A configured-but-nonexistent path also counts as missing.
    with app.app_context():
        cfg.save_config({'musubi_tuner': {
            'qwen_image_dit': str(tmp_path / 'nope.safetensors')}})
        missing = mt.missing_qwen_image_weights()
    assert 'dit' in missing
    assert 'vae' in missing and 'text_encoder' in missing


# --- write_dataset_toml() ---------------------------------------------------

def test_write_dataset_toml_exact_content(tmp_path):
    from app.services import musubi_tuner as mt
    dataset_folder = str(tmp_path / 'my_dataset')
    cache_dir = str(tmp_path / 'my_dataset_cache')
    toml_path = mt.write_dataset_toml(dataset_folder, cache_dir, resolution=1024, caption_ext='txt')
    assert toml_path == f'{dataset_folder}_musubi.toml'
    content = open(toml_path, encoding='utf-8').read()
    assert content == (
        '[general]\n'
        'resolution = [1024, 1024]\n'
        'caption_extension = ".txt"\n'
        'batch_size = 1\n'
        'num_repeats = 1\n'
        'enable_bucket = false\n'
        'bucket_no_upscale = false\n'
        '\n'
        '[[datasets]]\n'
        f'image_directory = "{dataset_folder}"\n'
        f'cache_directory = "{cache_dir}"\n'
    )


# --- build_train_argv() -----------------------------------------------------

def test_build_train_argv_exact_flags_model_version_original(app, tmp_path):
    from app.services import musubi_tuner as mt
    _configure_musubi(tmp_path, app, with_weights=True)
    with app.app_context():
        weights = mt.qwen_image_weights()
        argv = mt.build_train_argv(
            toml_path='/x/ds.toml', model_version='original', rank=32,
            learning_rate=5e-5, optimizer='adamw8bit', timestep_type='shift',
            max_train_epochs=16, output_dir='/x/out', output_name='lora_foo')
    assert argv == [
        '-m', 'accelerate.commands.launch',
        '--num_cpu_threads_per_process', '1', '--mixed_precision', 'bf16',
        mt.QWEN_IMAGE_SCRIPTS['train'],
        '--dit', weights['dit'], '--vae', weights['vae'],
        '--text_encoder', weights['text_encoder'], '--model_version', 'original',
        '--dataset_config', '/x/ds.toml', '--sdpa', '--mixed_precision', 'bf16',
        '--timestep_sampling', 'shift', '--weighting_scheme', 'none',
        '--discrete_flow_shift', '2.2',
        '--optimizer_type', 'adamw8bit', '--learning_rate', '5e-05',
        '--gradient_checkpointing',
        '--max_data_loader_n_workers', '2', '--persistent_data_loader_workers',
        '--network_module', 'networks.lora_qwen_image', '--network_dim', '32',
        '--max_train_epochs', '16', '--save_every_n_epochs', '1',
        '--seed', '42',
        '--output_dir', '/x/out', '--output_name', 'lora_foo',
    ]


def test_build_train_argv_model_version_edit2511_and_unknown_sampling_falls_back(app, tmp_path):
    from app.services import musubi_tuner as mt
    _configure_musubi(tmp_path, app, with_weights=True)
    with app.app_context():
        argv = mt.build_train_argv(
            toml_path='/x/ds.toml', model_version='edit-2511', rank=16,
            learning_rate=1e-4, optimizer='adamw', timestep_type='bogus',
            max_train_epochs=8, output_dir='/x/out', output_name='lora_bar')
    assert '--model_version' in argv
    assert argv[argv.index('--model_version') + 1] == 'edit-2511'
    assert '--timestep_sampling' in argv
    assert argv[argv.index('--timestep_sampling') + 1] == 'shift'   # unknown -> safe default
    assert '--network_alpha' not in argv   # unconfirmed flag, deliberately omitted


# --- run_precache() ---------------------------------------------------------

def test_run_precache_raises_with_log_tail_on_nonzero_exit(app, tmp_path):
    from app.services import musubi_tuner as mt
    _configure_musubi(tmp_path, app, with_weights=True)
    log_path = str(tmp_path / 'train.log')

    class _FakeResult:
        returncode = 1

    with app.app_context(), patch.object(mt.subprocess, 'run', return_value=_FakeResult()) as run_mock:
        with pytest.raises(RuntimeError, match=r'cache latents'):
            mt.run_precache('/x/ds.toml', 'original', log_path)
    assert run_mock.called


def test_run_precache_runs_both_steps_in_order_and_succeeds(app, tmp_path):
    from app.services import musubi_tuner as mt
    _configure_musubi(tmp_path, app, with_weights=True)
    log_path = str(tmp_path / 'train.log')

    class _OkResult:
        returncode = 0

    calls = []

    def _fake_run(argv, cwd, env, stdout, stderr):
        calls.append(argv)
        return _OkResult()

    with app.app_context(), patch.object(mt.subprocess, 'run', side_effect=_fake_run):
        mt.run_precache('/x/ds.toml', 'original', log_path)
    assert len(calls) == 2
    assert 'qwen_image_cache_latents.py' in calls[0][1]
    assert 'qwen_image_cache_text_encoder_outputs.py' in calls[1][1]
