"""Qwen Multi-angle — the camera-angle rotation GENERATION engine (not the
'qwen_image' TRAINING family, see test_qwen_image_family.py). Rotates an
existing render's camera angle using Qwen-Image-Edit-2511 + the community
'fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA', modeled on the Klein edit
engine's resolver/degrade-not-fail/job-queue pattern."""
import struct
from unittest.mock import patch

import pytest

# Smallest structurally-valid safetensors header (mirrors test_klein_models.py).
_VALID_ST = struct.pack('<Q', 2) + b'{}'


def _install(base, *relparts, data=_VALID_ST):
    p = base.joinpath(*relparts)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def _comfy(tmp_path, cfg, unet=True, vae=True, te=True, multiangle=True,
          consistency=False, lightning=False):
    """A ComfyUI tree with a configurable subset of the qwen_multiangle assets
    present, mirroring test_klein_models.py's _comfy fixture."""
    base = tmp_path / 'comfyui'
    (base / 'input').mkdir(parents=True)
    (base / 'output').mkdir(parents=True)
    (base / 'models').mkdir(parents=True)
    (base / 'main.py').write_text('# fake', encoding='utf-8')
    if unet:
        _install(base, 'models', 'unet', 'QwenImage', 'Qwen-Image-Edit-2511-FP8_e4m3fn.safetensors')
    if vae:
        _install(base, 'models', 'vae', 'qwen_image_vae.safetensors')
    if te:
        _install(base, 'models', 'text_encoders', 'qwen_2.5_vl_7b_fp8_scaled.safetensors')
    if multiangle:
        _install(base, 'models', 'loras', 'qwen-image-edit-2511-multiple-angles-lora.safetensors')
    if consistency:
        _install(base, 'models', 'loras', 'consistence_edit_v2.safetensors')
    if lightning:
        _install(base, 'models', 'loras', 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors')
    cfg.save_config({'comfyui': {'base_dir': str(base)}})
    return base


# --- Resolvers ---------------------------------------------------------------

def test_resolvers_find_assets_under_subfolder_or_root(app, tmp_path):
    import os
    from app import config as cfg
    from app.services import qwen_multiangle_helper as qmh
    with app.app_context():
        _comfy(tmp_path, cfg)
        assert qmh.resolve_qwen_ma_unet() == os.path.join(
            'QwenImage', 'Qwen-Image-Edit-2511-FP8_e4m3fn.safetensors')
        assert qmh.resolve_qwen_ma_vae() == 'qwen_image_vae.safetensors'
        assert qmh.resolve_qwen_ma_text_encoder() == 'qwen_2.5_vl_7b_fp8_scaled.safetensors'
        rel, path = qmh.resolve_qwen_ma_multiangle_lora()
        assert rel == 'qwen-image-edit-2511-multiple-angles-lora.safetensors'
        assert path is not None


def test_missing_assets_lists_every_absent_action(app, tmp_path):
    from app import config as cfg
    from app.services import qwen_multiangle_helper as qmh
    with app.app_context():
        _comfy(tmp_path, cfg, unet=False, multiangle=False)
        missing = qmh.qwen_ma_missing_assets()
        assert 'qwen_ma_unet' in missing
        assert 'qwen_ma_multiangle_lora' in missing
        assert 'qwen_ma_vae' not in missing   # present in this fixture


def test_missing_assets_empty_when_everything_present(app, tmp_path):
    from app import config as cfg
    from app.services import qwen_multiangle_helper as qmh
    with app.app_context():
        _comfy(tmp_path, cfg, consistency=True, lightning=True)
        assert qmh.qwen_ma_missing_assets() == []


# --- build_angle_prompt: byte-exact trigger vocabulary ------------------------

def test_build_angle_prompt_exact_string():
    from app.services import qwen_multiangle_helper as qmh
    p = qmh.build_angle_prompt('front view', 'eye-level shot', 'medium shot')
    assert p == '<sks> front view eye-level shot medium shot'
    p2 = qmh.build_angle_prompt('back view', 'high-angle shot', 'close-up')
    assert p2 == '<sks> back view high-angle shot close-up'


@pytest.mark.parametrize('azimuth,elevation,distance', [
    ('nonexistent view', 'eye-level shot', 'medium shot'),
    ('front view', 'nonexistent shot', 'medium shot'),
    ('front view', 'eye-level shot', 'nonexistent shot'),
])
def test_build_angle_prompt_rejects_unknown_tokens(azimuth, elevation, distance):
    from app.services import qwen_multiangle_helper as qmh
    with pytest.raises(ValueError):
        qmh.build_angle_prompt(azimuth, elevation, distance)


# --- enqueue_qwen_multiangle: graph mutation ----------------------------------

def test_enqueue_sets_prompt_source_and_loras(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import qwen_multiangle_helper as qmh
    with app.app_context():
        base = _comfy(tmp_path, cfg, consistency=True, lightning=True)
        src = tmp_path / 'source.png'
        src.write_bytes(b'\x89PNG\r\n\x1a\nfake')

        captured = {}

        def _fake_add_job(job_type='image', user_id='local', workflow_data=None,
                          prompt='', job_id=None, metadata=None):
            captured['workflow'] = workflow_data
            captured['prompt'] = prompt
            captured['metadata'] = metadata
            return job_id or 'job-1'

        monkeypatch.setattr(qmh.queue_manager, 'add_job', _fake_add_job)

        job_id = qmh.enqueue_qwen_multiangle(
            user_id='local', source_filename='source.png',
            azimuth='front view', elevation='eye-level shot', distance='medium shot',
            source_path=str(src))

        assert job_id
        wf = captured['workflow']
        assert wf['112']['inputs']['prompt'] == '<sks> front view eye-level shot medium shot'
        assert wf['115']['inputs']['image'].endswith('source.png')
        assert wf['108']['inputs']['unet_name']
        assert wf['95']['inputs']['vae_name'] == 'qwen_image_vae.safetensors'
        assert wf['93']['inputs']['clip_name'] == 'qwen_2.5_vl_7b_fp8_scaled.safetensors'
        assert wf['109']['inputs']['lora_name'] == 'qwen-image-edit-2511-multiple-angles-lora.safetensors'
        assert wf['109']['inputs']['strength_model'] == 1.0
        # consistency + Lightning present -> both stay in the chain, Lightning envelope active
        assert '114' in wf and wf['114']['inputs']['lora_name'] == 'consistence_edit_v2.safetensors'
        assert '102' in wf and wf['102']['inputs']['lora_name'] == \
            'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors'
        assert wf['106']['inputs']['steps'] == 4 and wf['106']['inputs']['cfg'] == 1
        assert captured['metadata']['model_name'] == 'qwen_multiangle_dataset'
        assert wf['121']['inputs']['filename_prefix'].startswith('local_QwenMultiangle_')


def test_enqueue_bypasses_consistency_and_lightning_when_absent(app, tmp_path, monkeypatch):
    """Neither optional LoRA is on disk -> both nodes are spliced out, and
    sampling falls back to the non-distilled step/cfg envelope."""
    from app import config as cfg
    from app.services import qwen_multiangle_helper as qmh
    with app.app_context():
        _comfy(tmp_path, cfg, consistency=False, lightning=False)
        src = tmp_path / 'source.png'
        src.write_bytes(b'\x89PNG\r\n\x1a\nfake')

        captured = {}
        monkeypatch.setattr(qmh.queue_manager, 'add_job',
                            lambda **kw: captured.update(kw) or (kw.get('job_id') or 'job-2'))

        qmh.enqueue_qwen_multiangle(
            user_id='local', source_filename='source.png',
            azimuth='left side view', elevation='low-angle shot', distance='wide shot',
            source_path=str(src))

        wf = captured['workflow_data']
        assert '114' not in wf   # consistency LoRA bypassed
        assert '102' not in wf   # Lightning LoRA bypassed
        assert wf['106']['inputs']['steps'] == 20 and wf['106']['inputs']['cfg'] == 4
        # the model chain still reaches the sampler through what's left
        assert wf['94']['inputs']['model'] == ['109', 0]


def test_enqueue_raises_when_multiangle_lora_missing(app, tmp_path):
    """The multi-angle LoRA is REQUIRED (unlike Klein's optional consistency
    LoRA) — without it this engine has no purpose."""
    from app import config as cfg
    from app.services import qwen_multiangle_helper as qmh
    with app.app_context():
        _comfy(tmp_path, cfg, multiangle=False)
        src = tmp_path / 'source.png'
        src.write_bytes(b'\x89PNG\r\n\x1a\nfake')
        with pytest.raises(qmh.QwenMultiangleModelsMissing) as exc:
            qmh.enqueue_qwen_multiangle(
                user_id='local', source_filename='source.png',
                azimuth='front view', elevation='eye-level shot', distance='medium shot',
                source_path=str(src))
        assert 'qwen_ma_multiangle_lora' in exc.value.missing


def test_enqueue_raises_on_missing_source_file(app, tmp_path):
    from app import config as cfg
    from app.services import qwen_multiangle_helper as qmh
    with app.app_context():
        _comfy(tmp_path, cfg)
        with pytest.raises(ValueError, match='source image not found'):
            qmh.enqueue_qwen_multiangle(
                user_id='local', source_filename='nope.png',
                azimuth='front view', elevation='eye-level shot', distance='medium shot',
                source_path=str(tmp_path / 'nope.png'))


# --- job-queue completion dispatch -------------------------------------------

def test_dispatch_completion_routes_qwen_multiangle_to_dataset_linker(app):
    """model_name == 'qwen_multiangle_dataset' reuses the SAME generic linker
    Klein dataset jobs use — no new linking logic needed."""
    from app.job_queue import queue_manager, _dispatch_completion
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QMA', 'qma')
        jid = queue_manager.add_job(workflow_data={'1': {}},
                                    metadata={'model_name': 'qwen_multiangle_dataset'})
        img = FaceDatasetImage(dataset_id=ds.id, source='generated', status='pending', job_id=jid)
        svc.db.session.add(img)
        svc.db.session.commit()

        from app.models import ImageGenerationQueue
        job = ImageGenerationQueue.query.filter_by(job_id=jid).one()
        with patch('app.services.face_dataset_service.link_completed_dataset_image') as linker:
            _dispatch_completion(job, 'out.png', False)
            linker.assert_called_once()
            assert linker.call_args[0][0] == jid
