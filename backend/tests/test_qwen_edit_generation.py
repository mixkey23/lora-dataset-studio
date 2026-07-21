"""Qwen Edit — the general-purpose "Generate variations" GENERATION engine
(Wave 4), a local peer of Klein running Qwen-Image-Edit-2511. Shares its
UNET/VAE/text-encoder/consistency-LoRA resolution with Qwen Multi-angle via
the extracted qwen_edit_assets module — the first half of this file is a
REGRESSION check that the extraction left Qwen Multi-angle's own behaviour
unchanged, mirroring test_qwen_multiangle.py's own fixtures."""
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


def _comfy(tmp_path, cfg, unet=True, vae=True, te=True,
          consistency=False, lightning=False):
    """A ComfyUI tree with a configurable subset of the shared Qwen-Image-Edit
    assets present (mirrors test_qwen_multiangle.py's own fixture, minus the
    multi-angle LoRA, which qwen_edit_helper doesn't need)."""
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
    if consistency:
        _install(base, 'models', 'loras', 'consistence_edit_v2.safetensors')
    if lightning:
        _install(base, 'models', 'loras', 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors')
    cfg.save_config({'comfyui': {'base_dir': str(base)}})
    return base


# --- Shared resolver extraction: Qwen Multi-angle stays behaviour-identical --

def test_multiangle_resolvers_unchanged_after_extraction(app, tmp_path):
    import os
    from app import config as cfg
    from app.services import qwen_multiangle_helper as qmh
    with app.app_context():
        _comfy(tmp_path, cfg)
        assert qmh.resolve_qwen_ma_unet() == os.path.join(
            'QwenImage', 'Qwen-Image-Edit-2511-FP8_e4m3fn.safetensors')
        assert qmh.resolve_qwen_ma_vae() == 'qwen_image_vae.safetensors'
        assert qmh.resolve_qwen_ma_text_encoder() == 'qwen_2.5_vl_7b_fp8_scaled.safetensors'


def test_multiangle_consistency_lora_resolver_unchanged_after_extraction(app, tmp_path):
    from app import config as cfg
    from app.services import qwen_multiangle_helper as qmh
    with app.app_context():
        _comfy(tmp_path, cfg, consistency=True)
        rel, path = qmh.resolve_qwen_ma_consistency_lora()
        assert rel == 'consistence_edit_v2.safetensors'
        assert path is not None


# --- qwen_edit_assets: shared resolvers ---------------------------------------

def test_qwen_edit_assets_resolve_unet_vae_text_encoder(app, tmp_path):
    import os
    from app import config as cfg
    from app.services import qwen_edit_assets as qea
    with app.app_context():
        _comfy(tmp_path, cfg)
        assert qea.resolve_unet() == os.path.join(
            'QwenImage', 'Qwen-Image-Edit-2511-FP8_e4m3fn.safetensors')
        assert qea.resolve_vae() == 'qwen_image_vae.safetensors'
        assert qea.resolve_text_encoder() == 'qwen_2.5_vl_7b_fp8_scaled.safetensors'


def test_qwen_edit_assets_resolve_consistency_and_lightning_loras(app, tmp_path):
    from app import config as cfg
    from app.services import qwen_edit_assets as qea
    with app.app_context():
        _comfy(tmp_path, cfg, consistency=True, lightning=True)
        cons_rel, cons_path = qea.resolve_consistency_lora()
        assert cons_rel == 'consistence_edit_v2.safetensors'
        assert cons_path is not None
        light_rel, light_path = qea.resolve_lightning_lora()
        assert light_rel == 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors'
        assert light_path is not None


# --- qwen_edit_helper: missing-assets probe -----------------------------------

def test_qwen_edit_missing_assets_lists_every_absent_action(app, tmp_path):
    from app import config as cfg
    from app.services import qwen_edit_helper as qeh
    with app.app_context():
        _comfy(tmp_path, cfg, unet=False)
        missing = qeh.qwen_edit_missing_assets()
        assert 'qwen_edit_unet' in missing
        assert 'qwen_edit_vae' not in missing   # present in this fixture
        # optional assets are missing by default (not installed in this fixture)
        assert 'qwen_edit_consistency_lora' in missing
        assert 'qwen_edit_lightning_lora' in missing


def test_qwen_edit_missing_assets_empty_when_everything_present(app, tmp_path):
    from app import config as cfg
    from app.services import qwen_edit_helper as qeh
    with app.app_context():
        _comfy(tmp_path, cfg, consistency=True, lightning=True)
        assert qeh.qwen_edit_missing_assets() == []


# --- wrap_variation_qwen_edit: render_style tail-swap + identity override ----

def test_wrap_variation_qwen_edit_photoreal_default_has_photo_wording(app):
    from app.services.face_variations import wrap_variation_qwen_edit
    with app.app_context():
        out = wrap_variation_qwen_edit('sitting at a cafe table', framing='bust')
        assert 'same character' not in out
        assert 'photograph' in out.lower()
        assert 'SFW' in out


def test_wrap_variation_qwen_edit_non_photoreal_swaps_person_to_character(app):
    from app.services.face_variations import wrap_variation_qwen_edit
    with app.app_context():
        out = wrap_variation_qwen_edit('sitting at a cafe table', framing='bust',
                                       render_style='anime_2d')
        assert 'same character' in out
        assert 'skin tone and texture' not in out
        assert 'photograph' not in out.lower()


def test_wrap_variation_qwen_edit_nsfw_drops_sfw_clamp(app):
    from app.services.face_variations import wrap_variation_qwen_edit
    with app.app_context():
        out = wrap_variation_qwen_edit('lying on a bed', framing='body', nsfw=True)
        assert 'Explicit nudity is allowed' in out
        assert out.strip().endswith('SFW.') is False


def test_identity_prompts_override_wins_over_qwen_edit_default(app):
    """The user's global identity_prompts.qwen_edit_identity override beats
    BOTH the shipped default and the render_style tail — same precedence rule
    Klein's own identity guard already has."""
    from app import config as cfg
    from app.services.face_variations import wrap_variation_qwen_edit
    with app.app_context():
        cfg.save_config({'identity_prompts': {'qwen_edit_identity': 'MY CUSTOM GUARD TEXT'}})
        out = wrap_variation_qwen_edit('sitting at a cafe table', framing='bust',
                                       render_style='anime_2d')
        assert 'MY CUSTOM GUARD TEXT' in out


# --- enqueue_qwen_edit_variation: graph mutation ------------------------------

def test_enqueue_sets_prompt_source_and_model_files(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import qwen_edit_helper as qeh
    with app.app_context():
        _comfy(tmp_path, cfg, consistency=True, lightning=True)
        src = tmp_path / 'source.png'
        src.write_bytes(b'\x89PNG\r\n\x1a\nfake')

        captured = {}

        def _fake_add_job(job_type='image', user_id='local', workflow_data=None,
                          prompt='', job_id=None, metadata=None):
            captured['workflow'] = workflow_data
            captured['prompt'] = prompt
            captured['metadata'] = metadata
            return job_id or 'job-1'

        monkeypatch.setattr(qeh.queue_manager, 'add_job', _fake_add_job)

        job_id = qeh.enqueue_qwen_edit_variation(
            user_id='local', source_filename='source.png',
            edit_prompt='Create a new image of the same character: test prompt',
            source_path=str(src))

        assert job_id
        wf = captured['workflow']
        assert wf['112']['inputs']['prompt'] == \
            'Create a new image of the same character: test prompt'
        assert wf['115']['inputs']['image'].endswith('source.png')
        assert wf['108']['inputs']['unet_name']
        assert wf['95']['inputs']['vae_name'] == 'qwen_image_vae.safetensors'
        assert wf['93']['inputs']['clip_name'] == 'qwen_2.5_vl_7b_fp8_scaled.safetensors'
        # consistency + Lightning present -> both stay in the chain, Lightning envelope active
        assert '114' in wf and wf['114']['inputs']['lora_name'] == 'consistence_edit_v2.safetensors'
        assert '102' in wf and wf['102']['inputs']['lora_name'] == \
            'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors'
        assert wf['106']['inputs']['steps'] == 4 and wf['106']['inputs']['cfg'] == 1
        assert captured['metadata']['model_name'] == 'qwen_edit_dataset'
        assert wf['121']['inputs']['filename_prefix'].startswith('local_QwenEdit_')


def test_enqueue_bypasses_consistency_lora_at_strength_zero(app, tmp_path, monkeypatch):
    """Same degrade-not-fail contract as Klein: an explicit 0 strength bypasses
    the node rather than failing the job, even when the file IS present."""
    from app import config as cfg
    from app.services import qwen_edit_helper as qeh
    with app.app_context():
        _comfy(tmp_path, cfg, consistency=True, lightning=True)
        src = tmp_path / 'source.png'
        src.write_bytes(b'\x89PNG\r\n\x1a\nfake')

        captured = {}
        monkeypatch.setattr(qeh.queue_manager, 'add_job',
                            lambda **kw: captured.update(kw) or (kw.get('job_id') or 'job-2'))

        qeh.enqueue_qwen_edit_variation(
            user_id='local', source_filename='source.png',
            edit_prompt='test prompt', source_path=str(src), lora_strength=0)

        wf = captured['workflow_data']
        assert '114' not in wf   # consistency LoRA bypassed
        assert '102' in wf       # Lightning LoRA untouched


def test_enqueue_bypasses_both_optional_loras_when_absent(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import qwen_edit_helper as qeh
    with app.app_context():
        _comfy(tmp_path, cfg, consistency=False, lightning=False)
        src = tmp_path / 'source.png'
        src.write_bytes(b'\x89PNG\r\n\x1a\nfake')

        captured = {}
        monkeypatch.setattr(qeh.queue_manager, 'add_job',
                            lambda **kw: captured.update(kw) or (kw.get('job_id') or 'job-3'))

        qeh.enqueue_qwen_edit_variation(
            user_id='local', source_filename='source.png',
            edit_prompt='test prompt', source_path=str(src))

        wf = captured['workflow_data']
        assert '114' not in wf   # consistency LoRA bypassed
        assert '102' not in wf   # Lightning LoRA bypassed
        assert wf['106']['inputs']['steps'] == 20 and wf['106']['inputs']['cfg'] == 4
        # the model chain still reaches the sampler through what's left
        assert wf['94']['inputs']['model'] == ['108', 0]


def test_enqueue_raises_when_unet_missing(app, tmp_path):
    from app import config as cfg
    from app.services import qwen_edit_helper as qeh
    with app.app_context():
        _comfy(tmp_path, cfg, unet=False)
        src = tmp_path / 'source.png'
        src.write_bytes(b'\x89PNG\r\n\x1a\nfake')
        with pytest.raises(qeh.QwenEditModelsMissing) as exc:
            qeh.enqueue_qwen_edit_variation(
                user_id='local', source_filename='source.png',
                edit_prompt='test prompt', source_path=str(src))
        assert 'qwen_edit_unet' in exc.value.missing


def test_enqueue_raises_on_missing_source_file(app, tmp_path):
    from app import config as cfg
    from app.services import qwen_edit_helper as qeh
    with app.app_context():
        _comfy(tmp_path, cfg)
        with pytest.raises(ValueError, match='source image not found'):
            qeh.enqueue_qwen_edit_variation(
                user_id='local', source_filename='nope.png',
                edit_prompt='test prompt', source_path=str(tmp_path / 'nope.png'))


# --- job-queue completion dispatch -------------------------------------------

def test_dispatch_completion_routes_qwen_edit_to_dataset_linker(app):
    """model_name == 'qwen_edit_dataset' reuses the SAME generic linker every
    other local engine uses — no new linking logic needed."""
    from app.job_queue import queue_manager, _dispatch_completion
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QE', 'qe')
        jid = queue_manager.add_job(workflow_data={'1': {}},
                                    metadata={'model_name': 'qwen_edit_dataset'})
        img = FaceDatasetImage(dataset_id=ds.id, source='generated', status='pending', job_id=jid)
        svc.db.session.add(img)
        svc.db.session.commit()

        from app.models import ImageGenerationQueue
        job = ImageGenerationQueue.query.filter_by(job_id=jid).one()
        with patch('app.services.face_dataset_service.link_completed_dataset_image') as linker:
            _dispatch_completion(job, 'out.png', False)
            linker.assert_called_once()
            assert linker.call_args[0][0] == jid
