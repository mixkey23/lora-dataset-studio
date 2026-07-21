"""Qwen Edit — the general-purpose "Generate variations" GENERATION engine
(Wave 4, redesigned after real-world debugging with the repo owner), a local
peer of Klein running Qwen-Image-Edit-2511 via a VLM-driven edit workflow
(TextEncodeQwenImageEditPlusCustom_lrzjason + QwenEditConfigPreparer +
QwenEditAdaptiveLongestEdge) instead of the Kontext-lineage nodes the first
cut used. No consistency LoRA (that LoRA fights drift across SEVERAL
reference images; this engine always starts from ONE). Shares its
UNET/VAE/text-encoder resolution with Qwen Multi-angle via the extracted
qwen_edit_assets module — the first half of this file is a REGRESSION check
that the extraction left Qwen Multi-angle's own behaviour unchanged,
mirroring test_qwen_multiangle.py's own fixtures."""
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


def _comfy(tmp_path, cfg, unet=True, vae=True, te=True, lightning=False, nsfw_checkpoint=False):
    """A ComfyUI tree with a configurable subset of the shared Qwen-Image-Edit
    assets present (mirrors test_qwen_multiangle.py's own fixture, minus the
    multi-angle LoRA and the consistency LoRA, neither of which
    qwen_edit_helper needs)."""
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
    if lightning:
        _install(base, 'models', 'loras', 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors')
    if nsfw_checkpoint:
        _install(base, 'models', 'checkpoints', 'Qwen', 'Qwen-Rapid-AIO-v1.safetensors')
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


def test_qwen_edit_assets_resolve_lightning_lora(app, tmp_path):
    from app import config as cfg
    from app.services import qwen_edit_assets as qea
    with app.app_context():
        _comfy(tmp_path, cfg, lightning=True)
        light_rel, light_path = qea.resolve_lightning_lora()
        assert light_rel == 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors'
        assert light_path is not None


def test_qwen_edit_assets_resolve_nsfw_checkpoint(app, tmp_path):
    import os
    from app import config as cfg
    from app.services import qwen_edit_assets as qea
    with app.app_context():
        _comfy(tmp_path, cfg, nsfw_checkpoint=True)
        assert qea.resolve_nsfw_checkpoint() == os.path.join('Qwen', 'Qwen-Rapid-AIO-v1.safetensors')


def test_qwen_edit_assets_resolve_nsfw_checkpoint_none_when_absent(app, tmp_path):
    from app import config as cfg
    from app.services import qwen_edit_assets as qea
    with app.app_context():
        _comfy(tmp_path, cfg, nsfw_checkpoint=False)
        assert qea.resolve_nsfw_checkpoint() is None


# --- qwen_edit_helper: missing-assets probe (no consistency LoRA anymore) ----

def test_qwen_edit_missing_assets_lists_every_absent_action(app, tmp_path):
    from app import config as cfg
    from app.services import qwen_edit_helper as qeh
    with app.app_context():
        _comfy(tmp_path, cfg, unet=False)
        missing = qeh.qwen_edit_missing_assets()
        assert 'qwen_edit_unet' in missing
        assert 'qwen_edit_vae' not in missing   # present in this fixture
        assert 'qwen_edit_lightning_lora' in missing   # not installed in this fixture
        assert 'qwen_edit_consistency_lora' not in missing   # not part of this engine anymore


def test_qwen_edit_missing_assets_empty_when_everything_present(app, tmp_path):
    from app import config as cfg
    from app.services import qwen_edit_helper as qeh
    with app.app_context():
        _comfy(tmp_path, cfg, lightning=True)
        assert qeh.qwen_edit_missing_assets() == []


def test_qwen_edit_recommended_has_no_consistency_lora():
    from app.services import qwen_edit_helper as qeh
    assert qeh.QWEN_EDIT_RECOMMENDED == ('qwen_edit_lightning_lora',)


# --- NSFW branch: separate checkpoint, fail-closed --------------------------

def test_qwen_edit_nsfw_missing_assets_when_checkpoint_absent(app, tmp_path):
    from app import config as cfg
    from app.services import qwen_edit_helper as qeh
    with app.app_context():
        _comfy(tmp_path, cfg, nsfw_checkpoint=False)
        assert qeh.qwen_edit_nsfw_missing_assets() == ['qwen_edit_nsfw_checkpoint']


def test_qwen_edit_nsfw_missing_assets_empty_when_checkpoint_present(app, tmp_path):
    from app import config as cfg
    from app.services import qwen_edit_helper as qeh
    with app.app_context():
        _comfy(tmp_path, cfg, nsfw_checkpoint=True)
        assert qeh.qwen_edit_nsfw_missing_assets() == []


def test_enqueue_nsfw_raises_when_checkpoint_missing_never_falls_back_to_sfw(app, tmp_path):
    """Fail-closed: an NSFW shot with no Rapid-AIO checkpoint configured must
    NEVER silently render on the (likely-censored) SFW checkpoint."""
    from app import config as cfg
    from app.services import qwen_edit_helper as qeh
    with app.app_context():
        _comfy(tmp_path, cfg, nsfw_checkpoint=False)
        src = tmp_path / 'source.png'
        src.write_bytes(b'\x89PNG\r\n\x1a\nfake')
        with pytest.raises(qeh.QwenEditModelsMissing) as exc:
            qeh.enqueue_qwen_edit_variation(
                user_id='local', source_filename='source.png',
                edit_prompt='test prompt', source_path=str(src), nsfw=True)
        assert exc.value.missing == ['qwen_edit_nsfw_checkpoint']


def test_enqueue_nsfw_uses_the_nsfw_graph_not_the_sfw_one(app, tmp_path, monkeypatch):
    import os
    from app import config as cfg
    from app.services import qwen_edit_helper as qeh
    with app.app_context():
        _comfy(tmp_path, cfg, nsfw_checkpoint=True)
        src = tmp_path / 'source.png'
        src.write_bytes(b'\x89PNG\r\n\x1a\nfake')

        captured = {}
        monkeypatch.setattr(qeh.queue_manager, 'add_job',
                            lambda **kw: captured.update(kw) or (kw.get('job_id') or 'job-nsfw'))

        qeh.enqueue_qwen_edit_variation(
            user_id='local', source_filename='source.png',
            edit_prompt='nsfw test prompt', negative_prompt='lowres',
            source_path=str(src), nsfw=True)

        wf = captured['workflow_data']
        # NSFW graph shape: CheckpointLoaderSimple (node 1), NOT the SFW
        # graph's split UNETLoader/CLIPLoader/VAELoader.
        assert wf['1']['class_type'] == 'CheckpointLoaderSimple'
        assert wf['1']['inputs']['ckpt_name'] == os.path.join('Qwen', 'Qwen-Rapid-AIO-v1.safetensors')
        assert wf['3']['inputs']['prompt'] == 'nsfw test prompt'
        assert wf['4']['inputs']['prompt'] == 'lowres'
        assert wf['8']['inputs']['image'].endswith('source.png')
        assert wf['6']['inputs']['filename_prefix'].startswith('local_QwenEditNSFW_')
        # no lrzjason/VLM nodes from the SFW graph in this one
        assert '10' not in wf and '13' not in wf


def test_workflow_nsfw_required_nodes_all_present_in_shipped_file():
    from app.services import qwen_edit_helper as qeh
    from app.utils.comfyui import load_workflow_local
    workflow = load_workflow_local(str(qeh.WORKFLOW_QWEN_EDIT_NSFW_PATH))
    for node in qeh._REQUIRED_NODES_NSFW:
        assert node in workflow, f'required node {node} missing from qwen_edit_variation_nsfw.json'


# --- render_style_presets: negative prompt ------------------------------------

def test_generation_negative_for_photoreal_avoids_stylization():
    from app.services.render_style_presets import generation_negative_for
    neg = generation_negative_for('photoreal')
    assert 'cartoon' in neg and 'anime' in neg
    assert 'photorealistic' not in neg


def test_generation_negative_for_non_photoreal_avoids_photorealism():
    from app.services.render_style_presets import generation_negative_for
    neg = generation_negative_for('anime_2d')
    assert 'photorealistic' in neg
    neg_3d = generation_negative_for('render_3d')
    assert neg_3d == neg   # every stylized target shares the same anti-photo direction


def test_generation_negative_for_never_empty():
    from app.services.render_style_presets import generation_negative_for
    for style in ('photoreal', 'render_3d', 'anime_2d', 'cartoon_semireal',
                  'illustration', 'custom', 'unknown-garbage', None):
        assert generation_negative_for(style)


# --- wrap_variation_qwen_edit: edit-instruction shape + render_style + nsfw --

def test_wrap_variation_qwen_edit_is_an_edit_instruction_not_a_creation(app):
    from app.services.face_variations import wrap_variation_qwen_edit
    with app.app_context():
        out = wrap_variation_qwen_edit('full body shot, standing, front view', framing='body')
        assert out.startswith('Keep the same character identity. Restage the image as:')
        assert 'Create a new' not in out


def test_wrap_variation_qwen_edit_identity_clause_present(app):
    from app.services.face_variations import wrap_variation_qwen_edit
    with app.app_context():
        out = wrap_variation_qwen_edit('sitting at a cafe table', framing='bust')
        assert 'Do not change the facial identity' in out
        assert 'eye shape and color' in out


def test_wrap_variation_qwen_edit_non_photoreal_appends_style_tail(app):
    from app.services.face_variations import wrap_variation_qwen_edit
    with app.app_context():
        out = wrap_variation_qwen_edit('sitting at a cafe table', framing='bust',
                                       render_style='anime_2d')
        assert 'cel-shaded 2D anime' in out
        assert 'Professional realistic photograph' not in out


def test_wrap_variation_qwen_edit_photoreal_default_mentions_photograph(app):
    from app.services.face_variations import wrap_variation_qwen_edit
    with app.app_context():
        out = wrap_variation_qwen_edit('sitting at a cafe table', framing='bust')
        assert 'Professional realistic photograph' in out
        assert out.rstrip().endswith('SFW.')


def test_wrap_variation_qwen_edit_nsfw_drops_sfw_clamp(app):
    from app.services.face_variations import wrap_variation_qwen_edit
    with app.app_context():
        out = wrap_variation_qwen_edit('lying on a bed', framing='body', nsfw=True)
        assert 'Explicit nudity is allowed' in out
        assert not out.rstrip().endswith('SFW.')


def test_identity_prompts_override_wins_over_qwen_edit_default(app):
    """The user's global identity_prompts.qwen_edit_identity override beats
    the shipped default, same precedence rule every identity-prompt kind has."""
    from app import config as cfg
    from app.services.face_variations import wrap_variation_qwen_edit
    with app.app_context():
        cfg.save_config({'identity_prompts': {'qwen_edit_identity': 'MY CUSTOM GUARD TEXT'}})
        out = wrap_variation_qwen_edit('sitting at a cafe table', framing='bust',
                                       render_style='anime_2d')
        assert 'MY CUSTOM GUARD TEXT' in out


# --- enqueue_qwen_edit_variation: graph mutation (new node ids, no cons LoRA) -

def test_enqueue_sets_prompt_negative_source_and_model_files(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import qwen_edit_helper as qeh
    with app.app_context():
        _comfy(tmp_path, cfg, lightning=True)
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
            edit_prompt='Keep the same character identity. Restage as: test prompt.',
            negative_prompt='lowres, worst quality',
            source_path=str(src))

        assert job_id
        wf = captured['workflow']
        assert wf['10']['inputs']['prompt'] == \
            'Keep the same character identity. Restage as: test prompt.'
        assert wf['21']['inputs']['text'] == 'lowres, worst quality'
        assert wf['39']['inputs']['image'].endswith('source.png')
        assert wf['31']['inputs']['unet_name']
        assert wf['7']['inputs']['vae_name'] == 'qwen_image_vae.safetensors'
        assert wf['6']['inputs']['clip_name'] == 'qwen_2.5_vl_7b_fp8_scaled.safetensors'
        # Lightning present -> stays in the chain, Lightning envelope active
        assert '23' in wf and wf['23']['inputs']['lora_name'] == \
            'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors'
        assert wf['17']['inputs']['steps'] == 8 and wf['17']['inputs']['cfg'] == 1
        # dynamic aspect-ratio sizing stays wired statically (GetImageSize -> EmptyLatentImage)
        assert wf['33']['inputs']['width'] == ['40', 0]
        assert wf['33']['inputs']['height'] == ['40', 1]
        assert wf['40']['inputs']['image'] == ['8', 0]
        assert captured['metadata']['model_name'] == 'qwen_edit_dataset'
        assert wf['38']['inputs']['filename_prefix'].startswith('local_QwenEdit_')
        # no consistency LoRA node in this graph
        assert '114' not in wf


def test_enqueue_negative_prompt_defaults_to_empty_string(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import qwen_edit_helper as qeh
    with app.app_context():
        _comfy(tmp_path, cfg)
        src = tmp_path / 'source.png'
        src.write_bytes(b'\x89PNG\r\n\x1a\nfake')
        captured = {}
        monkeypatch.setattr(qeh.queue_manager, 'add_job',
                            lambda **kw: captured.update(kw) or (kw.get('job_id') or 'job-x'))
        qeh.enqueue_qwen_edit_variation(
            user_id='local', source_filename='source.png',
            edit_prompt='test prompt', source_path=str(src))
        assert captured['workflow_data']['21']['inputs']['text'] == ''


def test_enqueue_bypasses_lightning_lora_when_absent(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import qwen_edit_helper as qeh
    with app.app_context():
        _comfy(tmp_path, cfg, lightning=False)
        src = tmp_path / 'source.png'
        src.write_bytes(b'\x89PNG\r\n\x1a\nfake')

        captured = {}
        monkeypatch.setattr(qeh.queue_manager, 'add_job',
                            lambda **kw: captured.update(kw) or (kw.get('job_id') or 'job-2'))

        qeh.enqueue_qwen_edit_variation(
            user_id='local', source_filename='source.png',
            edit_prompt='test prompt', source_path=str(src))

        wf = captured['workflow_data']
        assert '23' not in wf   # Lightning LoRA bypassed
        assert wf['17']['inputs']['steps'] == 20 and wf['17']['inputs']['cfg'] == 4
        # KSampler.model reconnects straight to node 5 (CFGNorm), skipping the
        # bypassed LoRA node — the chain is 31(UNET)->4->5->[23 skipped]->17.
        assert wf['17']['inputs']['model'] == ['5', 0]


def test_enqueue_lightning_disabled_explicitly_bypasses_even_when_file_present(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import qwen_edit_helper as qeh
    with app.app_context():
        _comfy(tmp_path, cfg, lightning=True)
        src = tmp_path / 'source.png'
        src.write_bytes(b'\x89PNG\r\n\x1a\nfake')
        captured = {}
        monkeypatch.setattr(qeh.queue_manager, 'add_job',
                            lambda **kw: captured.update(kw) or (kw.get('job_id') or 'job-3'))
        qeh.enqueue_qwen_edit_variation(
            user_id='local', source_filename='source.png',
            edit_prompt='test prompt', source_path=str(src), lightning_enabled=False)
        wf = captured['workflow_data']
        assert '23' not in wf
        assert wf['17']['inputs']['steps'] == 20 and wf['17']['inputs']['cfg'] == 4


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


def test_workflow_required_nodes_all_present_in_shipped_file():
    from app.services import qwen_edit_helper as qeh
    from app.utils.comfyui import load_workflow_local
    workflow = load_workflow_local(str(qeh.WORKFLOW_QWEN_EDIT_PATH))
    for node in qeh._REQUIRED_NODES:
        assert node in workflow, f'required node {node} missing from qwen_edit_variation.json'


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
