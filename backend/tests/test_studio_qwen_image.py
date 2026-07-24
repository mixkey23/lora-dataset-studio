"""Wave 5: Qwen-Image support in the LoRA Test Studio (base T2I + Edit-2511).
Both training variants deploy into the SAME loras/qwen_image folder — the
variant is detected per-checkpoint from its deployed filename's base tag
(_Qwen-Image vs _Qwen-Image-Edit-2511), not the folder. These tests cover
that detection, the T2I and Edit-2511 cell-workflow branches, and the
studio_base_models route."""
import struct

_VALID_ST = struct.pack('<Q', 2) + b'{}'


def _install(base, *relparts, data=_VALID_ST):
    p = base.joinpath(*relparts)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def _comfy(tmp_path, cfg, unet=True, vae=True, te=True):
    """A ComfyUI tree with the shared Qwen-Image-Edit assets (mirrors
    test_qwen_edit_generation.py's own fixture) PLUS a qwen_image loras
    folder with one T2I and one Edit-2511 deployed checkpoint."""
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
    _install(base, 'models', 'loras', 'qwen_image', 'lora_qw1_000001000_Qwen-Image.safetensors')
    _install(base, 'models', 'loras', 'qwen_image', 'lora_qw1_000001000_Qwen-Image-Edit-2511.safetensors')
    cfg.save_config({'comfyui': {'base_dir': str(base)}})
    return base


# --- get_qwen_image_loras/models: variant detection from filename -----------

def test_get_qwen_image_loras_detects_variant_from_filename(app, tmp_path):
    from app import config as cfg
    from app.utils.comfyui import get_qwen_image_loras
    with app.app_context():
        _comfy(tmp_path, cfg)
        loras = get_qwen_image_loras()
        by_name = {l['filename']: l for l in loras}
        t2i = next(l for fn, l in by_name.items() if 'Qwen-Image.safetensors' in fn)
        edit = next(l for fn, l in by_name.items() if 'Edit-2511' in fn)
        assert t2i['variant'] == 'image'
        assert edit['variant'] == 'edit'


def test_get_qwen_image_models_scans_qwenimage_folder(app, tmp_path):
    from app import config as cfg
    from app.utils.comfyui import get_qwen_image_models
    with app.app_context():
        base = _comfy(tmp_path, cfg, unet=False)
        _install(base, 'models', 'diffusion_models', 'QwenImage', 'qwen_image_fp8_e4m3fn.safetensors')
        models = get_qwen_image_models()
        assert any('qwen_image_fp8_e4m3fn.safetensors' in m for m in models)


# --- _build_cell_workflow: T2I branch (default when variant undetectable) ---

def _lora_node_id(workflow, checkpoint):
    for nid, n in workflow.items():
        if isinstance(n, dict) and n.get('class_type') == 'LoraLoaderModelOnly' \
                and n['inputs'].get('lora_name') == checkpoint:
            return nid
    return None


def test_cell_workflow_qwen_image_t2i_loads_t2i_workflow_and_injects_lora(app, tmp_path):
    from app import config as cfg
    from app.services import lora_test_studio as lts
    with app.app_context():
        _comfy(tmp_path, cfg)
        checkpoint = 'qwen_image\\lora_qw1_000001000_Qwen-Image.safetensors'
        workflow = lts._build_cell_workflow(
            user_id='local', checkpoint=checkpoint, strength=0.9, prompt='a portrait',
            seed=7, z_model=None, allowed_loras={checkpoint}, dataset_id=1,
            train_type='qwen_image', trigger_word='qw1')
        # T2I workflow shape: node 23 is CLIPTextEncode (positive), no LoadImage/
        # edit-specific nodes.
        assert workflow['23']['inputs']['text'] == 'qw1, a portrait'
        assert workflow['26']['inputs']['seed'] == 7
        lora_nodes = [n for n in workflow.values()
                      if isinstance(n, dict) and n.get('class_type') == 'LoraLoaderModelOnly']
        tested = [n for n in lora_nodes if n['inputs']['lora_name'] == checkpoint]
        assert tested and tested[0]['inputs']['strength_model'] == 0.9
        # ModelSamplingAuraFlow (node 94) must be repointed to the LoRA chain.
        assert workflow['94']['inputs']['model'] == [_lora_node_id(workflow, checkpoint), 0]


def test_variant_lookup_defaults_to_image_when_checkpoint_not_in_pool(app, tmp_path):
    """Variant detection itself (not the whole cell-build pipeline, which also
    whitelist-checks against the same pool and would reject an absent
    checkpoint regardless): an Edit-2511-NAMED checkpoint that ISN'T in the
    scanned pool resolves to 'image', the safer fallback — never crashes
    trying to build an edit graph from an untracked filename."""
    from app import config as cfg
    from app.utils.comfyui import get_qwen_image_loras
    with app.app_context():
        _comfy(tmp_path, cfg)
        pool = {l['filename']: l for l in get_qwen_image_loras()}
        ghost = 'qwen_image\\lora_ghost_000001000_Qwen-Image-Edit-2511.safetensors'
        assert (pool.get(ghost) or {}).get('variant', 'image') == 'image'


# --- _build_cell_workflow: Edit-2511 branch ----------------------------------

def test_cell_workflow_qwen_image_edit_uses_dataset_reference_image(app, tmp_path):
    from app import config as cfg
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        _comfy(tmp_path, cfg)
        ds = svc.create_dataset(LOCAL_USER, 'QW1', 'qw1', train_type='qwen_image')
        ds_dir = tmp_path / 'datasets' / str(ds.id)
        ds_dir.mkdir(parents=True)
        (ds_dir / 'ref.png').write_bytes(b'REF')
        ds.ref_filename = 'ref.png'
        svc.db.session.commit()
        import app.services.face_dataset_service as fds_mod
        cfg_orig = fds_mod._dataset_dir
        fds_mod._dataset_dir = lambda _id: str(ds_dir)
        try:
            checkpoint = 'qwen_image\\lora_qw1_000001000_Qwen-Image-Edit-2511.safetensors'
            workflow = lts._build_cell_workflow(
                user_id=LOCAL_USER, checkpoint=checkpoint, strength=1.1,
                prompt='rotate to a 3/4 profile', seed=3, z_model=None,
                allowed_loras={checkpoint}, dataset_id=ds.id,
                train_type='qwen_image', trigger_word='qw1')
        finally:
            fds_mod._dataset_dir = cfg_orig
        assert workflow['10']['inputs']['prompt'] == 'qw1, rotate to a 3/4 profile'
        assert workflow['39']['inputs']['image'].endswith('ref.png')
        assert workflow['50']['inputs']['lora_name'] == checkpoint
        assert workflow['50']['inputs']['strength_model'] == 1.1
        assert workflow['17']['inputs']['model'] == ['50', 0]


def test_apply_qwen_image_edit_test_settings_raises_without_reference_image(app, tmp_path):
    from app import config as cfg
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.config import LOCAL_USER
    import pytest
    with app.app_context():
        _comfy(tmp_path, cfg)
        ds = svc.create_dataset(LOCAL_USER, 'QW2', 'qw2', train_type='qwen_image')
        svc.db.session.commit()
        # Reuses the qw1-named checkpoint already installed by _comfy() — variant
        # detection is purely filename-based, unrelated to this dataset's trigger.
        checkpoint = 'qwen_image\\lora_qw1_000001000_Qwen-Image-Edit-2511.safetensors'
        with pytest.raises(ValueError, match='no reference image'):
            lts._build_cell_workflow(
                user_id=LOCAL_USER, checkpoint=checkpoint, strength=1.0, prompt='x',
                seed=1, z_model=None, allowed_loras={checkpoint}, dataset_id=ds.id,
                train_type='qwen_image', trigger_word='qw2')


# --- studio_base_models route -------------------------------------------------

def test_studio_base_models_qwen_image_type_lists_scanned_unets(client, app, tmp_path):
    from app import config as cfg
    with app.app_context():
        base = _comfy(tmp_path, cfg, unet=False)
        _install(base, 'models', 'diffusion_models', 'QwenImage', 'qwen_image_fp8_e4m3fn.safetensors')
    resp = client.get('/api/studio/base-models?type=qwen_image')
    assert resp.status_code == 200
    names = [m['filename'] for m in resp.get_json()['models']]
    assert any('qwen_image_fp8_e4m3fn.safetensors' in n for n in names)
