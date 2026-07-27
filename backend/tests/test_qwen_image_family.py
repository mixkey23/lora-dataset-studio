"""Qwen-Image ('qwen_image') — the 6th training family (ai-toolkit engine).

What matters here: the family is accepted end-to-end, the two model targets
(base T2I default / Edit-2511 opt-in) drive arch + name_or_path + distinct run
tags (their weights are incompatible — shared folders would corrupt a
resume), the possibly-extension arch guard refuses a launch on an ai-toolkit
that lacks the qwen_image arch (silent SD-loader fallback otherwise), and the
cloud path stays CLOSED for this family (local-only this wave, unlike
flux2klein)."""
import os

import pytest


def _configure_aitoolkit(tmp_path, app, supports_qwen=True, supports_qwen_edit=True):
    """Fake ai-toolkit install (venv python + run.py), with or without the
    qwen_image/qwen_image_edit archs under extensions_built_in — what the two
    support guards actually scan. Qwen-Image and Qwen-Image-Edit are SEPARATE
    ai-toolkit classes (QwenImageModel vs QwenImageEditModel) — a real install
    can have one without the other if Edit support landed in a later ai-toolkit
    version, hence the two independent flags."""
    from app import config as cfg
    root = tmp_path / 'aitoolkit'
    # POSIX venv layout (pre-existing bug fixed 2026-07-26: a hardcoded
    # Windows-only 'venv/Scripts/python.exe' never resolved on a POSIX test
    # runner — aitoolkit_path('venv_python') only looks under 'venv/bin/python'
    # when os.name != 'nt', silently leaving every launch_training() call in
    # this file raising "ai-toolkit is not configured" instead of exercising
    # the arch guards these tests are actually about).
    (root / 'venv' / 'bin').mkdir(parents=True)
    (root / 'venv' / 'bin' / 'python').write_text('fake')
    (root / 'run.py').write_text('fake')
    ext = root / 'extensions_built_in' / 'diffusion_models' / 'qwen_image'
    ext.mkdir(parents=True)
    if supports_qwen:
        (ext / 'qwen_image_model.py').write_text(
            'class QwenImageModel:\n    arch = "qwen_image"\n', encoding='utf-8')
    else:
        # a sibling extension only — an incidental 'qwen' mention must NOT count.
        (ext / 'other_model.py').write_text(
            '# qwen support not merged yet\n'
            'class OtherModel:\n    arch = "other"\n', encoding='utf-8')
    if supports_qwen_edit:
        (ext / 'qwen_image_edit.py').write_text(
            'class QwenImageEditModel(QwenImageModel):\n    arch = "qwen_image_edit"\n',
            encoding='utf-8')
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(root)}})
    return root


# --- 1) train_type accepted / normalized ---------------------------------------

def test_normalize_train_type_accepts_qwen_image_unknown_stays_zimage():
    from app.services import face_dataset_service as svc
    assert svc.normalize_train_type('qwen_image') == 'qwen_image'
    assert svc.normalize_train_type('QWEN_IMAGE') == 'qwen_image'   # case-fold
    assert svc.normalize_train_type('bogus') == 'zimage'            # unknown -> default
    assert svc.normalize_train_type(None) == 'zimage'


# --- 2) extension-arch guard + actionable launch refusal ------------------------

def test_aitoolkit_supports_qwen_image_scans_extension_archs(app, tmp_path):
    from app.services import lora_training as lt
    _configure_aitoolkit(tmp_path, app, supports_qwen=True)
    with app.app_context():
        assert lt._aitoolkit_supports_qwen_image() is True


def test_aitoolkit_supports_qwen_image_false_without_arch(app, tmp_path):
    """No qwen_image arch on disk -> False, even with 'qwen' mentioned in a
    comment (exact-arch match, no substring false positive). Unconfigured
    ai-toolkit -> False too."""
    from app.services import lora_training as lt
    with app.app_context():
        assert lt._aitoolkit_supports_qwen_image() is False   # not configured
    _configure_aitoolkit(tmp_path, app, supports_qwen=False)
    with app.app_context():
        assert lt._aitoolkit_supports_qwen_image() is False


def test_launch_refuses_qwen_image_when_arch_missing(app, tmp_path, monkeypatch):
    """Without the guard, get_model_class would silently fall back to the SD
    legacy loader -> corrupted LoRA. The refusal must be actionable (git pull)."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app, supports_qwen=False)
    monkeypatch.setattr(lt.shutil, 'disk_usage',
                        lambda p: type('u', (), {'free': 500e9})())
    with app.app_context():
        # qwen_image defaults to the musubi ENGINE — force ai-toolkit explicitly
        # so this actually exercises the ai-toolkit arch guard under test.
        ds = svc.create_dataset(LOCAL_USER, 'QI', 'zchar_qi', train_type='qwen_image')
        with pytest.raises(ValueError, match=r'update it \(git pull\)'):
            lt.launch_training(LOCAL_USER, ds.id, check_captions=False, engine='aitoolkit')
        # Same guard on the queue path — no deferred job doomed to the fallback.
        monkeypatch.setattr(lt, 'assert_trainable', lambda *_a, **_kw: None)
        with pytest.raises(ValueError, match=r'update it \(git pull\)'):
            lt.enqueue_training(LOCAL_USER, ds.id, extra_steps=100, engine='aitoolkit')


def test_aitoolkit_supports_qwen_image_edit_scans_extension_archs(app, tmp_path):
    from app.services import lora_training as lt
    _configure_aitoolkit(tmp_path, app, supports_qwen=True, supports_qwen_edit=True)
    with app.app_context():
        assert lt._aitoolkit_supports_qwen_image_edit() is True


def test_aitoolkit_supports_qwen_image_edit_false_without_arch(app, tmp_path):
    """Base Qwen-Image present but the Edit class isn't (bug fixed 2026-07-26:
    they're separate ai-toolkit classes) -> False, distinct from the base
    guard which is True here."""
    from app.services import lora_training as lt
    _configure_aitoolkit(tmp_path, app, supports_qwen=True, supports_qwen_edit=False)
    with app.app_context():
        assert lt._aitoolkit_supports_qwen_image() is True
        assert lt._aitoolkit_supports_qwen_image_edit() is False


def test_launch_refuses_qwen_image_edit_when_edit_arch_missing_but_base_present(app, tmp_path, monkeypatch):
    """Bug fixed 2026-07-26: an ai-toolkit with the base Qwen-Image class but
    not yet the Edit one used to silently train the DEFAULT (edit) variant
    through the wrong class (arch='qwen_image', which never reads
    batch.control_tensor) instead of refusing. The base T2I variant, which
    doesn't need the Edit class at all, must still be allowed to proceed past
    this specific guard."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app, supports_qwen=True, supports_qwen_edit=False)
    monkeypatch.setattr(lt.shutil, 'disk_usage',
                        lambda p: type('u', (), {'free': 500e9})())
    with app.app_context():
        # qwen_image defaults to the musubi ENGINE — force ai-toolkit explicitly
        # so this actually exercises the ai-toolkit arch guard under test.
        ds = svc.create_dataset(LOCAL_USER, 'QIE', 'zchar_qie', train_type='qwen_image')
        with pytest.raises(ValueError, match=r'qwen_image_edit arch missing'):
            lt.launch_training(LOCAL_USER, ds.id, check_captions=False, engine='aitoolkit')
        monkeypatch.setattr(lt, 'assert_trainable', lambda *_a, **_kw: None)
        with pytest.raises(ValueError, match=r'qwen_image_edit arch missing'):
            lt.enqueue_training(LOCAL_USER, ds.id, extra_steps=100, engine='aitoolkit')
        # Base T2I variant doesn't need the Edit class -> a DIFFERENT failure
        # further down the pipeline (no kept images), never this guard.
        with pytest.raises(ValueError, match=r'no kept images'):
            lt.launch_training(LOCAL_USER, ds.id, check_captions=False,
                              engine='aitoolkit', variant='image')


# --- 3) job config: base default / Edit-2511 opt-in -----------------------------

def test_build_job_config_qwen_image_edit_default_and_base_optin(app, tmp_path):
    """'edit' (Qwen-Image-Edit-2511) is the default (repo owner's product call,
    since musubi's Qwen-Image tooling here targets Edit-2511); 'image' (base
    T2I) is the opt-out. Non-distilled base -> real CFG previews.

    `arch` DIFFERS between the two targets (bug fixed 2026-07-26): ai-toolkit
    registers Qwen-Image and Qwen-Image-Edit as separate classes
    (QwenImageModel/'qwen_image' vs QwenImageEditModel/'qwen_image_edit' -
    only the latter reads batch.control_tensor), verified directly against
    ai-toolkit's own extensions_built_in/diffusion_models/qwen_image/
    qwen_image_edit.py source."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = svc.create_dataset(LOCAL_USER, 'Qee', 'zchar_qee', train_type='qwen_image')
        folder = tmp_path / 'ds'; folder.mkdir()

        assert lt._qwen_image_is_edit(ds) is True           # no variant -> edit (default)
        p = lt.build_job_config(ds, str(folder), steps=1500)['config']['process'][0]
        m = p['model']
        assert m['arch'] == 'qwen_image_edit'
        assert m['name_or_path'] == 'Qwen/Qwen-Image-Edit-2511'
        assert m['quantize'] is True and m['quantize_te'] is True
        assert m['low_vram'] is True and m['qtype'] == 'qfloat8'
        assert p['train']['timestep_type'] == 'sigmoid'
        assert p['train']['noise_scheduler'] == 'flowmatch'
        assert p['sample']['sampler'] == 'flowmatch'
        assert p['sample']['guidance_scale'] == 4 and p['sample']['sample_steps'] == 25
        assert p['datasets'][0]['caption_ext'] == 'txt'
        assert p['network'] == {'type': 'lora', 'linear': 32, 'linear_alpha': 32}

        ds.train_variant = 'image'
        svc.db.session.commit()
        assert lt._qwen_image_is_edit(ds) is False
        pi = lt.build_job_config(ds, str(folder), steps=1500)['config']['process'][0]
        assert pi['model']['arch'] == 'qwen_image'
        assert pi['model']['name_or_path'] == 'Qwen/Qwen-Image'


def test_qwen_image_expects_prose_captions(app):
    """Everything != sdxl expects prose: booru-tag captions on a qwen_image
    dataset trip the MISMATCH_CAPTION guard (forceable, like the others). The
    message must name the REAL family (bug fixed 2026-07-26: it was hardcoded
    to say "Z-Image" regardless of ttype, confusing anyone training a
    different family — Qwen-Image included — into thinking the app had
    somehow lost track of what it was training)."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QB', 'zchar_qb', train_type='qwen_image')
        booru = '1girl, solo, cafe, sitting, window, jeans, smile, looking_at_viewer'
        # 15 kept = the qwen_image family floor, so the caption-mismatch guard is
        # tested without tripping the readiness image-floor guard (below 15).
        for _ in range(15):
            svc.db.session.add(FaceDatasetImage(dataset_id=ds.id, status='keep',
                                                filename='x.webp', caption=booru))
        svc.db.session.commit()
        with pytest.raises(ValueError, match=r'MISMATCH_CAPTION: this Qwen-Image dataset') as exc:
            lt.assert_trainable(ds.id, train_type='qwen_image')
        assert 'Z-Image' not in str(exc.value)
        lt.assert_trainable(ds.id, train_type='qwen_image', allow_caption_mismatch=True)


# --- 4) distinct run tags per target (no image/edit telescoping) ---------------

def test_dest_base_tag_distinct_for_image_and_edit(app):
    """Base and Edit-2511 are incompatible checkpoints: same trigger, two
    targets -> two run folders / deployed names. A shared tag would make
    ai-toolkit auto-resume across targets (corrupted LoRA)."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QT', 'zchar_qt', train_type='qwen_image')
        tag_edit = lt._dest_base_tag(ds)                       # default variant -> edit
        assert tag_edit == '_Qwen-Image-Edit-2511'
        assert lt._run_name(ds).endswith('_Qwen-Image-Edit-2511')
        ds.train_variant = 'image'
        svc.db.session.commit()
        tag_image = lt._dest_base_tag(ds)
        assert tag_image == '_Qwen-Image'
        assert tag_image != tag_edit
        # ... and both are distinct from a zimage official run (empty tag).
        ds.train_type = 'zimage'
        svc.db.session.commit()
        assert lt._dest_base_tag(ds) == ''


def test_default_and_valid_variants_for_qwen_image():
    """'edit' (Qwen-Image-Edit-2511) is the family default everywhere no
    variant is given (repo owner's product call — musubi's Qwen-Image
    tooling here targets Edit-2511); the accepted enum is per-family
    ('image'/'edit' only — a leftover 'turbo'/'base' from another family
    must fall back to base, not leak into the config)."""
    from app.services import lora_training as lt
    assert lt._default_variant_for('qwen_image') == 'edit'
    assert lt._valid_variants_for('qwen_image') == ('image', 'edit')
    # historical families keep their enum untouched
    assert lt._valid_variants_for('krea') == ('turbo', 'base', 'deturbo')
    assert lt._valid_variants_for(None) == ('turbo', 'base', 'deturbo')
    assert lt._default_variant_for('krea') == 'base'
    assert lt._default_variant_for('zimage') == 'turbo'


# --- 5) deploy routing ----------------------------------------------------------

def test_lora_dest_dir_routes_qwen_image(app, tmp_path):
    import os
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'comfyui': {'base_dir': str(tmp_path / 'comfy')}})
        ds = svc.create_dataset(LOCAL_USER, 'QD', 'zchar_qd', train_type='qwen_image')
        dest = lt._lora_dest_dir(ds)
        assert dest.replace('/', os.sep).endswith(
            os.sep.join(('models', 'loras', 'qwen_image')))
        # family override (UI selector) wins over the persisted type
        assert lt._lora_dest_dir(ds, family='krea').endswith('krea')


# --- 6) cloud training stays closed this wave ------------------------------------

def test_cloud_training_refuses_qwen_image(app, tmp_path, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import cloud_training as ct
    from app.config import LOCAL_USER
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QC', 'zchar_qc', train_type='qwen_image')
        with pytest.raises(ValueError, match='local-only'):
            ct.launch_cloud_training(LOCAL_USER, ds.id)


# --- 7) family badge/label classification ---------------------------------------

def test_family_of_lora_classifies_qwen_image_folder():
    from app.utils.comfyui import family_of_lora, FAMILY_LABELS
    assert family_of_lora(r'qwen_image\x.safetensors') == 'qwen_image'
    assert family_of_lora('qwen_image/x.safetensors') == 'qwen_image'
    # the 'flux' prefix match must not misclassify a qwen_image folder.
    assert family_of_lora(r'flux\x.safetensors') == 'flux'
    assert FAMILY_LABELS['qwen_image'] == 'Qwen-Image'


# --- 8) Qwen-Image-Edit control images (both engines) ----------------------------
# Repo owner's guide (2026-07-25): character/style/concept LoRAs use a solid
# BLACK control image (erases spatial guidance so the model places the subject
# from the prompt alone); only a genuine before/after edit-pair dataset would
# use the real source image, and this app has no dataset kind for that yet.

def _add_kept_image(ds, filename='src.png', size=(48, 32), color=(200, 30, 30)):
    from PIL import Image
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    Image.new('RGB', size, color).save(os.path.join(svc._dataset_dir(ds.id), filename))
    svc.db.session.add(FaceDatasetImage(dataset_id=ds.id, status='keep',
                                        filename=filename, caption='a subject, plain background'))
    svc.db.session.commit()


def test_export_generates_black_control_images_for_edit_variant(app, tmp_path):
    from PIL import Image
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QCtrl', 'zchar_qctrl', train_type='qwen_image')
        assert lt._qwen_image_is_edit(ds) is True   # unset variant -> edit (default)
        _add_kept_image(ds, size=(48, 32))
        out = lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=False,
                                             dest_dir=tmp_path / 'export')
        control_dir = lt._qwen_edit_control_dir(str(out))
        assert os.path.isdir(control_dir)
        target_png = next((tmp_path / 'export').glob('*.png'))
        control_png = next(p for p in os.listdir(control_dir) if p.endswith('.png'))
        assert control_png == target_png.name   # same filename, filename-paired lookup
        with Image.open(os.path.join(control_dir, control_png)) as im:
            assert im.size == (48, 32)           # matches the TARGET image, not a fixed size
            assert im.convert('RGB').getpixel((0, 0)) == (0, 0, 0)   # solid black
            assert im.convert('RGB').getpixel((24, 16)) == (0, 0, 0)


def test_export_omits_control_images_for_base_qwen_image_variant(app, tmp_path):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QBase', 'zchar_qbase', train_type='qwen_image')
        ds.train_variant = 'image'   # base T2I, not Edit-2511
        svc.db.session.commit()
        _add_kept_image(ds)
        out = lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=False,
                                             dest_dir=tmp_path / 'export')
        assert not os.path.isdir(lt._qwen_edit_control_dir(str(out)))


def test_export_omits_control_images_for_non_qwen_image_family(app, tmp_path):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'ZChar', 'zchar_zc', train_type='zimage')
        _add_kept_image(ds)
        out = lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=False,
                                             dest_dir=tmp_path / 'export')
        assert not os.path.isdir(lt._qwen_edit_control_dir(str(out)))


def test_export_clears_stale_control_dir_when_switched_to_base_variant(app, tmp_path):
    """A dataset re-exported after switching Edit-2511 -> base T2I must not leave
    a stale control folder for build_job_config to accidentally pick up."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QSwitch', 'zchar_qsw', train_type='qwen_image')
        _add_kept_image(ds)
        dest = tmp_path / 'export'
        out = lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=False, dest_dir=dest)
        control_dir = lt._qwen_edit_control_dir(str(out))
        assert os.path.isdir(control_dir)

        ds.train_variant = 'image'
        svc.db.session.commit()
        lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=False, dest_dir=dest)
        assert not os.path.isdir(control_dir)


def test_build_job_config_qwen_image_control_path_follows_variant(app, tmp_path):
    """ai-toolkit's `control_path` (verified against toolkit/config_modules.py's
    DatasetConfig) tracks the SAME edit/base split as name_or_path: present only
    when the folder was actually populated by export_dataset_to_aitoolkit."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QJob', 'zchar_qjob', train_type='qwen_image')
        _add_kept_image(ds)
        out = lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=False,
                                             dest_dir=tmp_path / 'export')
        proc = lt.build_job_config(ds, str(out), steps=1500,
                                   training_folder='__test__')['config']['process'][0]
        assert proc['datasets'][0]['control_path'] == lt._qwen_edit_control_dir(str(out))

        ds.train_variant = 'image'
        svc.db.session.commit()
        lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=False,
                                       dest_dir=tmp_path / 'export')
        proc = lt.build_job_config(ds, str(out), steps=1500,
                                   training_folder='__test__')['config']['process'][0]
        assert 'control_path' not in proc['datasets'][0]
