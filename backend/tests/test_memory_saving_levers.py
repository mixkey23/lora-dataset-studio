"""Memory-saving levers — quantize / quantize_te / low_vram exposed as a per-dataset
override (GitHub issue #14, bobba84: "Disable Quant / LowVRAM for Z-Image Training —
I have a 5090").

The whole point of this feature is a TRI-STATE that leaves the calibrated defaults
alone: the recipes were tuned so a 12B DiT fits in 24 GB, and the majority of the
install base is at 24 GB or less. A changed default would break installs that
worked. So the first test here is the one that matters most — an untouched dataset
must produce a byte-identical job config on every family.
"""
import pytest

from app.config import LOCAL_USER

FAMILIES = ('zimage', 'krea', 'flux', 'flux2klein', 'anima', 'sdxl')
QUANTISED_BY_DEFAULT = ('zimage', 'krea', 'flux', 'flux2klein')
PLAIN_BY_DEFAULT = ('anima', 'sdxl')


def _mk(svc, tt, tmp_path):
    from app.extensions import db
    ds = svc.create_dataset(LOCAL_USER, tt.upper(), f'mem_{tt}', train_type=tt)
    if tt == 'sdxl':
        ds.train_base_model = str(tmp_path / 'base.safetensors')
        db.session.commit()
    return ds


def _model(lt, ds, folder):
    return lt.build_job_config(ds, str(folder), 1500)['config']['process'][0]['model']


def _ready(app, tmp_path):
    from app import config as cfg
    cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
    folder = tmp_path / 'ds'
    folder.mkdir(exist_ok=True)
    return folder


def test_untouched_dataset_keeps_every_family_default(app, tmp_path):
    """THE non-negotiable: a user who never opens Advanced options gets exactly
    what shipped before this feature — quantisation + low-VRAM on the four big
    families, off on the two small ones, and `low_vram`/`qtype` still absent from
    anima/sdxl rather than emitted as an explicit False."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    with app.app_context():
        folder = _ready(app, tmp_path)
        for tt in QUANTISED_BY_DEFAULT:
            m = _model(lt, _mk(svc, tt, tmp_path), folder)
            assert m['quantize'] is True, tt
            assert m['quantize_te'] is True, tt
            assert m['low_vram'] is True, tt
            assert m['qtype'] == 'qfloat8', tt
        for tt in PLAIN_BY_DEFAULT:
            m = _model(lt, _mk(svc, tt, tmp_path), folder)
            assert m['quantize'] is False, tt
            assert m['quantize_te'] is False, tt
            assert 'low_vram' not in m, tt      # ai-toolkit default is False → omitted
            assert 'qtype' not in m, tt         # meaningless without quantisation


def test_disable_all_three_on_every_family(app, tmp_path):
    """The request itself: turning the three off reaches the job config on EVERY
    family (the levers are ModelConfig fields, arch-agnostic — no whitelist), and
    `qtype` disappears with them because it only means something when quantising."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    with app.app_context():
        folder = _ready(app, tmp_path)
        for tt in FAMILIES:
            ds = _mk(svc, tt, tmp_path)
            lt.update_train_settings(LOCAL_USER, ds.id, {
                'quantize': False, 'quantize_te': False, 'low_vram': False})
            m = _model(lt, svc.get_dataset(LOCAL_USER, ds.id), folder)
            assert m['quantize'] is False, tt
            assert m['quantize_te'] is False, tt
            assert 'low_vram' not in m, tt
            assert 'qtype' not in m, tt


def test_lever_also_works_in_the_other_direction(app, tmp_path):
    """The same lever serves the opposite user: a small card can turn quantisation
    and streaming ON for a family whose default is off (anima/sdxl)."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    with app.app_context():
        folder = _ready(app, tmp_path)
        for tt in PLAIN_BY_DEFAULT:
            ds = _mk(svc, tt, tmp_path)
            lt.update_train_settings(LOCAL_USER, ds.id, {
                'quantize': True, 'quantize_te': True, 'low_vram': True})
            m = _model(lt, svc.get_dataset(LOCAL_USER, ds.id), folder)
            assert m['quantize'] is True and m['quantize_te'] is True, tt
            assert m['low_vram'] is True, tt
            assert m['qtype'] == 'qfloat8', tt


def test_partial_override_leaves_the_others_at_the_family_default(app, tmp_path):
    """Each key is independent: turning only `low_vram` off (the speed lever on a
    32 GB card that still wants qfloat8) must not silently drop quantisation."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    with app.app_context():
        folder = _ready(app, tmp_path)
        ds = _mk(svc, 'zimage', tmp_path)
        lt.update_train_settings(LOCAL_USER, ds.id, {'low_vram': False})
        m = _model(lt, svc.get_dataset(LOCAL_USER, ds.id), folder)
        assert 'low_vram' not in m
        assert m['quantize'] is True and m['quantize_te'] is True
        assert m['qtype'] == 'qfloat8'


def test_stored_false_survives_and_auto_clears(app, tmp_path):
    """A stored `False` is a VALUE, not an absence — the trap this feature had to
    avoid (a falsy-drop like `dual_captions` would have made "disable" impossible).
    'auto' puts the key back to the family default by REMOVING it."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = _mk(svc, 'zimage', tmp_path)
        lt.update_train_settings(LOCAL_USER, ds.id, {'quantize': False})
        assert lt.snapshot_train_settings(LOCAL_USER, ds.id) == {'quantize': False}
        lt.update_train_settings(LOCAL_USER, ds.id, {'quantize': 'auto'})
        assert 'quantize' not in lt.snapshot_train_settings(LOCAL_USER, ds.id)


def test_validation_rejects_non_boolean(app, tmp_path):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = _mk(svc, 'krea', tmp_path)
        with pytest.raises(ValueError) as e:
            lt.update_train_settings(LOCAL_USER, ds.id, {'low_vram': 'yes please'})
        assert 'low_vram' in str(e.value)


def test_effective_settings_expose_stored_default_and_effective(app, tmp_path):
    """What the Advanced panel reads: the stored choice (None → the select shows
    "Auto"), the family default, and what will actually be sent."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = _mk(svc, 'zimage', tmp_path)
        eff = lt.effective_train_settings(ds)
        assert eff['memory_saving'] == {'quantize': None, 'quantize_te': None,
                                        'low_vram': None}
        assert eff['memory_saving_default'] == {'quantize': True, 'quantize_te': True,
                                                'low_vram': True}
        assert eff['memory_saving_effective'] == eff['memory_saving_default']
        lt.update_train_settings(LOCAL_USER, ds.id, {'quantize': False})
        eff2 = lt.effective_train_settings(svc.get_dataset(LOCAL_USER, ds.id))
        assert eff2['memory_saving']['quantize'] is False
        assert eff2['memory_saving_default']['quantize'] is True      # default intact
        assert eff2['memory_saving_effective']['quantize'] is False


def test_launch_snapshot_and_share_stamp_the_memory_strategy(app, tmp_path):
    """Two runs of the same dataset can differ ONLY by these three; the run
    snapshot therefore stamps their effective value (including the default), and
    the ⎘ Share config knows them as first-class rows."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.services.run_share import _KNOWN_SETTING_KEYS
    with app.app_context():
        ds = _mk(svc, 'krea', tmp_path)
        snap = lt.launch_settings_snapshot(ds, 'krea')
        assert (snap['quantize'], snap['quantize_te'], snap['low_vram']) == (True, True, True)
        lt.update_train_settings(LOCAL_USER, ds.id, {'quantize': False, 'low_vram': False})
        snap2 = lt.launch_settings_snapshot(svc.get_dataset(LOCAL_USER, ds.id), 'krea')
        assert snap2['quantize'] is False and snap2['low_vram'] is False
        assert snap2['quantize_te'] is True                  # untouched → default
        for k in ('quantize', 'quantize_te', 'low_vram'):
            assert k in _KNOWN_SETTING_KEYS, k


def test_advice_is_indexed_on_the_real_card_and_never_decides(app, tmp_path, monkeypatch):
    """Hardware-indexed guidance: a 32 GB card is told it can turn them off, a
    12 GB card is told to leave them on. Advising is not deciding — neither
    verdict writes anything into train_settings."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.services import run_environment as env
    with app.app_context():
        ds = _mk(svc, 'zimage', tmp_path)          # needs ~18 GB unquantised
        monkeypatch.setattr(env, 'local_vram_gb', lambda: 31.4)
        monkeypatch.setattr(env, 'gpu_info', lambda: {'name': 'NVIDIA GeForce RTX 5090'})
        a = lt.effective_train_settings(ds)['memory_advice']
        assert a['verdict'] == 'can_disable'
        assert a['vram_gb'] == 31.4 and '5090' in a['gpu']
        monkeypatch.setattr(env, 'local_vram_gb', lambda: 12.0)
        assert lt.effective_train_settings(ds)['memory_advice']['verdict'] == 'keep_on'
        assert lt.snapshot_train_settings(LOCAL_USER, ds.id) == {}   # nothing written


def test_advice_degrades_to_unknown_when_the_probe_fails(app, tmp_path, monkeypatch):
    """No nvidia-smi, an AMD card, a GPU-less machine driving a cloud run: the
    panel falls back to generic text and NOTHING is blocked. Same rule as every
    other probe in run_environment — fail-open."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.services import run_environment as env
    with app.app_context():
        folder = _ready(app, tmp_path)
        ds = _mk(svc, 'zimage', tmp_path)

        def _boom():
            raise OSError('nvidia-smi not found')
        monkeypatch.setattr(env, 'local_vram_gb', _boom)
        a = lt.effective_train_settings(ds)['memory_advice']
        assert a['verdict'] == 'unknown' and a['vram_gb'] is None
        # and a launch still builds normally
        assert _model(lt, ds, folder)['quantize'] is True


def test_klein_4b_needs_less_than_9b(app, tmp_path):
    """The estimate must not lie by lumping the two Klein bases together: the 4B
    fits unquantised on a card where the 9B does not."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.extensions import db
    with app.app_context():
        ds9 = _mk(svc, 'flux2klein', tmp_path)
        ds9.train_variant = '9b'
        db.session.commit()
        ds4 = _mk(svc, 'flux2klein', tmp_path)
        ds4.train_variant = '4b'
        db.session.commit()
        need9 = lt.effective_train_settings(ds9)['memory_advice']['unquantised_vram_gb']
        need4 = lt.effective_train_settings(ds4)['memory_advice']['unquantised_vram_gb']
        assert need4 < need9


def test_cloud_rebuild_carries_the_memory_choice(app, tmp_path):
    """The cloud pod rebuilds the job at boot from a view of the live dataset, so
    a rented 48 GB GPU gets the un-quantised recipe the user asked for."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.services.cloud_training import _run_config_dataset
    with app.app_context():
        folder = _ready(app, tmp_path)
        ds = _mk(svc, 'krea', tmp_path)
        lt.update_train_settings(LOCAL_USER, ds.id, {'quantize': False, 'low_vram': False})
        view = _run_config_dataset(svc.get_dataset(LOCAL_USER, ds.id),
                                   {'train_type': 'krea', 'variant': 'base'})
        m = _model(lt, view, folder)
        assert m['quantize'] is False and 'low_vram' not in m


def test_preset_apply_accepts_the_keys(app, client):
    """Presets are validated through the same path, so a shared preset can carry a
    memory strategy (and an old preset without the keys leaves the defaults)."""
    ds_id = client.post('/api/dataset/create',
                        json={'name': 'M', 'trigger_word': 'mt',
                              'train_type': 'zimage'}).get_json()['id']
    r = client.post(f'/api/dataset/{ds_id}/train/presets/apply',
                    json={'settings': {'quantize': False, 'low_vram': False}})
    body = r.get_json()
    assert body['ok'] is True and body['rejected'] == []
    with app.app_context():
        from app.services import lora_training as lt
        assert lt.snapshot_train_settings('local', ds_id) == {'quantize': False,
                                                              'low_vram': False}


def test_resume_dialog_still_refuses_them_as_overrides(app):
    """Documented decision: the levers are safe on a resume (the base is frozen;
    the checkpoint holds only LoRA weights) but the ▶ Continue dialog has no
    control for them, so they stay OUT of the whitelist rather than becoming
    untested surface. The persisted value is what a resume re-reads."""
    from app.services import lora_training as lt
    with app.app_context():
        assert 'quantize' not in lt.RESUME_SAFE_SETTING_KEYS
        with pytest.raises(ValueError):
            lt.validate_resume_overrides({'quantize': False})
