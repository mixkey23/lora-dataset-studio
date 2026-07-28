"""A deployed checkpoint belongs to ONE run, not to every run sharing its step.

Reported: step 2500 of run #108 was imported, and step 2500 of runs #116 and
#110 turned "✓ Deployed" too. `_testable_by_step` keyed its map on the step
alone for the whole (dataset, family), so several runs that all saved at 2500
collapsed onto one entry — and the wrong run's Undeploy would have removed a
file it does not own.

The run tag was already in the deployed name (`_rc<id>` / `_rl<id>`) and already
parsed; it was only being used for the STEP-LESS final save. These tests pin it
for numbered steps too, and pin the legacy exemption that keeps older imports
visible.
"""
import pytest


@pytest.fixture
def _ds(app):
    from app.extensions import db
    from app.models import FaceDataset
    with app.app_context():
        d = FaceDataset(user_id='local', name='Collide', trigger_word='collide')
        db.session.add(d)
        db.session.commit()
        yield d.id


def _map(monkeypatch, deployed_names, run_tag):
    """_testable_by_step with the ComfyUI pool stubbed to `deployed_names`."""
    from app.services import cloud_training as ct, lora_test_studio as studio
    monkeypatch.setattr(studio, 'list_test_checkpoints',
                        lambda ds, family=None: [{'filename': n} for n in deployed_names])
    return ct, run_tag


def test_a_step_deployed_by_another_run_is_not_claimed(app, _ds, monkeypatch):
    from app.services import cloud_training as ct
    _map(monkeypatch, ['krea\lora_collide_000002500_Krea-2-Raw_rc108_v1.safetensors'], None)
    with app.app_context():
        # Asked on behalf of run 116 — the only deployed file belongs to 108.
        out = ct._testable_by_step(_ds, 'krea', run_tag=('cloud', 116))
        assert 2500 not in out, (
            'run 116 claims a checkpoint deployed from run 108 — importing one '
            "run's step would light up every other run's same step")


def test_a_run_still_sees_its_own_deployed_step(app, _ds, monkeypatch):
    from app.services import cloud_training as ct
    _map(monkeypatch, ['krea\lora_collide_000002500_Krea-2-Raw_rc108_v1.safetensors'], None)
    with app.app_context():
        out = ct._testable_by_step(_ds, 'krea', run_tag=('cloud', 108))
        assert out.get(2500, '').endswith('rc108_v1.safetensors')


def test_an_untagged_legacy_file_still_matches(app, _ds, monkeypatch):
    """Imports made before run tagging carry no tag. Refusing them would show
    every one of those as not-deployed."""
    from app.services import cloud_training as ct
    _map(monkeypatch, ['krea\lora_collide_000002500_Krea-2-Raw.safetensors'], None)
    with app.app_context():
        out = ct._testable_by_step(_ds, 'krea', run_tag=('cloud', 116))
        assert 2500 in out


def test_without_a_run_context_the_map_is_unchanged(app, _ds, monkeypatch):
    """The plain dataset+family map (no run asked for) keeps its old meaning."""
    from app.services import cloud_training as ct
    _map(monkeypatch, ['krea\lora_collide_000002500_Krea-2-Raw_rc108_v1.safetensors'], None)
    with app.app_context():
        assert 2500 in ct._testable_by_step(_ds, 'krea')
