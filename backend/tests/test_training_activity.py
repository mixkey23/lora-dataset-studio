"""🏋️ The live "something is training" signal behind the nav indicator.

Every page polls this, so the contract is as much about what it must NOT do as
about what it returns: no capability probe, no disk, no network — one persisted
flag plus one indexed COUNT. These tests pin the answers (local, cloud, both,
neither) and the fact that an unconfigured cloud reports zero instead of
failing.
"""


def _cloud_run(status, dataset_id=1):
    from app.extensions import db
    from app.models import CloudTrainingRun
    r = CloudTrainingRun(dataset_id=dataset_id, status=status)
    db.session.add(r)
    db.session.commit()
    return r


def _set_local_training(flag):
    from app.job_queue import queue_manager
    queue_manager._set_system_state('training_in_progress', flag)


def test_nothing_running_reports_not_running(app):
    from app.services import cloud_training as ct
    with app.app_context():
        assert ct.training_activity() == {'local': False, 'cloud': 0, 'running': False}


def test_a_local_training_lights_the_indicator(app):
    from app.services import cloud_training as ct
    with app.app_context():
        _set_local_training(True)
        out = ct.training_activity()
        assert out['local'] is True
        assert out['cloud'] == 0
        assert out['running'] is True


def test_active_cloud_runs_are_counted(app):
    """Every state of ACTIVE_STATES counts — a pod that is provisioning or
    uploading is already being billed, so the indicator must be lit well before
    the first training step."""
    from app.services import cloud_training as ct
    with app.app_context():
        for status in ('preparing', 'provisioning', 'uploading', 'training',
                       'downloading', 'terminating'):
            _cloud_run(status)
        out = ct.training_activity()
        assert out['cloud'] == 6
        assert out['running'] is True


def test_finished_cloud_runs_do_not_light_the_indicator(app):
    """The whole point: a stopped or errored run must not leave the dot lit."""
    from app.services import cloud_training as ct
    with app.app_context():
        _cloud_run('done')
        _cloud_run('error')
        _cloud_run('stopped')
        assert ct.training_activity() == {'local': False, 'cloud': 0, 'running': False}


def test_local_and_cloud_are_reported_independently(app):
    from app.services import cloud_training as ct
    with app.app_context():
        _set_local_training(True)
        _cloud_run('training')
        out = ct.training_activity()
        assert out['local'] is True and out['cloud'] == 1 and out['running'] is True


def test_endpoint_answers_200_on_an_unconfigured_install(client):
    """Ungated on purpose: a fresh install polls it too, and a 403 there would
    show up as a console error on every page."""
    r = client.get('/api/train/activity')
    assert r.status_code == 200
    assert r.get_json() == {'local': False, 'cloud': 0, 'running': False}
