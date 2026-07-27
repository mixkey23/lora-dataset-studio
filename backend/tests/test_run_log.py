"""Live CLI log viewer (repo owner request): the crash banner only shows a log
tail on a non-zero exit code — a run that silently produced nothing (or that
someone wants to watch WHILE it's running) had no way to see the raw
ai-toolkit/musubi-tuner output in-app. tail_run_log() reads THIS run's
training.log with the same scoping (train_type/base_model/variant) the
checkpoint browser and folder-browser already use."""
from app.config import LOCAL_USER


def _configure_aitoolkit_dir(tmp_path, app):
    """Just enough for _output_dir()/_run_dir() to resolve — these tests never
    launch a real training, so none of the venv/run.py/arch-detection scaffolding
    other test files set up is needed."""
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})


def test_tail_run_log_missing_file_is_not_an_error(app, tmp_path):
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    _configure_aitoolkit_dir(tmp_path, app)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'RL1', 'zchar_rl1')
        result = lt.tail_run_log(LOCAL_USER, ds.id)
        assert result == {'exists': False, 'lines': []}


def test_tail_run_log_reads_last_n_lines(app, tmp_path):
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    _configure_aitoolkit_dir(tmp_path, app)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'RL2', 'zchar_rl2')
        run_dir = lt._output_dir() / lt._run_name(ds) / 'lora_zchar_rl2'
        run_dir.mkdir(parents=True)
        (run_dir / 'training.log').write_text(
            ''.join(f'line {i}\n' for i in range(10)), encoding='utf-8')
        result = lt.tail_run_log(LOCAL_USER, ds.id, n=3)
        assert result['exists'] is True
        assert result['lines'] == ['line 7\n', 'line 8\n', 'line 9\n']


def test_tail_run_log_scoped_to_musubi_run_dir(app, tmp_path):
    """musubi's flat run dir (no lora_<trigger>/ subfolder, see _run_dir) must
    be the one tailed for a musubi-engine dataset — the same run-dir bug
    class list_checkpoints was fixed for."""
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    _configure_aitoolkit_dir(tmp_path, app)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'RL3', 'zchar_rl3', train_type='qwen_image')
        ds.train_engine = 'musubi'
        svc.db.session.commit()
        run_dir = lt._output_dir() / lt._run_name(ds)   # flat, no subfolder
        run_dir.mkdir(parents=True)
        (run_dir / 'training.log').write_text('musubi output\n', encoding='utf-8')
        result = lt.tail_run_log(LOCAL_USER, ds.id)
        assert result == {'exists': True, 'lines': ['musubi output\n']}


def test_train_log_route(app, tmp_path):
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    _configure_aitoolkit_dir(tmp_path, app)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'RL4', 'zchar_rl4')
        run_dir = lt._output_dir() / lt._run_name(ds) / 'lora_zchar_rl4'
        run_dir.mkdir(parents=True)
        (run_dir / 'training.log').write_text('hello\n', encoding='utf-8')
        did = ds.id
    client = app.test_client()
    r = client.get(f'/api/dataset/{did}/train/log')
    assert r.status_code == 200
    body = r.get_json()
    assert body['ok'] is True
    assert body['exists'] is True
    assert body['lines'] == ['hello\n']


def test_train_log_route_missing_dataset_is_404(app):
    client = app.test_client()
    r = client.get('/api/dataset/999999/train/log')
    assert r.status_code == 404
