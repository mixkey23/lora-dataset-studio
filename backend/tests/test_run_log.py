"""Run log viewer (repo owner requests):
1. The crash banner only shows a log tail on a non-zero exit code — a run
   that silently produced nothing (or that someone wants to watch WHILE it's
   running) had no way to see the raw ai-toolkit/musubi-tuner output in-app.
   tail_run_log() reads THIS run's training.log with the same scoping
   (train_type/base_model/variant) the checkpoint browser already uses.
2. A run that OOM'd and got retried shared ONE training.log across every
   attempt: ai-toolkit's `open(path, 'w')` truncated the previous attempt on
   the next launch, musubi-tuner's `open(path, 'a')` silently concatenated
   every attempt into one file — either way there was no way to read back
   ONE specific attempt's log once a later one had started.
   _archive_existing_log() (called from launch_training, right before a new
   log_path is opened) fixes this: every launch keeps its own file."""
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
        assert result['exists'] is False
        assert result['lines'] == []
        assert result['logs'] == []


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
        assert result['filename'] == 'training.log'


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
        assert result['exists'] is True
        assert result['lines'] == ['musubi output\n']


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


# --- per-launch log archiving (OOM-retry bug) --------------------------------

def test_archive_existing_log_renames_with_timestamp(app, tmp_path):
    from app.services import lora_training as lt
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    (run_dir / 'training.log').write_text('attempt 1\n', encoding='utf-8')
    lt._archive_existing_log(run_dir)
    assert not (run_dir / 'training.log').is_file()
    archived = [f for f in run_dir.iterdir() if f.name != 'training.log']
    assert len(archived) == 1
    assert archived[0].name.startswith('training_')
    assert archived[0].name.endswith('.log')
    assert archived[0].read_text(encoding='utf-8') == 'attempt 1\n'


def test_archive_existing_log_noop_when_nothing_to_archive(app, tmp_path):
    from app.services import lora_training as lt
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    lt._archive_existing_log(run_dir)   # must not raise
    assert list(run_dir.iterdir()) == []


def test_list_run_logs_lists_every_attempt_newest_first(app, tmp_path):
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    _configure_aitoolkit_dir(tmp_path, app)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'RL5', 'zchar_rl5')
        run_dir = lt._output_dir() / lt._run_name(ds) / 'lora_zchar_rl5'
        run_dir.mkdir(parents=True)
        (run_dir / 'training_20260101-000000.log').write_text('old attempt\n', encoding='utf-8')
        (run_dir / 'training.log').write_text('current attempt\n', encoding='utf-8')
        # An unrelated file in the run dir (a checkpoint) must never be listed.
        (run_dir / 'lora_zchar_rl5.safetensors').write_bytes(b'fake')
        logs = lt.list_run_logs(LOCAL_USER, ds.id)
        assert [l['filename'] for l in logs] == ['training.log', 'training_20260101-000000.log']
        assert logs[0]['current'] is True
        assert logs[1]['current'] is False


def test_tail_run_log_can_select_an_archived_attempt(app, tmp_path):
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    _configure_aitoolkit_dir(tmp_path, app)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'RL6', 'zchar_rl6')
        run_dir = lt._output_dir() / lt._run_name(ds) / 'lora_zchar_rl6'
        run_dir.mkdir(parents=True)
        (run_dir / 'training_20260101-000000.log').write_text('oom attempt\n', encoding='utf-8')
        (run_dir / 'training.log').write_text('retry attempt\n', encoding='utf-8')

        current = lt.tail_run_log(LOCAL_USER, ds.id)
        assert current['lines'] == ['retry attempt\n']

        archived = lt.tail_run_log(LOCAL_USER, ds.id, filename='training_20260101-000000.log')
        assert archived['lines'] == ['oom attempt\n']
        assert archived['filename'] == 'training_20260101-000000.log'


def test_tail_run_log_unknown_filename_falls_back_to_current(app, tmp_path):
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    _configure_aitoolkit_dir(tmp_path, app)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'RL7', 'zchar_rl7')
        run_dir = lt._output_dir() / lt._run_name(ds) / 'lora_zchar_rl7'
        run_dir.mkdir(parents=True)
        (run_dir / 'training.log').write_text('current\n', encoding='utf-8')
        # A path-traversal-shaped filename must never escape the whitelist.
        result = lt.tail_run_log(LOCAL_USER, ds.id, filename='../../etc/passwd')
        assert result['filename'] == 'training.log'
        assert result['lines'] == ['current\n']


def test_train_log_route_accepts_file_param(app, tmp_path):
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    _configure_aitoolkit_dir(tmp_path, app)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'RL8', 'zchar_rl8')
        run_dir = lt._output_dir() / lt._run_name(ds) / 'lora_zchar_rl8'
        run_dir.mkdir(parents=True)
        (run_dir / 'training_20260101-000000.log').write_text('old\n', encoding='utf-8')
        (run_dir / 'training.log').write_text('new\n', encoding='utf-8')
        did = ds.id
    client = app.test_client()
    r = client.get(f'/api/dataset/{did}/train/log?file=training_20260101-000000.log')
    body = r.get_json()
    assert body['lines'] == ['old\n']
    assert len(body['logs']) == 2
