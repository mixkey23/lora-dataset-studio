"""An engine added by an update must reach people who already saved settings.

`engines.enabled` is a LIST, and a saved list REPLACES the default instead of
merging with it — so everyone who had ever opened Settings once kept their old
three engines and never learned a fourth existed. These tests pin both halves of
the fix: the new engine shows up, and an engine the user deliberately unchecked
never comes back.
"""
import importlib, json


def _fresh(monkeypatch, tmp_path):
    monkeypatch.setenv('LDS_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('LDS_ENV', str(tmp_path / '.env'))
    import app.config as config
    importlib.reload(config)
    return config


def _write(tmp_path, payload):
    (tmp_path / 'config.json').write_text(json.dumps(payload), encoding='utf-8')


PRE_OPENROUTER = ['nanobanana', 'chatgpt', 'klein']


# --- the new engine reaches an existing install -------------------------------

def test_engine_added_by_an_update_reaches_a_saved_config(tmp_path, monkeypatch):
    """The exact reported scenario: a config saved before OpenRouter shipped."""
    _write(tmp_path, {'engines': {'default': 'chatgpt', 'enabled': list(PRE_OPENROUTER)}})
    config = _fresh(monkeypatch, tmp_path)
    assert 'openrouter' in config.get('engines.enabled')


def test_merge_keeps_every_previously_enabled_engine_and_its_order(tmp_path, monkeypatch):
    """Nothing the user had is lost or reshuffled — the new one is appended, so
    the `enabled[0]` fallback used when a row points at a disabled engine keeps
    naming the same engine as before the update."""
    _write(tmp_path, {'engines': {'enabled': list(PRE_OPENROUTER)}})
    config = _fresh(monkeypatch, tmp_path)
    assert config.get('engines.enabled')[:3] == PRE_OPENROUTER


def test_other_saved_engine_settings_survive_the_merge(tmp_path, monkeypatch):
    _write(tmp_path, {'engines': {'default': 'klein', 'enabled': list(PRE_OPENROUTER),
                                  'chatgpt_auth': 'api'},
                      'ollama': {'vision_concurrency': 2}})
    config = _fresh(monkeypatch, tmp_path)
    assert config.get('engines.default') == 'klein'
    assert config.get('engines.chatgpt_auth') == 'api'
    assert config.get('ollama.vision_concurrency') == 2
    # A scalar key added by the same update still falls back to its default.
    assert config.get('engines.openrouter_model') == 'google/gemini-3-pro-image'


def test_merge_does_not_rewrite_the_config_file(tmp_path, monkeypatch):
    """Reading is not a migration: an untouched install keeps its file byte-for-byte,
    so downgrading to the previous version still finds exactly what it wrote."""
    saved = {'engines': {'enabled': list(PRE_OPENROUTER)}}
    _write(tmp_path, saved)
    config = _fresh(monkeypatch, tmp_path)
    assert 'openrouter' in config.get('engines.enabled')
    assert json.loads((tmp_path / 'config.json').read_text(encoding='utf-8')) == saved


# --- but a deliberate opt-out is never undone ---------------------------------

def test_an_engine_unchecked_on_purpose_never_comes_back(tmp_path, monkeypatch):
    """The counter-test. Once a save has recorded which engines the app KNEW at
    the time, dropping one of them is an explicit choice and must stick."""
    config = _fresh(monkeypatch, tmp_path)
    catalog = list(config.DEFAULTS['engines']['enabled'])
    kept = [e for e in catalog if e != 'nanobanana']
    config.save_config({'engines': {'enabled': kept}})
    config = _fresh(monkeypatch, tmp_path)          # cold start, cache dropped
    assert config.get('engines.enabled') == kept
    assert 'nanobanana' not in config.get('engines.enabled')


def test_unchecking_the_freshly_merged_engine_sticks(tmp_path, monkeypatch):
    """The upgrade path end to end: legacy config -> OpenRouter appears -> the
    user doesn't want it and unchecks it -> it stays gone across restarts."""
    _write(tmp_path, {'engines': {'enabled': list(PRE_OPENROUTER)}})
    config = _fresh(monkeypatch, tmp_path)
    assert 'openrouter' in config.get('engines.enabled')
    config.save_config({'engines': {'enabled': list(PRE_OPENROUTER)}})
    config = _fresh(monkeypatch, tmp_path)
    assert config.get('engines.enabled') == PRE_OPENROUTER


def test_a_legacy_optout_survives_the_merge(tmp_path, monkeypatch):
    """A config written before the ledger existed that dropped one of the THEN
    known engines: that opt-out predates the ledger but is still explicit."""
    _write(tmp_path, {'engines': {'enabled': ['chatgpt', 'klein']}})   # nanobanana off
    config = _fresh(monkeypatch, tmp_path)
    enabled = config.get('engines.enabled')
    assert 'nanobanana' not in enabled
    assert 'openrouter' in enabled


# --- degrade cleanly ----------------------------------------------------------

def test_no_config_file_gets_the_full_default_catalog(tmp_path, monkeypatch):
    config = _fresh(monkeypatch, tmp_path)
    assert config.get('engines.enabled') == config.DEFAULTS['engines']['enabled']


def test_config_without_an_engines_section(tmp_path, monkeypatch):
    _write(tmp_path, {'comfyui': {'api_url': 'http://127.0.0.1:8188'}})
    config = _fresh(monkeypatch, tmp_path)
    assert config.get('engines.enabled') == config.DEFAULTS['engines']['enabled']


def test_corrupt_config_falls_back_to_defaults(tmp_path, monkeypatch):
    (tmp_path / 'config.json').write_text('{not json at all', encoding='utf-8')
    config = _fresh(monkeypatch, tmp_path)
    assert config.get('engines.enabled') == config.DEFAULTS['engines']['enabled']


def test_garbage_shapes_do_not_crash_the_load(tmp_path, monkeypatch):
    for engines in ('nonsense', ['a', 'list'], {'enabled': 'chatgpt'},
                    {'enabled': ['chatgpt', None, 7, 'klein']},
                    {'enabled': ['chatgpt'], 'known': 'nope'},
                    {'enabled': [], 'known': []}):
        _write(tmp_path, {'engines': engines})
        config = _fresh(monkeypatch, tmp_path)
        enabled = config.get('engines.enabled')
        assert isinstance(enabled, list)
        assert all(isinstance(e, str) for e in enabled)


def test_empty_enabled_list_is_left_alone(tmp_path, monkeypatch):
    """Downstream (face_dataset_service) reads an empty list as 'no restriction'.
    Filling it in would turn that into a real, restrictive list."""
    _write(tmp_path, {'engines': {'enabled': []}})
    config = _fresh(monkeypatch, tmp_path)
    assert config.get('engines.enabled') == []


def test_an_unknown_engine_left_in_a_config_is_preserved(tmp_path, monkeypatch):
    """Hand-edited or downgraded configs happen; never drop what we don't know."""
    _write(tmp_path, {'engines': {'enabled': ['chatgpt', 'some-future-engine']}})
    config = _fresh(monkeypatch, tmp_path)
    assert 'some-future-engine' in config.get('engines.enabled')


# --- the ledger itself --------------------------------------------------------

def test_saving_an_explicit_choice_records_the_catalog_it_was_made_from(tmp_path, monkeypatch):
    config = _fresh(monkeypatch, tmp_path)
    config.save_config({'engines': {'enabled': ['chatgpt']}})
    on_disk = json.loads((tmp_path / 'config.json').read_text(encoding='utf-8'))
    assert set(on_disk['engines']['known']) >= set(config.DEFAULTS['engines']['enabled'])


def test_a_save_that_does_not_touch_engines_does_not_freeze_the_catalog(tmp_path, monkeypatch):
    """Saving another section must not silently record 'the user has seen
    OpenRouter' — that would bury it forever for a legacy config."""
    _write(tmp_path, {'engines': {'enabled': list(PRE_OPENROUTER)}})
    config = _fresh(monkeypatch, tmp_path)
    config.save_config({'comfyui': {'api_url': 'http://127.0.0.1:9999'}})
    config = _fresh(monkeypatch, tmp_path)
    assert 'openrouter' in config.get('engines.enabled')
