"""Non-human shot catalogs (depth + curated presets) and user shot catalogs
imported from JSON — idea by ashish.sinha (Discord).

The load-bearing invariant is the one test_subject_types.test_labels_globally_unique
locks: prompt_by_label / aspect_for_label / is_nsfw_label resolve a STORED label
against the union of every catalog, with no subject_type threading. Everything
below exists so a user-imported shot can never join that union under a label
somebody else already answers to.
"""
from app.services import face_variations as fv


# --- catalog depth: a non-human dataset must be buildable, not a token draft --

def test_non_human_catalogs_are_deep_enough_and_cover_every_framing():
    """The first drafts shipped 10-16 shots against 53 for a human — not enough to
    build a varied dataset. Each type now carries a real spread, and every framing
    of the stored enum is represented so composition targets stay reachable."""
    minimums = {'animal': 45, 'creature': 32, 'object': 26, 'other': 20, 'anime': 40}
    for st, low in minimums.items():
        catalog = fv.variation_catalog(st)
        assert len(catalog) >= low, (st, len(catalog))
        framings = {e['framing'] for e in catalog}
        assert framings == {'face', 'bust', 'body', 'back'}, (st, framings)


def test_non_human_catalogs_never_get_the_human_outfit_expression_directives():
    """_augment_prompt is a HUMAN concern (an animal wears no outfit, an object has
    no expression) and is deliberately not run on these catalogs — including on the
    entries added later."""
    for st in ('animal', 'creature', 'object', 'other', 'anime'):
        for e in fv.variation_catalog(st):
            assert fv.OUTFIT_VARY not in e['prompt'], e['id']
            assert fv.EXPRESSION_NEUTRAL not in e['prompt'], e['id']


def test_no_catalog_label_shadows_a_legacy_alias():
    """A catalog label that is also a LEGACY key would be rewritten by
    canonical_label before every lookup — the label would resolve to someone else."""
    for e in fv._ALL_CATALOGS:
        assert e['label'] not in fv.LEGACY_LABEL_ALIASES, e['id']


def test_curated_presets_stay_affordable_and_balanced():
    """A preset used to be 'every id of the type', which on today's catalogs would
    queue (and, on an API engine, bill) up to 59 images in one click. Each preset is
    a deliberate composition now: bounded, and the balanced ones cover all four
    framings."""
    for st in ('animal', 'creature', 'object', 'other', 'anime'):
        for name, ids in fv.presets_for(st).items():
            assert 8 <= len(ids) <= 26, (st, name, len(ids))
            assert len(ids) == len(set(ids)), (st, name)      # no shot queued twice
            assert len(ids) < len(fv.variation_catalog(st)), (st, name)
        balanced = fv.select_preset(st + '_balanced', st)
        assert {e['framing'] for e in balanced} == {'face', 'bust', 'body', 'back'}, st


# --- the reserved label set ---------------------------------------------------

def test_all_catalog_labels_includes_every_catalog_and_the_legacy_aliases():
    labels = set(fv.all_catalog_labels())
    for st in ('human', 'animal', 'creature', 'object', 'other', 'anime'):
        for e in fv.variation_catalog(st):
            assert e['label'] in labels, e['id']
    for e in fv.NSFW_VARIATION_CATALOG:
        assert e['label'] in labels, e['id']
    # The set an import may not re-use MUST include the old French keys: they are
    # still stored on pre-migration rows and routed through canonical_label.
    assert 'Buste face' in labels
    assert set(fv.LEGACY_LABEL_ALIASES) <= labels


# --- sanitizer: the second line of defence behind the importer ----------------

def test_sanitize_custom_shots_keeps_only_well_formed_entries():
    kept = fv.sanitize_custom_shots({'animal': [
        {'id': 'imp_a', 'label': 'Dog zoomies', 'prompt': 'a dog running', 'framing': 'body'},
        {'id': 'imp_b', 'label': 'Dog head', 'prompt': 'a dog head', 'framing': 'FACE'},
        {'id': 'imp_c', 'label': 'Bad framing', 'prompt': 'x', 'framing': 'closeup'},
        {'id': 'imp_d', 'label': 'No prompt', 'framing': 'body'},
        {'id': 'imp_e', 'prompt': 'no label', 'framing': 'body'},
        {'label': 'No id', 'prompt': 'x', 'framing': 'body'},
        'not an object',
        {'id': 'imp_f', 'label': 'x' * 200, 'prompt': 'label too long', 'framing': 'body'},
        {'id': 'imp_g', 'label': 'Long prompt', 'prompt': 'y' * 600, 'framing': 'body'},
    ]})['animal']
    assert [s['id'] for s in kept] == ['imp_a', 'imp_b']
    assert kept[1]['framing'] == 'face'          # normalised to the stored enum
    assert all(s['imported'] is True for s in kept)


def test_sanitize_custom_shots_refuses_a_label_that_shadows_a_builtin():
    """config.json is hand-editable, so the importer's checks are re-run here: a shot
    labelled like a built-in (of ANY subject type, or a legacy alias) would hijack
    prompt/aspect/NSFW resolution for every row storing that label."""
    kept = fv.sanitize_custom_shots({'animal': [
        {'id': 'imp_a', 'label': 'Bust, front', 'prompt': 'stolen human label', 'framing': 'bust'},
        {'id': 'imp_b', 'label': 'Animal head, front', 'prompt': 'stolen animal label', 'framing': 'face'},
        {'id': 'imp_c', 'label': 'Buste face', 'prompt': 'stolen LEGACY label', 'framing': 'bust'},
        {'id': 'imp_d', 'label': 'Dog zoomies', 'prompt': 'fine', 'framing': 'body'},
    ]})
    assert [s['label'] for s in kept['animal']] == ['Dog zoomies']


def test_sanitize_custom_shots_dedupes_and_survives_garbage():
    kept = fv.sanitize_custom_shots({'animal': [
        {'id': 'imp_a', 'label': 'Dog A', 'prompt': 'one', 'framing': 'body'},
        {'id': 'imp_b', 'label': 'dog a', 'prompt': 'same label, other case', 'framing': 'body'},
        {'id': 'imp_a', 'label': 'Dog B', 'prompt': 'same id', 'framing': 'body'},
    ]})['animal']
    assert [s['prompt'] for s in kept] == ['one']
    assert fv.sanitize_custom_shots(None) == {}
    assert fv.sanitize_custom_shots({'animal': 'nope'}) == {}
    # An unknown subject type normalises to human rather than vanishing silently.
    assert fv.sanitize_custom_shots({'zzz': [
        {'id': 'i', 'label': 'L', 'prompt': 'p', 'framing': 'body'}]}) == {
            'human': [{'id': 'i', 'label': 'L', 'prompt': 'p', 'framing': 'body',
                       'imported': True}]}


def test_sanitize_custom_shots_caps_the_list():
    many = [{'id': 'imp_%d' % i, 'label': 'Shot %d' % i, 'prompt': 'p', 'framing': 'body'}
            for i in range(fv.MAX_CUSTOM_SHOTS_PER_SUBJECT + 50)]
    kept = fv.sanitize_custom_shots({'animal': many})['animal']
    assert len(kept) == fv.MAX_CUSTOM_SHOTS_PER_SUBJECT


# --- route --------------------------------------------------------------------

def test_a_promoted_custom_card_keeps_its_id():
    """The ⇪ Keep button promotes a hand-written ✨ card without minting a new id:
    ids are what saved presets store (datasetCustomPresetsV1.selectedIds), so the
    backend must accept the card's own `custom_…` id verbatim, not normalise it."""
    kept = fv.sanitize_custom_shots({'human': [
        {'id': 'custom_1700000000000', 'label': '✨ on a vintage motorbike',
         'prompt': 'full body shot, sitting on a vintage motorbike in a garage',
         'framing': 'body'},
    ]})['human']
    assert [s['id'] for s in kept] == ['custom_1700000000000']
    assert kept[0]['imported'] is True


def test_shot_catalog_route_round_trip(client, tmp_path, monkeypatch):
    from app import config as cfg
    monkeypatch.setattr(cfg, '_config_path', lambda: tmp_path / 'config.json')
    cfg._cache = None
    try:
        empty = client.get('/api/dataset/shot-catalog?subject_type=animal').get_json()
        assert empty['shots'] == []
        assert 'Animal head, front' in empty['reserved_labels']
        assert 'Buste face' in empty['reserved_labels']   # legacy alias is reserved too

        saved = client.put('/api/dataset/shot-catalog', json={'subject_type': 'animal', 'shots': [
            {'id': 'imp_a', 'label': 'Dog zoomies', 'prompt': 'a dog running', 'framing': 'body'},
            {'id': 'imp_b', 'label': 'Bust, front', 'prompt': 'shadows a built-in', 'framing': 'bust'},
        ]}).get_json()
        assert [s['label'] for s in saved['shots']] == ['Dog zoomies']
        assert saved['dropped'] == 1        # reported, never silently "saved"

        again = client.get('/api/dataset/shot-catalog?subject_type=animal').get_json()
        assert [s['label'] for s in again['shots']] == ['Dog zoomies']
        # Writing one subject type leaves the others alone.
        assert client.get('/api/dataset/shot-catalog?subject_type=object').get_json()['shots'] == []
        assert client.put('/api/dataset/shot-catalog',
                          json={'subject_type': 'animal', 'shots': 'nope'}).status_code == 400
    finally:
        cfg._cache = None
