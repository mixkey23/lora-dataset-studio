"""Subject type `anime` — a DRAWN character, not a photographed person.

Why a sixth type at all: LDS ships a dedicated anime TRAINING family (Anima), but
until now an anime character dataset had to declare itself `human`, which handed it
an identity lock written for photography ("skin tone and texture", "realistic
photographic portrait") and a shot catalog of lens/lighting conventions. Every layer
of the chain then pulled toward photorealism.

The trap this file exists to lock is the STYLE one. For every other subject type the
rendering is a constant ("Professional realistic photograph") that nobody has to
think about; for `anime` the style IS part of the subject, so three separate places
had to stop hardcoding photography:
  1. the identity lock (its own guard, naming hair/eyes/outfit/art style),
  2. `wrap_variation_klein`'s HARDCODED tail — "Create a new photograph …" +
     "Professional realistic photograph, SFW." — which is NOT part of the editable
     guard and would have silently fought the anime lock on every Klein generation,
  3. the shot catalog prompts ("close-up photo of …").
Tests below assert all three, plus the non-regression the other five types demand.
"""
import re

import pytest

from app.services import face_variations as fv


# --- registration -------------------------------------------------------------

def test_anime_is_registered_last():
    """APPENDED, never inserted: `subject_type` is a STORED column, so the order of
    the tuple is the order the UI renders and the values already in databases must
    keep their meaning. Adding at the end is the only safe move."""
    assert fv.SUBJECT_TYPES == ('human', 'animal', 'creature', 'object', 'other', 'anime')
    assert fv.normalize_subject_type('anime') == 'anime'
    assert fv.normalize_subject_type('  ANIME ') == 'anime'
    assert 'anime' in fv._SUBJECT_CATALOGS
    assert 'anime' in fv._IDENTITY_DEFAULTS_BY_SUBJECT


# --- the identity lock: character design, NOT a photographed face --------------

def test_anime_has_its_own_identity_lock_not_the_human_one():
    for kind in ('face_single', 'face_multi', 'klein_identity'):
        anime = fv.identity_prompt_default(kind, 'anime')
        assert anime != fv.identity_prompt_default(kind, 'human'), kind
        assert anime.strip()


def test_anime_lock_names_the_traits_that_define_a_drawn_character():
    """A drawn character is not defined by skin texture. It is defined by hair
    (colour + shape), eyes, the signature outfit, the accessories and the marks —
    and by the ART STYLE itself, which every other type treats as a constant."""
    for kind in ('face_single', 'face_multi', 'klein_identity'):
        low = fv.identity_prompt_default(kind, 'anime').lower()
        for trait in ('hair', 'eye', 'outfit', 'accessor', 'art style', 'line'):
            assert trait in low, (kind, trait)


def test_anime_lock_forbids_converting_the_drawing_into_a_photo():
    """THE trap. Every other lock ENDS by demanding a realistic photograph; for an
    anime subject that instruction destroys the subject."""
    for kind in ('face_single', 'face_multi', 'klein_identity'):
        text = fv.identity_prompt_default(kind, 'anime')
        low = text.lower()
        assert not re.search(r'realistic photograph|photographic portrait|natural skin texture',
                             low), kind
        # …and it says so OUT LOUD rather than merely omitting it: the edit engines
        # default to photo, so silence is not a style instruction.
        assert re.search(r'not turn it into a photograph', low), kind
        assert re.search(r'anime illustration|anime art style', low), kind


# --- the Klein wrapper: the hardcoded photographic tail ------------------------

def test_klein_wrapper_stops_hardcoding_photography_for_anime():
    """`wrap_variation_klein` builds "Create a new photograph of the same {noun} …"
    and appends "Professional realistic photograph, SFW." OUTSIDE the editable
    guard — so even a perfect anime lock would have been overruled twice per
    prompt."""
    out = fv.wrap_variation_klein('standing in a school corridor', framing='body',
                                  subject_type='anime')
    assert 'Create a new illustration of the same character' in out
    assert not out.startswith('Create a new photograph')
    # Photography may only ever appear as a PROHIBITION here — never as the opening
    # command and never as the closing style tag, the two positions an instruction
    # model weighs most. Those two were hardcoded and are what this feature fixes.
    assert 'realistic photograph' not in out.lower()
    assert 'do not turn it into a photograph' in out.lower()
    assert 'no photographic skin texture' in out.lower()
    assert out.rstrip().endswith('Anime illustration, same art style as the reference, SFW.')
    assert fv.identity_prompt_default('klein_identity', 'anime') in out
    # the framing detail is anime-flavoured too, not a 85mm-lens hint
    assert 'lens' not in out.lower()


def test_klein_nsfw_tail_for_anime_stays_drawn():
    out = fv.wrap_variation_klein('a shot', framing='bust', subject_type='anime', nsfw=True)
    assert 'realistic photograph' not in out.lower()
    assert 'anime' in out.lower()


def test_api_wrapper_uses_the_anime_lock():
    single = fv.wrap_variation('a full body shot', subject_type='anime')
    assert single.startswith(fv.IDENTITY_GUARD_ANIME)
    multi = fv.wrap_variation('a full body shot', ref_count=3, subject_type='anime')
    assert multi.startswith(fv.IDENTITY_GUARD_ANIME_MULTI)


# --- the catalog: drawn framing conventions, and the outfit is IDENTITY --------

def test_anime_catalog_is_its_own_and_deep_enough():
    cat = fv.variation_catalog('anime')
    assert cat is not fv.VARIATION_CATALOG
    assert len(cat) >= 40, len(cat)
    assert {e['framing'] for e in cat} == {'face', 'bust', 'body', 'back'}
    ids = [e['id'] for e in cat]
    assert len(ids) == len(set(ids))


def test_anime_catalog_speaks_illustration_not_photography():
    """A catalog that says "close-up photo of" hands the engine the photographic
    intent the lock just forbade."""
    for e in fv.variation_catalog('anime'):
        assert not re.search(r'\bphoto\b|\bphotograph\b|\bproduct photo\b', e['prompt'], re.I), e['id']


def test_anime_catalog_treats_the_signature_outfit_as_identity():
    """The single biggest divergence from `human`. The human catalog BAKES
    "wearing a different casual everyday outfit" into every shot that names no
    outfit, because for a real person clothing must not bind to the identity. For
    an anime character the signature outfit IS the character — varying it by
    default would train the LoRA to forget it. The augmentation must therefore
    never run here, and the catalog must say "signature outfit" out loud."""
    cat = fv.variation_catalog('anime')
    for e in cat:
        assert fv.OUTFIT_VARY not in e['prompt'], e['id']
        assert fv.EXPRESSION_NEUTRAL not in e['prompt'], e['id']
    named = [e for e in cat if 'signature outfit' in e['prompt']]
    assert len(named) >= 20, len(named)
    # …while still offering a DELIBERATE alternate-outfit shot the user opts into.
    assert any('alternate' in e['prompt'].lower() for e in cat)


def test_anime_labels_join_the_global_union_without_collision():
    labels = [e['label'] for e in fv._ALL_CATALOGS]
    assert len(labels) == len(set(labels))
    for e in fv.variation_catalog('anime'):
        assert e['label'] in labels
        assert fv.prompt_by_label(e['label']) == e['prompt'], e['id']
        assert fv.aspect_for_label(e['label'], e['framing']) == fv.aspect_for_entry(e), e['id']
        assert fv.is_nsfw_label(e['label']) is False, e['id']
        # a catalog label that is ALSO a legacy alias key would be rewritten by
        # canonical_label before every lookup and resolve to somebody else
        assert e['label'] not in fv.LEGACY_LABEL_ALIASES, e['id']
    assert set(fv.all_catalog_labels()) >= {e['label'] for e in fv.variation_catalog('anime')}


def test_anime_presets_resolve_and_stay_affordable():
    presets = fv.presets_for('anime')
    assert presets
    for name, ids in presets.items():
        assert 8 <= len(ids) <= 26, (name, len(ids))
        assert len(ids) == len(set(ids)), name
        assert len(fv.select_preset(name, 'anime')) == len(ids), name
    meta = fv.preset_meta_for('anime')
    assert meta and all(m['key'] in presets for m in meta)
    assert {e['framing'] for e in fv.select_preset('anime_balanced', 'anime')} == \
        {'face', 'bust', 'body', 'back'}


def test_anime_has_no_nsfw_catalog_like_every_non_human_type():
    assert fv.nsfw_variation_catalog('anime') == []


# --- NON-REGRESSION: nobody else moved ---------------------------------------

def test_human_is_strictly_unchanged_by_the_new_type():
    """Explicit non-regression: the point of appending a type is that no existing
    dataset wakes up with a different lock, catalog or preset."""
    assert fv.variation_catalog('human') is fv.VARIATION_CATALOG
    assert fv.variation_catalog(None) is fv.VARIATION_CATALOG
    assert fv.variation_catalog('anime') is not fv.VARIATION_CATALOG
    assert fv.identity_prompt_default('face_single', 'human') == fv.IDENTITY_GUARD
    assert fv.identity_prompt_default('face_multi', 'human') == fv.IDENTITY_GUARD_MULTI
    assert fv.identity_prompt_default('klein_identity', 'human') == fv.IDENTITY_GUARD_KLEIN
    assert fv.wrap_variation('p') == fv.wrap_variation('p', subject_type='human')
    assert fv.wrap_variation_klein('p', framing='bust') == \
        fv.wrap_variation_klein('p', framing='bust', subject_type='human')
    assert len(fv.select_preset('balanced_25')) == 25
    assert fv.preset_meta_for('human') == []
    assert fv.nsfw_variation_catalog('human') is fv.NSFW_VARIATION_CATALOG


@pytest.mark.parametrize('st', ['animal', 'creature', 'object', 'other'])
def test_other_types_keep_their_photographic_tail(st):
    """The Klein render tail was made per-subject; the five pre-existing types must
    come out byte-identical, tail included."""
    out = fv.wrap_variation_klein('a shot', framing='body', subject_type=st)
    assert out.startswith('Create a new photograph of the same ')
    assert out.endswith('Professional realistic photograph, SFW.')
    nsfw = fv.wrap_variation_klein('a shot', framing='body', subject_type=st, nsfw=True)
    assert nsfw.endswith('Explicit nudity is allowed; render natural, anatomically '
                         'correct forms. Professional realistic photograph.')


def test_anime_identity_override_is_scoped_to_anime(monkeypatch):
    """Same storage contract as the other non-human types: its own branch, never
    the flat legacy key (which is the HUMAN override)."""
    assert (fv.identity_prompt_config_key('face_single', 'anime')
            == 'identity_prompts.by_subject.anime.face_single')
    assert fv.identity_prompt_config_key('klein_improve', 'anime') == 'identity_prompts.klein_improve'


# --- persistence: the stored column round-trips (app_context, always) ---------

def test_anime_subject_type_persists(app):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Waifu', 'zwaifu', subject_type='anime')
        assert ds.subject_type == 'anime'
        assert svc.subject_type_of(ds) == 'anime'
        # 'anime' fits the VARCHAR(16) column and survives a round trip
        assert svc.get_dataset(LOCAL_USER, ds.id).subject_type == 'anime'


def test_variations_route_serves_the_anime_payload(client):
    d = client.get('/api/dataset/variations?subject_type=anime').get_json()
    assert d['subject_type'] == 'anime'
    assert d['nsfw_catalog'] == []
    assert d['preset_meta'] and d['catalog']
    assert any('Anime' in e['label'] for e in d['catalog'])
    # human payload shape untouched
    assert set(client.get('/api/dataset/variations').get_json()) == \
        {'catalog', 'nsfw_catalog', 'presets'}
