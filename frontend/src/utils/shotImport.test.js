import test from 'node:test';
import assert from 'node:assert/strict';

import {
  SHOT_IMPORT_FORMAT, FRAMINGS, MAX_IMPORT_BYTES, MAX_IMPORT_SHOTS,
  MAX_LABEL_LEN, MAX_PROMPT_LEN,
  parseShotImport, applyShotImport, buildShotExport, promoteCustomShot,
} from './shotImport.js';

/** A valid file with `n` distinct shots, so each test only writes the ONE thing
 *  it is about. */
const file = (shots, extra = {}) => JSON.stringify({
  format: SHOT_IMPORT_FORMAT, subject_type: 'animal', shots, ...extra,
});
const shot = (i) => ({ label: `Dog shot ${i}`, framing: 'body',
  prompt: `full body photo of the animal, take ${i}, outdoor` });
const parse = (text, opts = {}) => parseShotImport(text, { subjectType: 'animal', ...opts });

// --- Blocking refusals: nothing in the file is importable ---------------------

test('a file over the size cap is refused before parsing', () => {
  const res = parse(file([shot(1)]), { byteLength: MAX_IMPORT_BYTES + 1 });
  assert.equal(res.blocked.code, 'too_large');
  assert.match(res.blocked.message, /512 KB/);
  assert.deepEqual(res.accepted, []);
});

test('invalid JSON is refused with a readable message, not an exception', () => {
  const res = parse('{ "shots": [ {label: nope} ] }');
  assert.equal(res.blocked.code, 'not_json');
  assert.match(res.blocked.message, /valid JSON/i);
});

test('a JSON array (not an object) at the root is refused', () => {
  const res = parse('[{"label":"x"}]');
  assert.equal(res.blocked.code, 'not_object');
});

test('a missing or non-array `shots` key is refused', () => {
  assert.equal(parse('{"format":"lds-shots/1"}').blocked.code, 'no_shots');
  assert.equal(parse('{"shots":"a, b"}').blocked.code, 'no_shots');
});

test('an empty shots array is refused', () => {
  assert.equal(parse(file([])).blocked.code, 'empty');
});

test('more shots than the cap is refused, and the message says how many', () => {
  const many = Array.from({ length: MAX_IMPORT_SHOTS + 1 }, (_, i) => shot(i));
  const res = parse(file(many));
  assert.equal(res.blocked.code, 'too_many');
  assert.match(res.blocked.message, new RegExp(`${MAX_IMPORT_SHOTS + 1}`));
});

test('a file authored for another subject type is refused by name', () => {
  const res = parse(file([shot(1)], { subject_type: 'object' }), { subjectType: 'animal' });
  assert.equal(res.blocked.code, 'subject_mismatch');
  assert.match(res.blocked.message, /object/);
  assert.match(res.blocked.message, /animal/);
});

test('a file with no subject_type at all is accepted (an LLM often drops it)', () => {
  const res = parse('{"shots":[{"label":"Dog A","framing":"body","prompt":"a dog"}]}');
  assert.equal(res.blocked, null);
  assert.equal(res.accepted.length, 1);
});

// --- Per-entry refusals -------------------------------------------------------

const rejectOne = (entry, opts = {}) => {
  const res = parse(file([entry]), opts);
  assert.equal(res.blocked, null, 'should be an entry-level refusal, not a blocking one');
  assert.equal(res.accepted.length, 0);
  assert.equal(res.rejected.length, 1);
  return res.rejected[0];
};

test('an entry that is not an object is rejected by index', () => {
  const bad = rejectOne('just a string');
  assert.equal(bad.code, 'not_object');
  assert.equal(bad.index, 1);
});

test('missing / blank / non-string label is rejected', () => {
  assert.equal(rejectOne({ framing: 'body', prompt: 'a dog' }).code, 'missing_label');
  assert.equal(rejectOne({ label: '   ', framing: 'body', prompt: 'a dog' }).code, 'missing_label');
  assert.equal(rejectOne({ label: 42, framing: 'body', prompt: 'a dog' }).code, 'missing_label');
});

test('missing / blank / non-string prompt is rejected, and names the entry', () => {
  const bad = rejectOne({ label: 'Dog A', framing: 'body' });
  assert.equal(bad.code, 'missing_prompt');
  assert.equal(bad.label, 'Dog A');
  assert.match(bad.message, /Dog A/);
});

test('a missing framing is rejected and lists the four allowed values', () => {
  const bad = rejectOne({ label: 'Dog A', prompt: 'a dog' });
  assert.equal(bad.code, 'missing_framing');
  for (const f of FRAMINGS) assert.match(bad.message, new RegExp(f));
});

test('an unknown framing is NEVER remapped: it is rejected, naming the bad value', () => {
  const bad = rejectOne({ label: 'Dog A', framing: 'closeup', prompt: 'a dog' });
  assert.equal(bad.code, 'bad_framing');
  assert.match(bad.message, /closeup/);
  for (const f of FRAMINGS) assert.match(bad.message, new RegExp(f));
});

test('framing matching is case-insensitive but normalises to the stored enum', () => {
  const res = parse(file([{ label: 'Dog A', framing: 'BODY', prompt: 'a dog' }]));
  assert.equal(res.accepted[0].framing, 'body');
});

test('an over-long label or prompt is rejected with the cap in the message', () => {
  const longLabel = rejectOne({ label: 'x'.repeat(MAX_LABEL_LEN + 1), framing: 'body', prompt: 'a dog' });
  assert.equal(longLabel.code, 'label_too_long');
  assert.match(longLabel.message, new RegExp(`${MAX_LABEL_LEN}`));
  const longPrompt = rejectOne({ label: 'Dog A', framing: 'body', prompt: 'x'.repeat(MAX_PROMPT_LEN + 1) });
  assert.equal(longPrompt.code, 'prompt_too_long');
  assert.match(longPrompt.message, new RegExp(`${MAX_PROMPT_LEN}`));
});

test('an nsfw flag that is not a boolean is rejected rather than coerced', () => {
  const bad = rejectOne({ label: 'Dog A', framing: 'body', prompt: 'a dog', nsfw: 'yes' });
  assert.equal(bad.code, 'bad_nsfw');
});

// --- Label collisions: the load-bearing invariant ------------------------------

test('two entries in the same file sharing a label: the SECOND is rejected', () => {
  const res = parse(file([
    { label: 'Dog A', framing: 'body', prompt: 'first' },
    { label: 'Dog A', framing: 'face', prompt: 'second' },
  ]));
  assert.equal(res.accepted.length, 1);
  assert.equal(res.accepted[0].prompt, 'first');
  assert.equal(res.rejected[0].code, 'duplicate_label_in_file');
  assert.equal(res.rejected[0].index, 2);
  assert.match(res.rejected[0].message, /Dog A/);
});

test('a duplicate that differs only by case is caught too', () => {
  const res = parse(file([
    { label: 'Dog A', framing: 'body', prompt: 'first' },
    { label: 'dog a', framing: 'body', prompt: 'second' },
  ]));
  assert.equal(res.rejected[0].code, 'duplicate_label_in_file');
});

test('a label colliding with a built-in of ANOTHER subject type is rejected', () => {
  // 'Bust, front' is a HUMAN catalog label; the user is importing for 'animal'.
  // prompt_by_label / aspect_for_label / is_nsfw_label search the union of every
  // catalog with the label alone, so this would resolve to the wrong entry.
  const res = parse(file([{ label: 'Bust, front', framing: 'bust', prompt: 'a dog bust' }]), {
    reservedLabels: ['Bust, front', 'Animal head, front'],
  });
  assert.equal(res.accepted.length, 0);
  assert.equal(res.rejected[0].code, 'label_collides_builtin');
  assert.match(res.rejected[0].message, /Bust, front/);
  assert.match(res.rejected[0].message, /rename/i);
});

test('a label colliding with a LEGACY French alias is rejected', () => {
  // Old rows still store 'Buste face'; the alias map resolves it to a built-in.
  // An imported shot usurping it would hijack that resolution in silence.
  const res = parse(file([{ label: 'Buste face', framing: 'bust', prompt: 'a dog bust' }]), {
    reservedLabels: ['Bust, front', 'Buste face'],
  });
  assert.equal(res.rejected[0].code, 'label_collides_builtin');
  assert.match(res.rejected[0].message, /Buste face/);
});

test('a label colliding with an already-imported / custom shot is rejected', () => {
  const res = parse(file([{ label: 'Dog zoomies', framing: 'body', prompt: 'again' }]), {
    existingLabels: ['Dog zoomies'],
  });
  assert.equal(res.rejected[0].code, 'label_collides_existing');
  assert.match(res.rejected[0].message, /already/i);
});

// --- Ignored, but never in silence --------------------------------------------

test('top-level `examples` are skipped by contract and counted, not imported', () => {
  const res = parse(file([shot(1)], {
    examples: [{ label: 'Animal head, front', framing: 'face', prompt: 'built-in sample' }],
  }));
  assert.equal(res.accepted.length, 1);
  assert.equal(res.skippedExamples, 1);
  assert.equal(res.rejected.length, 0);
});

test('unknown per-shot fields are reported, not silently dropped, and `aspect` is called out', () => {
  const res = parse(file([{ label: 'Dog A', framing: 'body', prompt: 'a dog',
    aspect: '16:9', axis: 'framing', vibe: 'cool' }]));
  assert.equal(res.accepted.length, 1);
  assert.deepEqual(res.ignoredFields.sort(), ['aspect', 'axis', 'vibe']);
  assert.equal(res.accepted[0].aspect, undefined);
});

test('an id in the file is advisory: it never lands as-is', () => {
  const res = parse(file([{ id: 'bust_front', label: 'Dog A', framing: 'body', prompt: 'a dog' }]));
  assert.equal(res.accepted[0].id, undefined);
  assert.ok(!res.ignoredFields.includes('id'), 'id is a known field, not an unknown one');
});

// --- Nothing is written before confirmation ------------------------------------

test('parse is PURE: it mutates neither the input nor the existing shots', () => {
  const existing = [{ id: 'imp_1', label: 'Kept', framing: 'body', prompt: 'kept' }];
  const snapshot = JSON.stringify(existing);
  const text = file([shot(1), { label: 'Kept', framing: 'body', prompt: 'dupe' }]);
  const before = text;
  parse(text, { existingLabels: existing.map((s) => s.label) });
  assert.equal(text, before);
  assert.equal(JSON.stringify(existing), snapshot);
});

test('a file with one bad entry still yields the good ones — but only via applyShotImport', () => {
  const res = parse(file([
    shot(1), shot(2),
    { label: 'Dog bad', framing: 'nope', prompt: 'a dog' },
  ]));
  assert.equal(res.accepted.length, 2);
  assert.equal(res.rejected.length, 1);
  const merged = applyShotImport([], res.accepted);
  assert.equal(merged.length, 2);          // all-or-nothing on the ACCEPTED set
});

// --- applyShotImport: ids are minted here, never trusted from the file ---------

test('imported shots get fresh unique ids that cannot collide with existing ones', () => {
  const existing = [{ id: 'imp_dog_a_1', label: 'Old', framing: 'body', prompt: 'old' }];
  const res = parse(file([{ label: 'Dog A', framing: 'body', prompt: 'a dog' },
    { label: 'Dog B', framing: 'face', prompt: 'a dog head' }]));
  const merged = applyShotImport(existing, res.accepted);
  assert.equal(merged.length, 3);
  const ids = merged.map((s) => s.id);
  assert.equal(new Set(ids).size, 3);
  for (const id of ids) assert.match(id, /^imp_/);
  assert.deepEqual(merged[0], existing[0]);          // existing untouched, order kept
});

test('applyShotImport keeps only the stored shape (no stray keys reach localStorage/config)', () => {
  const res = parse(file([{ label: 'Dog A', framing: 'body', prompt: 'a dog', nsfw: true, vibe: 'x' }]));
  const [added] = applyShotImport([], res.accepted);
  assert.deepEqual(Object.keys(added).sort(), ['framing', 'id', 'imported', 'label', 'nsfw', 'prompt']);
  assert.equal(added.imported, true);
  assert.equal(added.nsfw, true);
});

// --- Export / round-trip -------------------------------------------------------

test('the export carries the format, the subject, the user shots and built-in examples', () => {
  const payload = JSON.parse(buildShotExport({
    subjectType: 'animal',
    shots: [{ id: 'imp_1', label: 'Dog A', framing: 'body', prompt: 'a dog', imported: true }],
    catalog: Array.from({ length: 20 }, (_, i) => ({
      id: `animal_${i}`, label: `Animal ${i}`, framing: 'face', prompt: `built-in ${i}` })),
  }));
  assert.equal(payload.format, SHOT_IMPORT_FORMAT);
  assert.equal(payload.subject_type, 'animal');
  assert.equal(payload.shots.length, 1);
  assert.deepEqual(Object.keys(payload.shots[0]).sort(), ['framing', 'label', 'prompt']);
  assert.ok(payload.examples.length > 0 && payload.examples.length <= 6);
  assert.ok(payload.instructions, 'the file tells the LLM what to produce');
});

test('an export re-imports cleanly: its own shots come back, its examples do not', () => {
  const catalog = [{ id: 'animal_head_front', label: 'Animal head, front', framing: 'face',
    prompt: 'close-up photo of the animal' }];
  const exported = buildShotExport({
    subjectType: 'animal',
    shots: [{ id: 'imp_1', label: 'Dog A', framing: 'body', prompt: 'a dog', imported: true }],
    catalog,
  });
  // Re-imported on a FRESH install (the user's own shots are gone, the built-ins
  // are reserved) — the round-trip is the backup path, so it must not collide.
  const res = parse(exported, { reservedLabels: catalog.map((e) => e.label) });
  assert.equal(res.blocked, null);
  assert.equal(res.accepted.length, 1);
  assert.equal(res.accepted[0].label, 'Dog A');
  assert.equal(res.rejected.length, 0);
  assert.equal(res.skippedExamples, 1);
});

test('a 🔞 shot survives the round-trip — nsfw is a first-class field', () => {
  // The importer accepts nsfw, applyShotImport keeps it and the backend stores
  // it. Dropping it on export turned an explicit shot into a safe one on the way
  // back — the SAME label, quietly rerouted to another engine on regenerate.
  const exported = buildShotExport({
    subjectType: 'human',
    shots: [
      { id: 'imp_1', label: 'Spicy A', framing: 'body', prompt: 'a', nsfw: true, imported: true },
      { id: 'imp_2', label: 'Safe B', framing: 'face', prompt: 'b', imported: true },
    ],
    catalog: [],
  });
  const payload = JSON.parse(exported);
  assert.equal(payload.shots[0].nsfw, true);
  // a safe shot stays exactly as it was — no `nsfw: false` noise in the file
  assert.deepEqual(Object.keys(payload.shots[1]).sort(), ['framing', 'label', 'prompt']);

  const res = parseShotImport(exported, { subjectType: 'human' });
  assert.equal(res.blocked, null);
  assert.equal(res.accepted.length, 2);
  assert.equal(res.accepted[0].nsfw, true);
  assert.equal(res.accepted[1].nsfw, undefined);
  assert.equal(applyShotImport([], res.accepted)[0].nsfw, true);
});

test('every refusal names the entry AND what is wrong with it — that is the whole point', () => {
  const res = parse(file([
    shot(1),
    { label: 'Dog bad', framing: 'nope', prompt: 'a dog' },
  ]));
  assert.equal(res.accepted.length, 1);
  const [bad] = res.rejected;
  assert.match(bad.message, /Dog bad/);       // WHICH entry
  assert.match(bad.message, /nope/);          // WHAT is wrong
  assert.match(bad.message, /face, bust, body, back/);   // and what to do instead
});

// --- Keep: promote a hand-written ✨ card into the durable catalog -------------
// The ✨ cards live in localStorage and die with the browser cache. Exporting +
// re-importing them cannot rescue them (they collide with themselves), so the
// only real fix is a direct promotion — and it has to MOVE the card, not copy it.

const custom = (over = {}) => ({ id: 'custom_1700000000000', label: '✨ on a vintage motorbike',
  prompt: 'full body shot, sitting on a vintage motorbike in a garage, warm light',
  framing: 'body', ...over });

test('promoting moves the card: gone from the custom list, present in the imported one', () => {
  const shotToKeep = custom();
  const other = custom({ id: 'custom_2', label: '✨ another' });
  const res = promoteCustomShot({ shot: shotToKeep, customShots: [shotToKeep, other], importedShots: [] });
  assert.equal(res.ok, true);
  // MOVED, not copied — two copies would mean two shots with the same label,
  // which is the very collision the importer refuses.
  assert.deepEqual(res.customShots.map((s) => s.id), ['custom_2']);
  assert.deepEqual(res.importedShots.map((s) => s.label), ['✨ on a vintage motorbike']);
  assert.equal(res.importedShots[0].imported, true);
});

test('promoting KEEPS the card id, so a saved preset that selected it still resolves', () => {
  // datasetCustomPresetsV1.selectedIds stores shot IDS. Minting a fresh `imp_…`
  // here would silently orphan every preset that had this card selected.
  const shotToKeep = custom();
  const res = promoteCustomShot({ shot: shotToKeep, customShots: [shotToKeep], importedShots: [] });
  assert.equal(res.importedShots[0].id, 'custom_1700000000000');
});

test('a 🔞 card keeps its nsfw flag through the promotion', () => {
  const hot = custom({ id: 'custom_hot', label: '🔞 something', nsfw: true });
  const res = promoteCustomShot({ shot: hot, customShots: [hot], importedShots: [] });
  assert.equal(res.importedShots[0].nsfw, true);
});

test('promoting a card whose label shadows a built-in is refused, in the importer wording', () => {
  const clash = custom({ label: 'Bust, front' });
  const res = promoteCustomShot({ shot: clash, customShots: [clash], importedShots: [],
    reservedLabels: ['Bust, front'] });
  assert.equal(res.ok, false);
  assert.equal(res.code, 'label_collides_builtin');
  assert.match(res.message, /Bust, front/);
  assert.match(res.message, /rename/i);
});

test('promoting a card that clashes with an already-imported shot is refused', () => {
  const clash = custom({ label: 'Dog zoomies' });
  const res = promoteCustomShot({ shot: clash, customShots: [clash],
    importedShots: [{ id: 'imp_dog', label: 'Dog zoomies', prompt: 'x', framing: 'body' }] });
  assert.equal(res.ok, false);
  assert.equal(res.code, 'label_collides_existing');
  assert.match(res.message, /already/i);
});

test('a refused promotion changes NOTHING — the card stays where it was', () => {
  const clash = custom({ label: 'Bust, front' });
  const before = [clash];
  const res = promoteCustomShot({ shot: clash, customShots: before, importedShots: [],
    reservedLabels: ['Bust, front'] });
  assert.equal(res.ok, false);
  assert.equal(res.customShots, undefined);
  assert.deepEqual(before, [clash]);
});

test('an id already taken among imported shots is uniquified rather than overwriting it', () => {
  const shotToKeep = custom({ id: 'custom_dup' });
  const res = promoteCustomShot({ shot: shotToKeep, customShots: [shotToKeep],
    importedShots: [{ id: 'custom_dup', label: 'Older', prompt: 'x', framing: 'body' }] });
  assert.equal(res.ok, true);
  assert.equal(res.importedShots.length, 2);
  assert.notEqual(res.importedShots[1].id, 'custom_dup');
  assert.equal(new Set(res.importedShots.map((s) => s.id)).size, 2);
});

test('a malformed card is refused instead of poisoning the durable catalog', () => {
  const broken = { id: 'custom_x', label: '', prompt: 'x', framing: 'body' };
  assert.equal(promoteCustomShot({ shot: broken, customShots: [broken] }).ok, false);
  const noFraming = { id: 'custom_y', label: 'Fine', prompt: 'x', framing: 'closeup' };
  const res = promoteCustomShot({ shot: noFraming, customShots: [noFraming] });
  assert.equal(res.ok, false);
  assert.match(res.message, /face, bust, body, back/);
});
