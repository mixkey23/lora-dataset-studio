import test from 'node:test';
import assert from 'node:assert/strict';
import {
  captionWordDiff, datasetChangeChips, datasetIsIdentical, sortedEnvRows, sideLabel,
} from './runCompare.js';

// ---- captionWordDiff: the payoff of recording caption TEXT -----------------

test('captionWordDiff marks exactly the words that changed', () => {
  const segs = captionWordDiff('a woman in a red coat', 'a woman in a blue coat');
  assert.deepEqual(segs.filter((s) => s.type === 'removed').map((s) => s.text), ['red']);
  assert.deepEqual(segs.filter((s) => s.type === 'added').map((s) => s.text), ['blue']);
  // the untouched words survive as context, in order
  assert.equal(segs.filter((s) => s.type === 'same').map((s) => s.text).join(' | '),
    'a woman in a | coat');
});

test('captionWordDiff: identical captions are all context, no edits', () => {
  const segs = captionWordDiff('same text here', 'same text here');
  assert.deepEqual(segs, [{ type: 'same', text: 'same text here' }]);
});

test('captionWordDiff handles an empty side as a pure add or a pure remove', () => {
  assert.deepEqual(captionWordDiff('', 'brand new caption'),
    [{ type: 'added', text: 'brand new caption' }]);
  assert.deepEqual(captionWordDiff('old caption', null),
    [{ type: 'removed', text: 'old caption' }]);
  assert.deepEqual(captionWordDiff('', ''), []);
});

test('captionWordDiff merges runs of the same type instead of one span per word', () => {
  const segs = captionWordDiff('one two three four', 'one nine ten four');
  // "two three" leaves together, "nine ten" arrives together
  assert.deepEqual(segs.map((s) => [s.type, s.text]), [
    ['same', 'one'], ['removed', 'two three'], ['added', 'nine ten'], ['same', 'four'],
  ]);
});

test('captionWordDiff ignores punctuation-only churn on an otherwise shared word', () => {
  // Appending a clause gives the previous last word a comma. Without normalisation
  // that word reads as removed AND added, burying the real edit in noise.
  const segs = captionWordDiff('wind in her hair', 'wind in her hair, shot on 85mm');
  assert.equal(segs.filter((s) => s.type === 'removed').length, 0);
  assert.deepEqual(segs.filter((s) => s.type === 'added').map((s) => s.text),
    ['shot on 85mm']);
  // the shared words render with their LATER spelling (the comma is kept)
  assert.equal(segs[0].text, 'wind in her hair,');
});

test('captionWordDiff still reports a real word change next to punctuation', () => {
  const segs = captionWordDiff('wearing a red coat, smiling', 'wearing a blue coat, smiling');
  assert.deepEqual(segs.filter((s) => s.type === 'removed').map((s) => s.text), ['red']);
  assert.deepEqual(segs.filter((s) => s.type === 'added').map((s) => s.text), ['blue']);
});

test('captionWordDiff: this test bites — a diff that ignored `after` would fail', () => {
  const segs = captionWordDiff('alpha beta', 'gamma delta');
  assert.ok(segs.some((s) => s.type === 'added' && s.text.includes('gamma')));
  assert.ok(segs.some((s) => s.type === 'removed' && s.text.includes('alpha')));
  assert.equal(segs.filter((s) => s.type === 'same').length, 0);
});

// ---- the headline chips ----------------------------------------------------

test('datasetChangeChips counts withheld entries, not just the shown ones', () => {
  const chips = datasetChangeChips({
    added: [{ id: 1 }], added_withheld: 4,
    removed: [], removed_withheld: 0,
    caption_changed: [{ id: 2 }, { id: 3 }], caption_withheld: 0,
    content_changed: [], content_withheld: 0,
  });
  const byKey = Object.fromEntries(chips.map((c) => [c.key, c]));
  assert.equal(byKey.added.count, 5);
  assert.equal(byKey.added.label, '5 images added');
  assert.equal(byKey.caption_changed.label, '2 captions edited');
  assert.ok(!byKey.removed);          // zero -> no chip at all
  assert.ok(!byKey.content_changed);
});

test('datasetChangeChips singularises a single change', () => {
  const chips = datasetChangeChips({ removed: [{ id: 9 }] });
  assert.equal(chips[0].label, '1 image removed');
});

test('datasetIsIdentical only claims sameness when nothing is unknown', () => {
  const empty = { added: [], removed: [], caption_changed: [], content_changed: [] };
  assert.equal(datasetIsIdentical({ images: empty, notes: [] }), true);
  // a run that predates snapshots leaves a note -> we must NOT claim sameness
  assert.equal(datasetIsIdentical({ images: empty, notes: ['predates full snapshots'] }), false);
  assert.equal(datasetIsIdentical({ images: null, notes: [] }), false);
  assert.equal(datasetIsIdentical(null), false);
});

// ---- environment ordering ---------------------------------------------------

test('sortedEnvRows floats the differences to the top', () => {
  const rows = [
    { key: 'gpu', changed: false }, { key: 'torch', changed: true },
    { key: 'cuda', changed: false }, { key: 'aitoolkit.commit', changed: true },
  ];
  assert.deepEqual(sortedEnvRows(rows).map((r) => r.key),
    ['torch', 'aitoolkit.commit', 'gpu', 'cuda']);
  assert.deepEqual(sortedEnvRows(undefined), []);
});

test('sortedEnvRows does not mutate its input', () => {
  const rows = [{ key: 'a', changed: false }, { key: 'b', changed: true }];
  sortedEnvRows(rows);
  assert.deepEqual(rows.map((r) => r.key), ['a', 'b']);
});

// ---- side label -------------------------------------------------------------

test('sideLabel names the version, the dataset and the record', () => {
  assert.equal(sideLabel({ version: 3, dataset_name: 'marion', record_id: 117 }),
    'v3 · marion · #117');
  assert.equal(sideLabel({ record_id: 5 }), '#5');
  assert.equal(sideLabel(null), '');
});
