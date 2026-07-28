import test from 'node:test';
import assert from 'node:assert/strict';

import {
  BANK_SORTS, bankSortOptions,
  DATASET_SORTS, datasetSortOptions, sortDatasetImages, normalizeDatasetSort,
} from './gridSort.js';
import { filterImagesByStatus } from './gridStatusFilter.js';

// id → the dataset row shape the grid actually receives.
const img = (id, face_score, status = 'keep') => ({
  id, face_score, status, filename: `${id}.png`,
});

// Deliberately unordered scores + two rows the face pass never reached.
const IMAGES = [
  img(5, 0.41), img(4, null), img(3, 0.88), img(2, 0.62), img(1, undefined),
];
const order = (list) => list.map((i) => i.id);

// ---- dataset: the central assertion ---------------------------------------

test('face similarity sorts both ways and sinks the unscored to the END', () => {
  // Closest first: 0.88 > 0.62 > 0.41, then the two unscored — never at the top.
  assert.deepEqual(order(sortDatasetImages(IMAGES, 'face_desc')), [3, 2, 5, 4, 1]);
  // Least alike first: the unscored are STILL last, not first — a "worst first"
  // sort that opened on the un-measured pile would be worse than no sort at all.
  assert.deepEqual(order(sortDatasetImages(IMAGES, 'face_asc')), [5, 2, 3, 4, 1]);
  // Ties (both unscored) fall back to the default newest-first order.
  assert.deepEqual(order(sortDatasetImages([img(1, null), img(9, null)], 'face_desc')),
    [9, 1]);
});

test('default (and an unknown id) is a no-op that keeps the same array', () => {
  assert.equal(sortDatasetImages(IMAGES, 'default'), IMAGES);
  assert.equal(sortDatasetImages(IMAGES, 'bogus-from-an-old-localStorage'), IMAGES);
  assert.equal(normalizeDatasetSort('bogus'), 'default');
  assert.equal(normalizeDatasetSort('face_asc'), 'face_asc');
});

test('sorting never mutates the input', () => {
  const before = order(IMAGES);
  sortDatasetImages(IMAGES, 'face_desc');
  assert.deepEqual(order(IMAGES), before);
});

// ---- dataset: composition with the existing filters ------------------------

test('sort composes with the decision filter — no image lost, none invented', () => {
  const mixed = [
    img(5, 0.41, 'keep'), img(4, 0.95, 'reject'), img(3, 0.88, 'keep'),
    img(2, null, 'keep'), img(1, 0.62, 'reject'),
  ];
  const kept = filterImagesByStatus(mixed, 'kept');
  // Filter → sort and sort → filter agree, and both keep exactly the kept set.
  const filterThenSort = sortDatasetImages(kept, 'face_desc');
  const sortThenFilter = filterImagesByStatus(sortDatasetImages(mixed, 'face_desc'), 'kept');
  assert.deepEqual(order(filterThenSort), [3, 5, 2]);
  assert.deepEqual(order(sortThenFilter), order(filterThenSort));
  assert.deepEqual(new Set(order(filterThenSort)), new Set(order(kept)));
  // The rejected rows stay out even though one of them has the best score.
  assert.ok(!order(filterThenSort).includes(4));
});

test('"select all" follows the active sort — it takes the list the grid shows', () => {
  // DatasetGrid derives BOTH its tiles and its select-all from the same `images`
  // prop, so whatever the workspace sorted+filtered IS the selection universe.
  const shown = sortDatasetImages(filterImagesByStatus(IMAGES, 'kept'), 'face_asc');
  const selectAll = shown.filter((i) => i.filename).map((i) => i.id);
  assert.deepEqual(selectAll, order(shown));
  assert.deepEqual(selectAll, [5, 2, 3, 4, 1]);
});

// ---- menus: an option whose data does not exist is disabled WITH the reason -

test('dataset menu greys out similarity until something is scored', () => {
  const none = datasetSortOptions([img(1, null), img(2, undefined)]);
  assert.deepEqual(none.map((o) => o.disabled), [false, true, true]);
  for (const o of none.slice(1)) {
    assert.match(o.label, /Analyze faces/, 'the reason must be IN the label');
    assert.match(o.title, /No image has a face score yet/);
  }
  // One scored image is enough to make the sort meaningful.
  const some = datasetSortOptions(IMAGES);
  assert.deepEqual(some.map((o) => o.disabled), [false, false, false]);
  assert.deepEqual(some.map((o) => o.id), DATASET_SORTS.map((s) => s.id));
});

test('bank menu greys out what the passes have not produced yet', () => {
  const fresh = bankSortOptions({ scanned: 0, scored: 0 });
  const byId = Object.fromEntries(fresh.map((o) => [o.id, o]));
  assert.equal(byId.default.disabled, false);
  for (const id of ['res_desc', 'res_asc', 'sharp_desc', 'sharp_asc']) {
    assert.equal(byId[id].disabled, true, `${id} should be disabled`);
    assert.match(byId[id].label, /Scan quality/);
  }
  for (const id of ['aesthetic_desc', 'aesthetic_asc']) {
    assert.equal(byId[id].disabled, true, `${id} should be disabled`);
    assert.match(byId[id].label, /Score/);
  }
  // Scanned but not scored: resolution + sharpness usable, aesthetics not.
  const scanned = Object.fromEntries(
    bankSortOptions({ scanned: 120, scored: 0 }).map((o) => [o.id, o.disabled]));
  assert.deepEqual(scanned, {
    default: false, res_desc: false, res_asc: false,
    aesthetic_desc: true, aesthetic_asc: true, sharp_desc: false, sharp_asc: false,
  });
  // Everything measured: nothing greyed out.
  assert.ok(bankSortOptions({ scanned: 120, scored: 120 }).every((o) => !o.disabled));
  // Payload not loaded yet — "I don't know" is never rendered as "you can't".
  assert.ok(bankSortOptions(null).every((o) => !o.disabled));
  assert.ok(bankSortOptions(undefined).every((o) => !o.disabled));
});

test('bank sort ids are the ones the server accepts (never rename: they are stored)', () => {
  assert.deepEqual(BANK_SORTS.map((s) => s.id), [
    'default', 'res_desc', 'res_asc',
    'aesthetic_desc', 'aesthetic_asc', 'sharp_desc', 'sharp_asc',
  ]);
});
