import assert from 'node:assert/strict';
import test from 'node:test';
import {
  CANVAS_SELECTION_KEY,
  readSelection, resolveSelection, selectionSummary, toggleSelection, writeSelection,
} from './canvasSelection.js';

const memoryStore = () => {
  const data = {};
  return {
    data,
    getItem: (k) => (k in data ? data[k] : null),
    setItem: (k, v) => { data[k] = String(v); },
  };
};

test('nothing stored yet shows every dataset', () => {
  const store = memoryStore();
  assert.equal(readSelection(store), null);
  assert.deepEqual(resolveSelection([3, 1, 2], readSelection(store)), [3, 1, 2]);
});

test('an empty stored selection is respected, not reset to all', () => {
  const store = memoryStore();
  writeSelection(store, []);
  assert.deepEqual(readSelection(store), []);
  assert.deepEqual(resolveSelection([1, 2], readSelection(store)), []);
});

test('a stored selection keeps the index order, not the tick order', () => {
  const store = memoryStore();
  writeSelection(store, [2, 7]);
  assert.deepEqual(resolveSelection([7, 5, 2], readSelection(store)), [7, 2]);
});

test('a stored id whose dataset is gone is dropped silently', () => {
  const store = memoryStore();
  writeSelection(store, [1, 999]);
  assert.deepEqual(resolveSelection([1, 2], readSelection(store)), [1]);
});

test('a corrupt stored value reads as "never chosen" instead of blanking the board', () => {
  const store = memoryStore();
  store.setItem(CANVAS_SELECTION_KEY, 'not json');
  assert.equal(readSelection(store), null);
  store.setItem(CANVAS_SELECTION_KEY, '{"a":1}');
  assert.equal(readSelection(store), null);
  assert.deepEqual(resolveSelection([4], readSelection(store)), [4]);
});

test('non-numeric ids are filtered out on the way in and out', () => {
  const store = memoryStore();
  store.setItem(CANVAS_SELECTION_KEY, '[1,"2","oops",null]');
  assert.deepEqual(readSelection(store), [1, 2]);
});

test('a store that refuses to write never breaks the canvas', () => {
  const throwing = { getItem: () => { throw new Error('blocked'); },
    setItem: () => { throw new Error('quota'); } };
  assert.equal(writeSelection(throwing, [1]), false);
  assert.equal(readSelection(throwing), null);
});

test('toggling adds and removes, always in the available order', () => {
  const available = [5, 3, 1];
  assert.deepEqual(toggleSelection([5], 1, available), [5, 1]);
  assert.deepEqual(toggleSelection([5, 1], 5, available), [1]);
  // Ticking in a weird order still yields the index order.
  assert.deepEqual(toggleSelection([1], 3, available), [3, 1]);
});

test('toggling an id that is not available yields nothing new', () => {
  assert.deepEqual(toggleSelection([1], 42, [1, 2]), [1]);
});

test('the collapsed filter button says what is on the board', () => {
  assert.equal(selectionSummary(0, 0), 'No trained datasets');
  assert.equal(selectionSummary(0, 7), 'None of 7');
  assert.equal(selectionSummary(3, 7), '3 of 7');
  assert.equal(selectionSummary(7, 7), 'All 7 datasets');
  assert.equal(selectionSummary(1, 1), '1 dataset');
});
