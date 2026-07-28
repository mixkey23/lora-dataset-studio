import test from 'node:test';
import assert from 'node:assert/strict';
import { runDeletionLosses, runDeletionKeeps, runDeletionMessage } from './runDeletionSummary.js';

test('the confirmation names every count, with singular/plural', () => {
  const msg = runDeletionMessage(12, {
    notes: 12, previews: 8, canvas_positions: 1, images_unlinked: 34,
    children_detached: 1, archived_images_released: 6,
  });
  assert.match(msg, /Remove run #12/);
  assert.match(msg, /12 checkpoint notes/);
  assert.match(msg, /8 preview links/);
  assert.match(msg, /its saved position on the canvas/);
  assert.match(msg, /6 archived source images/);
  assert.match(msg, /34 generated images stay in the Test Studio/);
  assert.match(msg, /1 run that continued from it stays in the graph, as its own root/);
});

test('the survivor sentences agree in number (1 image STAYS, 2 images STAY)', () => {
  assert.deepEqual(runDeletionKeeps({ images_unlinked: 1, children_detached: 1 }), [
    '1 generated image stays in the Test Studio — it only loses the link to this run.',
    '1 run that continued from it stays in the graph, as its own root.',
  ]);
  assert.deepEqual(runDeletionKeeps({ images_unlinked: 2, children_detached: 3 }), [
    '2 generated images stay in the Test Studio — they only lose the link to this run.',
    '3 runs that continued from it stay in the graph, as their own roots.',
  ]);
});

test('zero counts are omitted, never printed as "0 notes"', () => {
  const lines = runDeletionLosses({ notes: 0, previews: 3, canvas_positions: 0,
                                    archived_images_released: 0 });
  assert.deepEqual(lines, ['3 preview links']);
  assert.deepEqual(runDeletionKeeps({ images_unlinked: 0, children_detached: 0 }), []);
});

test('a bare run says nothing else is attached', () => {
  const msg = runDeletionMessage(3, { notes: 0, previews: 0, canvas_positions: 0,
                                      images_unlinked: 0, children_detached: 0,
                                      archived_images_released: 0 });
  assert.match(msg, /Nothing else is attached to it/);
  assert.match(msg, /No LoRA file is deleted/);
});

test('a failed impact probe falls back to generic wording, never invented numbers', () => {
  const msg = runDeletionMessage(4, null);
  assert.match(msg, /Generated images are kept/);
  assert.doesNotMatch(msg, /\d+ checkpoint note/);
  assert.doesNotMatch(msg, /NaN|undefined/);
});

test('garbage counts degrade to nothing rather than printing NaN', () => {
  assert.deepEqual(runDeletionLosses({ notes: 'x', previews: null, canvas_positions: -2 }), []);
  assert.deepEqual(runDeletionLosses(undefined), []);
});
