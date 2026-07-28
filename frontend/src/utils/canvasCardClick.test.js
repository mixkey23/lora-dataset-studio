/* ◉ A press on a run card — the trap of this whole feature.
 *
 * The board's cards are DRAGGABLE ("Drag a run to move it"), and a drop lands on
 * the card it moved, so the browser fires a click straight after every
 * rearrangement. Opening the run gallery on that click would mean every tidy-up
 * of the board ends with a bottom sheet the user never asked for — on a phone,
 * covering the board they were arranging. It is invisible in a screenshot and
 * nothing throws, so it is pinned here.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import { cardClickAction, runGalleryTarget } from './canvasCardClick.js';

test('DRAGGING a card opens nothing', () => {
  assert.equal(cardClickAction({ dragged: true }), 'ignored');
  // …not even with shift held: the gesture was a move, whatever else was down.
  assert.equal(cardClickAction({ dragged: true, shiftKey: true }), 'ignored');
});

test('a plain click opens the run', () => {
  assert.equal(cardClickAction({ dragged: false }), 'open');
  assert.equal(cardClickAction({}), 'open');
  assert.equal(cardClickAction(), 'open');
});

test('shift-click still selects for compare', () => {
  assert.equal(cardClickAction({ shiftKey: true }), 'compare');
});

test('the run target is explicit about being a whole run', () => {
  const node = { record_id: 87, note: 'best so far' };
  assert.deepEqual(runGalleryTarget(node), { kind: 'run', recordId: 87, node });
  // A node with no id opens nothing rather than a gallery of run "undefined".
  assert.equal(runGalleryTarget(null), null);
  assert.equal(runGalleryTarget({}), null);
});
