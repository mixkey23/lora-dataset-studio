/* The preview must draw what the trainer will be given. These are the SAME numbers
   asserted on the Python side (backend/tests/test_concept_face_masking.py
   test_dilate_box_is_the_shared_arithmetic) — if one side drifts, one of the two
   suites goes red instead of the user being shown a lie. */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  dilateBox, clampBox, coverageFraction, boxStyle, MAX_COVERAGE, SHIFT_UP,
} from './faceMaskBox.js';

const r6 = (n) => Math.round(n * 1e6) / 1e6;

test('a face box grows into a head box, biased upward for the hair', () => {
  // 0.2 x 0.2 centred at (0.5, 0.5), expand 2.0 -> exactly the Python assertion
  const [x1, y1, x2, y2] = dilateBox([0.4, 0.4, 0.6, 0.6], 2.0);
  assert.equal(r6(x1), 0.3);
  assert.equal(r6(x2), 0.7);
  // shifted up by 10% of the face height: not symmetric around 0.5, on purpose
  assert.equal(r6(y1), 0.28);
  assert.equal(r6(y2), 0.68);
});

test('expand 1.0 keeps the face box size and only applies the shift', () => {
  const [, y1, , y2] = dilateBox([0, 0, 1, 1], 1.0);
  assert.equal(r6(y1), -0.1);
  assert.equal(r6(y2), 0.9);
  assert.equal(SHIFT_UP, 0.1);
});

test('a bigger expand covers strictly more', () => {
  const small = dilateBox([0.4, 0.4, 0.6, 0.6], 1.5);
  const big = dilateBox([0.4, 0.4, 0.6, 0.6], 2.5);
  assert.ok(big[0] < small[0] && big[2] > small[2], 'wider');
  assert.ok(big[1] < small[1] && big[3] > small[3], 'taller');
});

test('growth past the frame is clamped for DISPLAY only', () => {
  // A face at the edge legitimately grows off-frame; the raw box keeps the truth,
  // the clamped one is what can be drawn inside the image.
  const raw = dilateBox([0.0, 0.0, 0.1, 0.1], 3.0);
  assert.ok(raw[0] < 0, 'raw box keeps the off-frame extent');
  const [cx1, cy1] = clampBox(raw);
  assert.equal(cx1, 0);
  assert.equal(cy1, 0);
});

test('coverage is what decides an image is not worth masking', () => {
  // Small face, small coverage — the normal case, masking is cheap and safe.
  assert.ok(coverageFraction([[0.45, 0.45, 0.55, 0.55]], 2.0) < 0.05);
  // Face filling the frame: over the ceiling, so the export leaves it unmasked
  // rather than silently multiplying the loss on the sliver that remains.
  assert.ok(coverageFraction([[0.1, 0.1, 0.9, 0.9]], 2.0) > MAX_COVERAGE);
  assert.equal(coverageFraction([], 2.0), 0);
  // never reports more than the whole frame
  assert.ok(coverageFraction([[0, 0, 1, 1], [0, 0, 1, 1]], 3.0) <= 1);
});

test('several faces are all counted — masking only the biggest is the bug', () => {
  const two = coverageFraction([[0.1, 0.4, 0.2, 0.5], [0.7, 0.4, 0.8, 0.5]], 2.0);
  const one = coverageFraction([[0.1, 0.4, 0.2, 0.5]], 2.0);
  assert.ok(two > one);
});

test('boxStyle yields percentages an absolutely-positioned overlay can use', () => {
  const s = boxStyle([0.4, 0.4, 0.6, 0.6], 2.0);
  assert.equal(s.left, '30%');
  assert.equal(s.width, '40%');
  for (const v of Object.values(s)) assert.match(v, /%$/);
});
