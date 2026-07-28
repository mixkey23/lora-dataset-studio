/* The face-mask preview must never show a wait it cannot explain.

   Reported live: "Looking for face…" and nothing else for the whole pass — no way
   to tell working from crashed. What is asserted here is the wording contract of
   that progress, which is where the honesty actually lives:

   * the model load is NAMED, and shows no percentage — a determinate bar pinned
     at 0% for tens of seconds is the exact lie being removed;
   * the first-run download says it is a download, not a slow detection;
   * a failure renders as its message and stops the running state.
*/
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  previewError, previewPercent, previewProgressValue, previewRunning, previewStatusLabel,
} from '../src/utils/faceMaskProgress.js';

const job = (o) => ({ phase: 'starting', done: 0, total: 0, error: null, finished: false, ...o });

test('every phase before the first image is named, and none of them fakes a percentage', () => {
  for (const phase of ['starting', 'downloading', 'loading']) {
    const j = job({ phase, total: 40 });
    const label = previewStatusLabel(j);
    assert.ok(label.length > 0, `${phase} must say something`);
    assert.ok(!/\d+ of \d+/.test(label), `${phase} must not claim a per-image count`);
    assert.equal(previewProgressValue(j), null, `${phase} must stay indeterminate`);
  }
});

test('the first-run download says it is a download, with its size', () => {
  const label = previewStatusLabel(job({ phase: 'downloading' }));
  assert.match(label, /download/i);
  assert.match(label, /MB/);
  assert.match(label, /first run/i);
});

test('detection counts up and drives a real bar', () => {
  const j = job({ phase: 'detecting', done: 3, total: 40 });
  assert.equal(previewStatusLabel(j), 'Analyzing image 4 of 40…');
  assert.deepEqual(previewProgressValue(j), { done: 3, total: 40 });
  assert.equal(previewPercent(j), 8);
});

test('the count never overshoots its own total', () => {
  const j = job({ phase: 'detecting', done: 40, total: 40 });
  assert.equal(previewStatusLabel(j), 'Analyzing image 40 of 40…');
  assert.equal(previewPercent(j), 100);
});

test('a failure reads as the failure, and stops the spinner', () => {
  const j = job({ phase: 'loading', finished: true,
    error: 'face detection stopped unexpectedly (exit 3)' });
  assert.equal(previewStatusLabel(j), 'face detection stopped unexpectedly (exit 3)');
  assert.equal(previewRunning(j), false);
  assert.equal(previewError(j), 'face detection stopped unexpectedly (exit 3)');
  assert.equal(previewProgressValue(j), null);
});

test('no job at all is not a running state', () => {
  assert.equal(previewRunning(null), false);
  assert.equal(previewStatusLabel(null), '');
  assert.equal(previewError(null), '');
});

test('a finished pass with zero faces is not an error', () => {
  const j = job({ phase: 'detecting', done: 5, total: 5, finished: true });
  assert.equal(previewError(j), '');
  assert.equal(previewRunning(j), false);
});
