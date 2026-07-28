import test from 'node:test';
import assert from 'node:assert/strict';
import {
  cascadeBlockedReason, cascadeConfirmation, cascadeKeeps, cascadeLosses,
  cascadeResultMessage, formatBytes,
} from './runCascadeDeletion.js';

const IMPACT = {
  notes: 3, previews: 2, canvas_positions: 1, children_detached: 2,
  archived_images_released: 4,
  cascade: {
    checkpoints: 14, checkpoint_bytes: 24 * 1024 * 1024 * 1024,
    images_deleted: 37, images_kept_rated: 5, deployed_kept: 1,
    training_active: null,
  },
};

test('formatBytes: file-manager units, blank for nothing', () => {
  assert.equal(formatBytes(0), '');
  assert.equal(formatBytes(null), '');
  assert.equal(formatBytes(512), '512 B');
  assert.equal(formatBytes(1024 * 1024 * 1536), '1.5 GB');
});

test('the confirmation counts what disappears, checkpoints and size first', () => {
  const losses = cascadeLosses(IMPACT);
  assert.equal(losses[0], '14 checkpoints · 24.0 GB');
  assert.equal(losses[1], '37 generated images');
  assert.ok(losses.includes('3 checkpoint notes'));
  assert.ok(losses.includes('its place on the canvas'));
});

test('zero counts are omitted, never printed as "0 notes"', () => {
  const losses = cascadeLosses({ notes: 0, cascade: { checkpoints: 1, images_deleted: 0 } });
  assert.deepEqual(losses, ['1 checkpoint']);
});

test('what survives is stated: children, 👍 images, deployed LoRAs', () => {
  const keeps = cascadeKeeps(IMPACT);
  assert.ok(keeps.some((l) => l.includes('2 runs that continued from it are kept')));
  assert.ok(keeps.some((l) => l.includes('5 images you marked 👍 are kept')));
  assert.ok(keeps.some((l) => l.includes('1 LoRA already deployed into ComfyUI stays there')));
});

test('the deployed-LoRA line agrees with itself in the plural', () => {
  const [line] = cascadeKeeps({ cascade: { deployed_kept: 3 } });
  assert.match(line, /3 LoRAs already deployed into ComfyUI stay there/);
  assert.match(line, /if you want them gone/);
});

test('a run with nothing attached says so instead of an empty list', () => {
  assert.deepEqual(cascadeKeeps({}), ['Nothing else in the app points at this run.']);
});

test('a training run is refused with the reason, local and cloud told apart', () => {
  assert.equal(cascadeBlockedReason(IMPACT), null);
  assert.match(cascadeBlockedReason({ cascade: { training_active: 'local' } }),
    /training right now/);
  assert.match(cascadeBlockedReason({ cascade: { training_active: 'cloud' } }),
    /cloud pod/);
});

test('a missing impact is flagged unknown rather than invented', () => {
  const c = cascadeConfirmation(12, null);
  assert.equal(c.unknown, true);
  assert.deepEqual(c.losses, []);
  assert.match(c.title, /Delete run #12 and everything it produced\?/);
});

test('the result message carries the real counts, not a flat success', () => {
  assert.equal(cascadeResultMessage({ checkpoints_deleted: 2, images_deleted: 7 }),
    'Run deleted — 2 checkpoints and 7 images removed.');
  assert.equal(cascadeResultMessage({ checkpoints_deleted: 1, images_kept: 3 }),
    'Run deleted — 1 checkpoint removed. 3 images kept.');
  assert.equal(cascadeResultMessage({}), 'Run deleted.');
});
