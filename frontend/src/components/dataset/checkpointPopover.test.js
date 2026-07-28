import assert from 'node:assert/strict';
import test from 'node:test';

import {
  POPOVER_H, POPOVER_W, checkpointActionModel, checkpointPopoverPlacement,
  clampPopoverToViewport, deployRefusal, downloadRefusal,
} from './checkpointPopover.js';

/* The popover's brain. Every case below is one of "the app must work
   everywhere": a checkpoint that is not deployed, a file that left the disk, a
   cloud run that is not linked here, a run still training. In each of them the
   popover owes the user either a live action or a stated reason — never a button
   that does nothing. */

const localNode = {
  record_id: 12, source: 'local', status: 'done', train_type: 'flux',
  variant: 'dev', base_model: 'flux1-dev.safetensors',
};
const pill = (over = {}) => ({
  step: 2000, w: 60, h: 20, x: 100, y: 200,
  filename: 'ck-2000.safetensors', download_url: '/dl/ck-2000', present: true,
  ...over,
});

test('a plain local checkpoint offers download, deploy and delete', () => {
  const a = checkpointActionModel(localNode, pill(), { folderLabel: 'loras/flux' });
  assert.equal(a.download.url, '/dl/ck-2000');
  assert.equal(a.deployed, false);
  assert.ok(a.deploy.payload, 'deployable');
  assert.equal(a.deploy.folder, 'loras/flux');
  assert.equal(a.undeploy, null);
  assert.equal(a.del.kind, 'save');
});

test('a DEPLOYED checkpoint swaps deploy for ✓ Deployed + ⏏ Undeploy, and hides the 🗑 save row', () => {
  // Progressive deletion: while the ComfyUI copy exists, the destructive row has
  // nothing to offer yet — undeploy first, and only then does 🗑 reach the save.
  const a = checkpointActionModel(localNode,
    pill({ testable: true, deployed_filename: 'flux_ds_2000.safetensors' }));
  assert.equal(a.deployed, true);
  assert.equal(a.deploy, null);
  assert.equal(a.undeploy.label, 'Undeploy');
  assert.match(a.undeploy.title, /Reversible/);
  assert.equal(a.del, null);
});

test('a save that left the disk states it instead of offering a dead download', () => {
  const a = checkpointActionModel(localNode, pill({ present: false, download_url: null }));
  assert.equal(a.download.url, undefined);
  assert.match(a.download.reason, /no longer on this machine/);
  assert.match(a.deploy.reason, /no longer on disk/);
  assert.equal(a.del, null, 'nothing to delete either');
});

test('a checkpoint with no file says so rather than offering to deploy nothing', () => {
  assert.match(deployRefusal(localNode, pill({ filename: null })), /no file/);
  const a = checkpointActionModel(localNode, pill({ filename: null }));
  assert.equal(a.deploy.payload, undefined);
  assert.ok(a.deploy.reason);
});

test('an unlinked cloud run cannot be deployed, and the reason names the cause', () => {
  const cloud = { ...localNode, source: 'cloud', run_id: null, status: 'done' };
  assert.match(deployRefusal(cloud, pill()), /cloud run is not linked/);
  const a = checkpointActionModel(cloud, pill());
  assert.ok(a.deploy.reason, 'stated, not silently missing');
});

test('a cloud run still in flight offers no delete at all (its pod keeps syncing)', () => {
  const flying = { ...localNode, source: 'cloud', run_id: 7, status: 'training' };
  const a = checkpointActionModel(flying, pill());
  assert.equal(a.del, null);
  assert.equal(a.undeploy, null);
});

test('▶ Continue is live only where a host wired it, and otherwise SAYS where it lives', () => {
  const cloud = { ...localNode, source: 'cloud', run_id: 7, status: 'done' };
  // The Runs hub: cloud lane + a handler → live.
  assert.deepEqual(
    checkpointActionModel(cloud, pill(), { hasContinueHandler: true }).continue, { ok: true });
  // A host with neither handler nor explanation shows no row at all — better
  // than a disabled one that explains nothing.
  assert.equal(checkpointActionModel(cloud, pill()).continue, null);
  // The canvas: no resume flow of its own, so the row states where the gesture is.
  const onCanvas = checkpointActionModel(cloud, pill(), { continueReason: 'open it from the Runs page' });
  assert.equal(onCanvas.continue.ok, undefined);
  assert.match(onCanvas.continue.reason, /Runs page/);
  // A LOCAL run stays out of the cloud lane, exactly as before.
  assert.equal(
    checkpointActionModel(localNode, pill(), { hasContinueHandler: true }).continue, null);
  assert.deepEqual(
    checkpointActionModel(localNode, pill(),
      { hasContinueHandler: true, continueSource: 'any' }).continue, { ok: true });
});

test('a RUN card opens the same popover with only its run-level rows', () => {
  const a = checkpointActionModel(localNode, null);
  assert.equal(a.isRun, true);
  for (const k of ['download', 'continue', 'deploy', 'undeploy', 'del']) assert.equal(a[k], null);
});

test('no node at all answers null rather than half a popover', () => {
  assert.equal(checkpointActionModel(null, pill()), null);
  assert.equal(downloadRefusal(null), null);
});

/* --- geometry: the popover never leaves its frame ------------------------- */

test('inside an svg the popover flips above a bottom-row pill and clamps sideways', () => {
  const world = { width: 800, height: 400 };
  // Room below → sits under the pill.
  assert.deepEqual(checkpointPopoverPlacement({ x: 10, y: 20, h: 20 }, world),
    { x: 10, y: 44 });
  // No room below → flips above.
  const flipped = checkpointPopoverPlacement({ x: 10, y: 380, h: 20 }, world);
  assert.ok(flipped.y < 380, 'above the pill');
  assert.ok(flipped.y >= 0);
  // A pill at the right edge is pulled back inside.
  const right = checkpointPopoverPlacement({ x: 790, y: 20, h: 20 }, world);
  assert.equal(right.x, world.width - POPOVER_W);
});

test('on a 400-px screen the floating popover stays inside the window, both axes', () => {
  const viewport = { width: 400, height: 800 };
  const at = clampPopoverToViewport({ x: 395, y: 100 }, viewport);
  assert.ok(at.left >= 8, 'left margin kept');
  assert.ok(at.left + at.width <= 400 - 8, 'never past the right edge');
  assert.equal(at.width, POPOVER_W, '210 px fits a 400-px screen');
  // A click near the bottom flips the popover above it rather than off-screen.
  const low = clampPopoverToViewport({ x: 200, y: 780 }, viewport);
  assert.ok(low.top + POPOVER_H <= 800 - 8);
  assert.ok(low.top >= 8);
});

test('a window narrower than the popover NARROWS it — it never scrolls the page sideways', () => {
  const at = clampPopoverToViewport({ x: 60, y: 40 }, { width: 150, height: 600 });
  assert.ok(at.width <= 150 - 16, `narrowed to ${at.width}`);
  assert.ok(at.left + at.width <= 150);
});

test('a degenerate viewport does not produce NaN coordinates', () => {
  const at = clampPopoverToViewport(null, null);
  assert.ok(Number.isFinite(at.left) && Number.isFinite(at.top) && Number.isFinite(at.width));
});
