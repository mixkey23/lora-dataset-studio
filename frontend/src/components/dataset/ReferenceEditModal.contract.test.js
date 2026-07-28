import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

/* `node --test` cannot parse JSX, so the modal itself is pinned by READING it.
   Crude on purpose: the helpers next door are unit-tested to death, and none of
   that is worth anything if the component quietly stops calling them. This file
   exists because that has already happened once in this modal — an invisible fix
   (the opaque panel token) died in a rewrite with every test still green. */

const here = dirname(fileURLToPath(import.meta.url));
const modal = readFileSync(join(here, 'ReferenceEditModal.jsx'), 'utf8');
const workspace = readFileSync(join(here, 'DatasetWorkspace.jsx'), 'utf8');

test('the engine list is DERIVED, never spelled out in the component', () => {
  assert.ok(modal.includes('editEngineOptions('), 'the modal must build its list from the helper');
  // A literal engine label in the JSX is the hardcoded list coming back.
  for (const label of ['Krea 2 Edit', 'Nano Banana Pro', 'OpenRouter']) {
    assert.ok(!modal.includes(`>${label}<`), `${label} is spelled out in the JSX`);
  }
});

test('the cost and Keep lines come from the engine-aware helpers', () => {
  assert.ok(modal.includes('editCostNote('), 'cost line must be per-engine');
  assert.ok(modal.includes('editKeepNote('), 'Keep line must be per-engine');
  // The old unconditional sentence must be gone: it is a lie on a local engine.
  assert.ok(!modal.includes('Each edit is a paid API call'),
    'the hardcoded "paid API call" line is back in the JSX');
});

test('the reference picker is gated, and the reference note is rendered', () => {
  assert.ok(modal.includes('acceptsExtraEditRefs('), 'the picker must be gated per engine');
  assert.ok(modal.includes('editRefNote(') && modal.includes('{refNote}'),
    'the modal must SHOW what the engine does with extra references');
});

test('an unavailable engine renders its reason, not just a disabled button', () => {
  assert.ok(modal.includes('current?.blocked') || modal.includes('current.blocked'),
    'the blocked reason must reach the render');
  assert.ok(modal.includes('editBlockedReason(prompt, engine, '),
    'the blocked reason must also gate the Generate button');
});

test('the workspace hands the modal the SAME capabilities the generation panel reads', () => {
  assert.ok(workspace.includes('localEngineUnavailableReason'),
    'the reason must be the shared one, not a second inline copy');
  assert.ok(workspace.includes('comfyuiConfigured={hasComfyui(caps)}'),
    'an install with no ComfyUI must be told apart from one with a fixable gap');
  assert.ok(workspace.includes('engineAvailable={caps.engines'),
    'live readiness must reach the modal');
});

test('the engine pills wrap instead of overflowing a 400px screen', () => {
  // Five pills on a narrow phone. `flex-wrap` is the whole mechanism; losing it
  // pushes the modal wider than the viewport, which is not visible on a desktop
  // rewrite and is exactly how this regresses.
  const pills = modal.slice(modal.indexOf('{options.map('));
  assert.ok(modal.includes('flex-wrap min-w-0'),
    'the engine row must wrap and be allowed to shrink');
  assert.ok(pills.length > 0);
});
