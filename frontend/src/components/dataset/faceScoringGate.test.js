import test from 'node:test';
import assert from 'node:assert/strict';
import { faceAnalysisState, faceAnalysisLabel, autoTriageAvailable } from './faceScoringGate.js';

const BLOCKED = 'Face similarity needs a photographic face; it cannot read a drawn one. '
  + 'Set the subject type to Human if this dataset is photographic.';

test('a blocked dataset disables the pass AND carries the reason', () => {
  const s = faceAnalysisState({ blockedReason: BLOCKED, hasRef: true, busy: false });
  assert.equal(s.disabled, true);
  // "Disappears in silence" is the bug, not the fix: the reason IS the title.
  assert.equal(s.title, BLOCKED);
  assert.equal(s.blocked, true);
});

test('the reason wins over "set a reference photo first"', () => {
  // A drawn dataset with no reference must not be sent off to add one: it would
  // not have helped.
  const s = faceAnalysisState({ blockedReason: BLOCKED, hasRef: false, busy: false });
  assert.equal(s.title, BLOCKED);
});

test('a photographic dataset is strictly unchanged', () => {
  const s = faceAnalysisState({ blockedReason: null, hasRef: true, busy: false });
  assert.equal(s.disabled, false);
  assert.equal(s.blocked, false);
  assert.match(s.title, /facial resemblance/i);

  const noRef = faceAnalysisState({ blockedReason: null, hasRef: false, busy: false });
  assert.equal(noRef.disabled, true);
  assert.equal(noRef.title, 'Set a reference photo first');

  const busy = faceAnalysisState({ blockedReason: null, hasRef: true, busy: true });
  assert.equal(busy.disabled, true);
  assert.match(busy.title, /facial resemblance/i);
});

test('the gate is the SERVER string — the UI never re-derives the rule', () => {
  // Any non-empty reason blocks, whatever it says: when the server grows a second
  // reason, the UI already honours it. Nothing here mentions "anime".
  const s = faceAnalysisState({ blockedReason: 'some future reason', hasRef: true });
  assert.equal(s.disabled, true);
  assert.equal(s.title, 'some future reason');
  assert.doesNotMatch(faceAnalysisState.toString(), /anime/i);
});

// ── Missing install: name the gap, never an inert button ────────────────────

test('without insightface the button says what is missing AND where to get it', () => {
  const s = faceAnalysisState({ blockedReason: null, hasRef: true, capable: false });
  assert.equal(s.disabled, true);
  assert.equal(s.blocked, true, 'blocked => the reason is rendered in place, not only in a tooltip');
  assert.match(s.reason, /not installed/i);
  assert.match(s.reason, /Setup/i, 'a gap with no way out is just a dead end');
  assert.equal(s.setupRoute, '#/setup?step=quality');
});

test('the server refusal outranks a missing install', () => {
  // Installing InsightFace would not make a drawn face readable: do not send
  // anyone off to install something that would not have helped.
  const s = faceAnalysisState({ blockedReason: BLOCKED, hasRef: true, capable: false });
  assert.equal(s.reason, BLOCKED);
  assert.equal(s.setupRoute, null);
});

test('a missing install outranks "set a reference photo first"', () => {
  const s = faceAnalysisState({ blockedReason: null, hasRef: false, capable: false });
  assert.match(s.reason, /not installed/i);
});

test('capabilities still loading does not flash "not installed"', () => {
  const s = faceAnalysisState({ blockedReason: null, hasRef: true, capable: false, capsLoading: true });
  assert.equal(s.blocked, false);
  assert.equal(s.reason, null);
});

// ── The button names its own scope ──────────────────────────────────────────

test('the label states how many images the pass will actually score', () => {
  // A mystery pass on an unknown set is what "🎭 Analyze faces" used to be.
  assert.equal(faceAnalysisLabel({ total: 42, unscored: 42 }), '🎭 Analyze faces (42)');
  assert.equal(faceAnalysisLabel({ total: 42, unscored: 7 }), '🎭 Analyze faces (42 · 7 new)');
  assert.equal(faceAnalysisLabel({ total: 42, unscored: 0 }), '🎭 Analyze faces (42)');
  // An older payload has no scope: fall back to the bare label rather than
  // inventing a count.
  assert.equal(faceAnalysisLabel(null), '🎭 Analyze faces');
  assert.equal(faceAnalysisLabel(undefined), '🎭 Analyze faces');
  assert.equal(faceAnalysisLabel({ total: 0, unscored: 0 }), '🎭 Analyze faces');
  assert.equal(faceAnalysisLabel({ total: 'x' }), '🎭 Analyze faces');
});

test('the tooltip says the triage pile is in scope', () => {
  // The scope change is the whole point: generated variations sit at 'pending',
  // and until now the pass never touched them.
  const s = faceAnalysisState({ blockedReason: null, hasRef: true });
  assert.match(s.title, /✓\/✕/, 'the undecided pile must be named, not implied');
});

test('auto-triage is withheld when scoring is blocked', () => {
  // Auto-triage BATCHES keep/reject from face_score. Letting it act on scores from
  // a pass that cannot read this dataset is the one place stale numbers do damage.
  assert.equal(autoTriageAvailable(BLOCKED), false);
  assert.equal(autoTriageAvailable(null), true);
  assert.equal(autoTriageAvailable(undefined), true);
  assert.equal(autoTriageAvailable(''), true);
});
