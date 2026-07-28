import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {
  deleteDestination, deletePreviewState, isRecoverable, overlapNotice, sharedFileCount,
  sharedFilesWarning,
} from './bankOverlap.js';

test('a permanent delete is never described as recoverable', () => {
  assert.equal(isRecoverable('trash'), true);
  assert.equal(isRecoverable('app_trash'), true);
  assert.equal(isRecoverable('delete'), false);
  assert.match(deleteDestination('delete'), /for good/);
});

test('each destination names a place the user can actually go look', () => {
  assert.match(deleteDestination('trash'), /Recycle Bin/);
  assert.match(deleteDestination('app_trash'), /Settings/);
});

test('an unknown mode is treated as permanent, never as safe', () => {
  assert.equal(isRecoverable(undefined), false);
  assert.match(deleteDestination(undefined), /for good/);
});

test('overlapNotice: a bank on its own says nothing', () => {
  assert.equal(overlapNotice([]), null);
  assert.equal(overlapNotice(null), null);
  assert.equal(overlapNotice([{ id: 1 }]), null);   // nameless row is not a notice
});

test('overlapNotice names the other bank and why it matters', () => {
  const n = overlapNotice([{ id: 2, name: 'Everything', relation: 'parent' }]);
  assert.match(n, /Everything/);
  assert.match(n, /Delete rejected/);
});

test('overlapNotice lists every overlapping bank', () => {
  const n = overlapNotice([{ id: 2, name: 'A' }, { id: 3, name: 'B' }]);
  assert.match(n, /“A”, “B”/);
});

test('sharedFilesWarning: nothing shared -> no warning', () => {
  assert.equal(sharedFilesWarning({ shared: [] }), null);
  assert.equal(sharedFilesWarning(null), null);
  assert.equal(sharedFilesWarning({ shared: [{ name: 'A', files: 0 }] }), null);
});

test('sharedFilesWarning says how many files the other bank loses', () => {
  const w = sharedFilesWarning({ shared: [{ id: 2, name: 'Everything', files: 1541 }] });
  assert.match(w, /1541/);
  assert.match(w, /Everything/);
  assert.match(w, /decisions/);      // the cost is the triage, not just the bytes
});

test('sharedFileCount totals across banks and survives junk', () => {
  assert.equal(sharedFileCount({ shared: [{ files: 3 }, { files: 4 }] }), 7);
  assert.equal(sharedFileCount({ shared: [{ files: 'x' }] }), 0);
  assert.equal(sharedFileCount(null), 0);
});

// ── The destructive button fails CLOSED when its evidence is missing ─────────

test('deletePreviewState: no preview yet -> not ready, and it says it is checking', () => {
  const s = deletePreviewState(null);
  assert.equal(s.ready, false);
  assert.equal(s.state, 'checking');
  assert.match(s.title, /Checking/);
});

test('deletePreviewState: a FAILED preview never arms the button', () => {
  const s = deletePreviewState({ failed: true });
  assert.equal(s.ready, false);
  assert.equal(s.state, 'failed');
  // and it says so out loud — the old code dropped the ⚠ banner in silence
  assert.match(s.title, /Could not check/);
  assert.match(s.text, /Nothing is deleted/);
});

test('deletePreviewState: a real preview arms, even with nothing shared', () => {
  assert.equal(deletePreviewState({ mode: 'trash', shared: [] }).ready, true);
  assert.equal(deletePreviewState({ mode: 'delete' }).state, 'ready');
});

test('the delete dialog wires the fail-closed check into the destructive button', () => {
  const src = fs.readFileSync(new URL('./DeleteRejectedDialog.jsx', import.meta.url), 'utf8');
  // armed alone is not enough: the preview has to have landed
  assert.match(src, /disabled=\{busy \|\| !armed \|\| !check\.ready\}/);
  assert.match(src, /if \(busy \|\| !armed \|\| !check\.ready\) return/);
  // and an unverified destination is never asserted as fact
  assert.match(src, /still being checked/);
});
