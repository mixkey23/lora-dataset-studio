import test from 'node:test';
import assert from 'node:assert/strict';
import { CUDA_TORCH_DOWNLOAD, holdsTheGpu, scoreDeviceNote } from './bankScoreDevice.js';

test('a GPU pass says nothing — there is nothing to fix', () => {
  assert.equal(scoreDeviceNote({ device: 'cuda', gpu: true }), null);
});

test('no payload yet -> no note, never a flash of wrong advice', () => {
  assert.equal(scoreDeviceNote(null), null);
  assert.equal(scoreDeviceNote(undefined), null);
});

test('nothing is said while the scoring extra is not installed', () => {
  const cpu = { device: 'cpu', gpu: false, gpu_present: true, eta_minutes: 57 };
  assert.equal(scoreDeviceNote(cpu, false), null);   // the button already says "needs setup"
  assert.notEqual(scoreDeviceNote(cpu, true), null);
});

test('a machine with a card gets the warning AND the download size', () => {
  const n = scoreDeviceNote({ device: 'cpu', gpu: false, gpu_present: true, eta_minutes: 57 });
  assert.equal(n.tone, 'warn');
  assert.match(n.text, /57 minutes/);
  assert.match(n.text, /20×/);
  assert.ok(n.text.includes(CUDA_TORCH_DOWNLOAD), 'the 2.5 GB cost must be stated up front');
});

test('a machine WITHOUT a card is told how it is, not sold a CUDA install', () => {
  const n = scoreDeviceNote({ device: 'cpu', gpu: false, gpu_present: false, eta_minutes: 12 });
  assert.equal(n.tone, 'info');
  assert.match(n.text, /no NVIDIA GPU detected/);
  assert.ok(!n.text.includes(CUDA_TORCH_DOWNLOAD), 'never suggest a download it cannot use');
});

test('one minute is singular, and a missing estimate is simply omitted', () => {
  assert.match(scoreDeviceNote({ device: 'cpu', gpu_present: true, eta_minutes: 1 }).text,
    /About 1 minute for/);
  const none = scoreDeviceNote({ device: 'cpu', gpu_present: true, eta_minutes: null });
  assert.ok(!/About/.test(none.text));
});

test('holdsTheGpu is true only for a pass that really takes the card', () => {
  assert.equal(holdsTheGpu({ device: 'cuda', gpu: true }), true);
  assert.equal(holdsTheGpu({ device: 'cpu', gpu: false }), false);
  assert.equal(holdsTheGpu(null), false);
});
