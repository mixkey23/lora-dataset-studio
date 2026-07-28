import test from 'node:test';
import assert from 'node:assert/strict';
import { runSilenceWarning, stopOutcomeMessage } from './runSilence.js';

const run = (over) => ({
  status: 'training', idle_seconds: 0, idle_limit_seconds: 45 * 60, ...over,
});

test('a progressing run says nothing', () => {
  assert.equal(runSilenceWarning(run({ idle_seconds: 30 })), null);
});

test('a training run silent for 12 min warns and names the cut-off', () => {
  const w = runSilenceWarning(run({ idle_seconds: 12 * 60 }));
  assert.equal(w.level, 'warn');
  assert.match(w.text, /12 min/);
  assert.match(w.text, /45 min/);
});

test('past the limit the message stops promising a future termination', () => {
  const w = runSilenceWarning(run({ idle_seconds: 50 * 60 }));
  assert.equal(w.level, 'critical');
  assert.match(w.text, /being terminated/);
});

test('with the watchdog off the user is told to act', () => {
  const w = runSilenceWarning(run({ idle_seconds: 3 * 3600, idle_limit_seconds: 0 }));
  assert.equal(w.level, 'warn');
  assert.match(w.text, /Automatic termination is off/);
});

test('slow-by-design phases are not called out at training thresholds', () => {
  for (const status of ['preparing', 'provisioning', 'uploading', 'downloading']) {
    assert.equal(runSilenceWarning(run({ status, idle_seconds: 20 * 60 })), null, status);
  }
});

test('finished runs never warn', () => {
  assert.equal(runSilenceWarning(run({ status: 'done', idle_seconds: 99999 })), null);
  assert.equal(runSilenceWarning(null), null);
});

test('a failed stop names the instance instead of claiming success', () => {
  const m = stopOutcomeMessage({
    ok: false, error: 'Could not terminate instance 12345 — destroy it in the vast.ai console.',
  });
  assert.equal(m.level, 'error');
  assert.match(m.text, /12345/);
});

test('a forced stop says the pod was shut down directly', () => {
  const m = stopOutcomeMessage({ ok: true, mode: 'forced' });
  assert.equal(m.level, 'warn');
  assert.match(m.text, /terminated/i);
});

test('a graceful stop keeps the calm wording', () => {
  assert.equal(stopOutcomeMessage({ ok: true, mode: 'graceful' }).level, 'info');
});
