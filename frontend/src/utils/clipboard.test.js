import assert from 'node:assert/strict';
import { test } from 'node:test';
import { copyToClipboard } from './clipboard.js';

/* Repo owner report: "Copy diagnostic report" failed with "Cannot read
 * properties of undefined (reading 'writeText')" — navigator.clipboard is
 * only defined in a secure context, and every copy button called it
 * directly. These stub `navigator`/`document` minimally (no jsdom in this
 * runner) to exercise both the modern and fallback paths without a browser. */

// Node ships a read-only global `navigator` (Node 21+) — plain assignment
// throws "which has only a getter". defineProperty overrides it instead.
function withGlobals(overrides, fn) {
  const saved = {};
  for (const key of Object.keys(overrides)) {
    saved[key] = Object.getOwnPropertyDescriptor(globalThis, key);
    Object.defineProperty(globalThis, key,
      { value: overrides[key], configurable: true, writable: true });
  }
  const restore = () => {
    for (const key of Object.keys(overrides)) {
      if (saved[key]) Object.defineProperty(globalThis, key, saved[key]);
      else delete globalThis[key];
    }
  };
  try {
    const result = fn();
    return Promise.resolve(result).finally(restore);
  } catch (e) {
    restore();
    throw e;
  }
}

test('copyToClipboard uses navigator.clipboard.writeText when available', async () => {
  const calls = [];
  await withGlobals({
    navigator: { clipboard: { writeText: async (t) => { calls.push(t); } } },
  }, () => copyToClipboard('hello'));
  assert.deepEqual(calls, ['hello']);
});

test('copyToClipboard falls back to execCommand when navigator.clipboard is undefined', async () => {
  // The exact bug reported: navigator exists but .clipboard does not (plain
  // HTTP LAN origin / older embedded webview) — must not throw
  // "Cannot read properties of undefined (reading 'writeText')".
  const appended = [];
  const fakeTextarea = {
    style: {},
    focus() {}, select() {}, value: '',
  };
  await withGlobals({
    navigator: {},
    document: {
      createElement: () => fakeTextarea,
      body: {
        appendChild: (el) => appended.push(el),
        removeChild: (el) => { appended.splice(appended.indexOf(el), 1); },
      },
      execCommand: () => true,
    },
  }, () => copyToClipboard('fallback text'));
  assert.equal(fakeTextarea.value, 'fallback text');
  assert.equal(appended.length, 0);   // cleaned up after copying
});

test('copyToClipboard throws when both the modern and fallback paths fail', async () => {
  const fakeTextarea = { style: {}, focus() {}, select() {}, value: '' };
  await assert.rejects(
    () => withGlobals({
      navigator: {},
      document: {
        createElement: () => fakeTextarea,
        body: { appendChild: () => {}, removeChild: () => {} },
        execCommand: () => false,
      },
    }, () => copyToClipboard('nope')),
  );
});

test('copyToClipboard treats null/undefined as empty string', async () => {
  const calls = [];
  await withGlobals({
    navigator: { clipboard: { writeText: async (t) => { calls.push(t); } } },
  }, () => copyToClipboard(undefined));
  assert.deepEqual(calls, ['']);
});
