import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {
  bestUpgrade,
  canSelect,
  detectionFailure,
  detectionSummary,
  dialogCopy,
  enteredNote,
  missingLabels,
  openerLabel,
  selectionNote,
  sortInterpreters,
  statusBadge,
} from './scoringPython.js';

const DEPS = ['PyTorch', 'OpenCLIP', 'Transformers', 'timm', 'NumPy', 'Pillow'];

function row(over = {}) {
  const missing = over.missingLabels || [];
  return {
    path: 'python',
    label: 'Some Python',
    status: 'gpu_ready',
    cuda: true,
    usable: missing.length === 0,
    selected: false,
    detail: '',
    deps: DEPS.map((label) => ({ label, present: !missing.includes(label) })),
    ...over,
  };
}

test('a CUDA interpreter missing OpenCLIP is not usable, and says which package', () => {
  const r = row({ status: 'incomplete', cuda: true, missingLabels: ['OpenCLIP'] });
  assert.deepEqual(missingLabels(r), ['OpenCLIP']);
  assert.equal(canSelect(r), false, 'CUDA alone must never be enough to pick it');
  assert.equal(statusBadge(r.status).label, 'Missing packages');
  // The summary names the interpreter AND the package — the whole point of the
  // feature is saying "ai-toolkit has CUDA but lacks OpenCLIP", not "no".
  const summary = detectionSummary([r]);
  assert.match(summary, /OpenCLIP/);
  assert.match(summary, /Some Python/);
});

test('the best suggestion is a GPU-ready interpreter that is not already in use', () => {
  const rows = [
    row({ label: 'App Python', status: 'unreachable', cuda: false, usable: false }),
    row({ label: 'Scoring env', status: 'cpu_only', cuda: false, selected: true }),
    row({ label: 'ai-toolkit', status: 'gpu_ready' }),
  ];
  assert.equal(bestUpgrade(rows).label, 'ai-toolkit');
  assert.equal(sortInterpreters(rows)[0].label, 'ai-toolkit');
  assert.equal(sortInterpreters(rows).at(-1).label, 'App Python');
  assert.equal(detectionSummary(rows), '1 of 3 can run ✨ Score on your GPU.');
});

test('the one already selected is never offered again', () => {
  const rows = [row({ label: 'ai-toolkit', selected: true })];
  assert.equal(bestUpgrade(rows), null);
  assert.equal(canSelect(rows[0]), false);
});

test('no GPU anywhere is stated plainly instead of inviting a hunt', () => {
  const rows = [row({ label: 'Scoring env', status: 'cpu_only', cuda: false })];
  assert.match(detectionSummary(rows), /stays on the CPU/);
  assert.equal(bestUpgrade(rows), null);
  assert.equal(canSelect(rows[0]), true, 'a complete CPU interpreter is still a valid choice');
});

test('an interpreter that did not answer is inert, never a crash', () => {
  const r = row({ status: 'unreachable', cuda: false, usable: false, deps: [] });
  assert.equal(canSelect(r), false);
  assert.equal(statusBadge(r.status).label, 'No answer');
  assert.deepEqual(missingLabels(r), []);
  assert.deepEqual(missingLabels(null), []);
  assert.equal(detectionSummary([]), 'No Python interpreters found to check yet.');
  assert.equal(detectionSummary(null), 'No Python interpreters found to check yet.');
});

// ── Four machines, four sentences ────────────────────────────────────────────
// This app runs on installs we will never see. Each of them must get the line
// that is true FOR IT — that is the whole feature. An opaque "no" is the
// failure mode we are designing against.

test('a machine with no NVIDIA card is never shown a word about CUDA', () => {
  const rows = [row({ label: 'App Python', status: 'incomplete', cuda: false,
    missingLabels: ['PyTorch', 'OpenCLIP'] })];
  const summary = detectionSummary(rows, false);
  assert.match(summary, /No NVIDIA card detected/);
  assert.match(summary, /runs on the CPU either way/);
  assert.ok(!/CUDA/i.test(summary), 'nothing to fix, so nothing is suggested');
  const copy = dialogCopy(false);
  assert.ok(!/CUDA/i.test(copy.title + copy.intro), 'no CUDA pitch to a card-less machine');
  assert.ok(!/GPU/.test(openerLabel(false)), 'and no promise of a "GPU Python"');
  // …but the offer itself survives: borrowing still saves a second install.
  assert.match(copy.intro, /already carries them/);
});

test('a card-less machine that HAS a usable interpreter is told it can skip the install', () => {
  const rows = [row({ label: 'ComfyUI', status: 'cpu_only', cuda: false })];
  const summary = detectionSummary(rows, false);
  assert.match(summary, /1 interpreter here already has the packages/);
  assert.ok(!/faster|speed|hours/i.test(summary), 'never sold a speed-up it cannot have');
});

test('a machine with a card and nothing usable is told exactly what to do next', () => {
  const rows = [row({ label: 'App Python', status: 'unreachable', cuda: false, usable: false })];
  const summary = detectionSummary(rows, true);
  assert.match(summary, /stays on the CPU/);
  assert.match(summary, /enter its path below/, 'the manual route is offered, not hidden');
});

test('nothing found at all is a state, not an error', () => {
  assert.equal(detectionSummary([], true), 'No Python interpreters found to check yet.');
  assert.match(detectionSummary([], false), /No NVIDIA card detected/);
  assert.equal(bestUpgrade([]), null);
  assert.equal(sortInterpreters(undefined).length, 0);
});

test('the wording defaults to "there is a card" while the probe has not answered', () => {
  // Flashing "no NVIDIA card" at someone who has one is the one wrong guess.
  assert.match(detectionSummary([row()]), /can run/);
  assert.match(dialogCopy().title, /GPU Python/);
  assert.match(openerLabel(), /GPU Python/);
});

// ── The typed path always gets an answer ─────────────────────────────────────

test('a typed path that lands on an interpreter already listed says which one', () => {
  // Found live: the row already existed, so nothing visibly happened and the
  // button read as broken. This is the route most installs depend on.
  const note = enteredNote({
    entered_status: 'resolved',
    interpreters: [row({ label: "The app's own Python", source: 'app', entered: true,
      detail: 'ready — scores on a 4090' })],
  });
  assert.equal(note.tone, 'info');
  assert.match(note.text, /already listed as “The app's own Python”/);
  assert.match(note.text, /4090/);
});

test('a folder holding no interpreter is named as such, never left silent', () => {
  const note = enteredNote({ entered_status: 'no_interpreter', interpreters: [] });
  assert.equal(note.tone, 'warn');
  assert.match(note.text, /holds nothing that looks like a Python interpreter/);
  assert.match(note.text, /or at the environment folder/);
});

test('a freshly checked new interpreter reports its verdict, and silence means silence', () => {
  const note = enteredNote({
    entered_status: 'resolved',
    interpreters: [row({ source: 'manual', entered: true, detail: 'missing OpenCLIP' })],
  });
  assert.match(note.text, /Checked: missing OpenCLIP/);
  assert.equal(enteredNote({ entered_status: '', interpreters: [] }), null);
  assert.equal(enteredNote(null), null);
});

test('a failed search says so, instead of passing for an empty machine', () => {
  const f = detectionFailure({ detection_failed: true, detection_error: 'boom', interpreters: [] });
  assert.match(f.title, /Could not look/);
  assert.match(f.text, /the search failed/);
  assert.match(f.text, /Check again/);      // the retry is the whole point
  assert.equal(f.detail, 'boom');
});

test('an honestly empty machine is never flagged as a failure', () => {
  assert.equal(detectionFailure({ interpreters: [] }), null);
  assert.equal(detectionFailure({ detection_failed: false, interpreters: [] }), null);
  assert.equal(detectionFailure(null), null);
});

test('the picker replaces its summary line with the failure banner', () => {
  const src = fs.readFileSync(new URL('./ScoringPythonDialog.jsx', import.meta.url), 'utf8');
  // the "No Python interpreters found to check yet" line must not run for a
  // list that is empty only because the search broke
  assert.match(src, /!loading && !failure && \(/);
  assert.match(src, /detectionFailure\(result\)/);
});

test('the panel names the interpreter in use, and stays quiet on the default', () => {
  const chosen = row({ label: 'ai-toolkit', selected: true, detail: 'ready — scores on a 4090' });
  assert.match(selectionNote({ interpreters: [chosen] }), /ai-toolkit/);
  assert.match(selectionNote({ interpreters: [chosen] }), /4090/);
  assert.equal(selectionNote({ interpreters: [row()] }), null);
  assert.equal(selectionNote(null), null);
});
