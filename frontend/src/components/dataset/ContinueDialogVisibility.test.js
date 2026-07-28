/* The ▶ Continue dialog must render where it can be SEEN.
 *
 * TrainingPanel is mounted once, inside the workspace's Training section, which
 * the workspace keeps `display:none` while another section is shown. Its
 * checkpoint manager, however, PORTALS into the Checkpoints section — so the
 * ▶ Continue buttons and the ◉ Graph pills are clickable from a section where
 * everything the panel renders on its own is invisible. Reported as "I click
 * Continue training and nothing happens": the dialog did open, inside the hidden
 * container (proven in a browser: the element was in the DOM with a 0×0 box under
 * a `display:none` ancestor), so there was no dialog, no toast and no request —
 * it only appeared once the user navigated to the Training section.
 *
 * Contract: the dialog is portalled to document.body, so no section's visibility
 * can swallow it. */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const panel = readFileSync(new URL('./TrainingPanel.jsx', import.meta.url), 'utf8');
const workspace = readFileSync(new URL('./DatasetWorkspace.jsx', import.meta.url), 'utf8');

test('the Continue dialog is rendered at the body level, not in the hidden panel subtree', () => {
  assert.match(panel, /\{continueOpen && createPortal\(\(\s*\n\s*<ContinueDialog/,
    'ContinueDialog must be wrapped in createPortal(...)');
  const open = panel.indexOf('{continueOpen && createPortal((');
  const close = panel.indexOf('), document.body)}', open);
  assert.ok(open > 0 && close > open,
    'the ContinueDialog portal must target document.body');
});

test('the premise still holds: the checkpoint manager portals out of a hideable section', () => {
  // If either of these changes, revisit the fix above rather than deleting it.
  assert.match(panel, /<CheckpointPortal host=\{checkpointHost\}>/);
  assert.match(workspace, /const sectionCls = \(id\) => \(section === id \? '[^']*' : 'hidden'\)/,
    'inactive sections are hidden with display:none while staying mounted');
});
