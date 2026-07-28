/* Krea 2 Edit — the one-click install, on the front-end side.
   Two behaviours that decide whether the feature reads as working:
   what the button queues (never a re-download of what the user already has),
   and the ONE thing no installer can do — a ComfyUI restart. */
import test from 'node:test';
import assert from 'node:assert/strict';

import { kreaUnavailableReason } from './kreaEngine.js';
import {
  kreaInstallPlan, kreaNeedsComfyuiRestart, KREA_INSTALL_ORDER,
  INSTALL_ALL_ACTION_LABELS, installAllPlan, installCatalog,
  deriveCapabilitySummary,
} from '../hooks/useSetupSteps.js';

const caps = (comfyui) => ({ comfyui });

test('the Krea button queues the node pack first, then the missing weights', () => {
  const plan = kreaInstallPlan(caps({
    dir_valid: true, reachable: true,
    krea_missing: ['krea_model', 'krea_vae'],
    krea_nodes_missing: ['Krea2EditModelPatch'],
    krea_nodes_installed: false,
  }));
  assert.deepEqual(plan, ['krea_nodes', 'krea_model', 'krea_vae']);
  // The order mirrors the backend group so the progress list is not a second truth.
  assert.equal(KREA_INSTALL_ORDER[0], 'krea_nodes');
});

test('a Krea asset the user already has is never queued again', () => {
  // Retrofit: someone who placed the weights by hand sees a SHORTER plan, not a
  // 20 GB re-download.
  assert.deepEqual(
    kreaInstallPlan(caps({ dir_valid: true, reachable: true, krea_missing: [], krea_nodes_missing: [] })),
    [],
  );
  assert.deepEqual(
    kreaInstallPlan(caps({ dir_valid: true, reachable: true, krea_missing: ['krea_identity_lora'] })),
    ['krea_identity_lora'],
  );
  // ComfyUI STOPPED: its node probe fails open, so "nothing missing" proves
  // nothing. A pack that is not on disk is still queued.
  assert.deepEqual(
    kreaInstallPlan(caps({ dir_valid: true, reachable: false, krea_missing: [], krea_nodes_missing: [] })),
    ['krea_nodes'],
  );
});

test('an installed-but-not-loaded node pack is a RESTART, not a re-install', () => {
  const pending = caps({
    dir_valid: true, reachable: true, krea_missing: [],
    krea_nodes_missing: ['Krea2EditModelPatch'], krea_nodes_installed: true,
  });
  assert.deepEqual(kreaInstallPlan(pending), []);
  assert.equal(kreaNeedsComfyuiRestart(pending), true);
  // …and the engine card must say exactly that, instead of telling someone to
  // install what they just watched install.
  const reason = kreaUnavailableReason({
    comfyuiReachable: true, missingNodes: ['Krea2EditModelPatch'],
    nodePackInstalled: true,
  });
  assert.match(reason, /restart ComfyUI/i);
  assert.doesNotMatch(reason, /^⚠ Install the/);
});

test('a pack that was never installed still says to install it', () => {
  const reason = kreaUnavailableReason({
    comfyuiReachable: true, missingNodes: ['Krea2EditModelPatch'],
    nodePackInstalled: false,
  });
  assert.match(reason, /Install the comfyui-krea2edit node pack/);
  assert.equal(kreaNeedsComfyuiRestart(caps({
    krea_nodes_missing: ['Krea2EditModelPatch'], krea_nodes_installed: false })), false);
});

test('nothing is planned without a validated ComfyUI folder', () => {
  // Nowhere to put 20 GB and third-party code — never guess a path.
  assert.deepEqual(kreaInstallPlan(caps({
    dir_valid: false, krea_missing: ['krea_model'],
    krea_nodes_missing: ['Krea2EditModelPatch'] })), []);
  assert.deepEqual(kreaInstallPlan(undefined), []);
});

test('"Install everything" never silently pulls the second engine', () => {
  // ~20 GB on top of Klein's ~20. The unattended shortcut stays Klein-only;
  // Krea installs when it is asked for.
  const plan = installAllPlan({
    python: { ml_supported: true },
    face_scoring: true, masks: true, watermark_inpaint: true,
    ollama: { reachable: false },
    comfyui: { dir_valid: true, klein_missing: ['klein_vae'],
      krea_missing: ['krea_model', 'krea_vae'],
      krea_nodes_missing: ['Krea2EditModelPatch'] },
  });
  assert.deepEqual(plan, ['klein_vae']);
});

test('every queued action has a human label', () => {
  for (const a of KREA_INSTALL_ORDER) {
    assert.ok(INSTALL_ALL_ACTION_LABELS[a], `no label for ${a}`);
  }
});

/* ── The certification bug ───────────────────────────────────────────────────
   Setup's last screen said "🎉 You're all set — 11 of 11 capabilities ready" on a
   machine with no Krea at all: the engine was not in the list, so it was not in
   the DENOMINATOR either. That is worse than a red row — the user finishes setup
   believing there is nothing left, and meets a dark engine card weeks later. An
   absent capability must be visible and COUNTED. */

test('Krea is a counted capability, never quietly dropped from the total', () => {
  const rows = deriveCapabilitySummary({
    engines: { nanobanana: true, chatgpt: true, openrouter: true, klein: true, krea: false },
    comfyui: { dir_valid: true, reachable: true, krea_missing: ['krea_model'],
      krea_nodes_missing: ['Krea2EditModelPatch'], krea_nodes_installed: false },
  });
  const krea = rows.find((r) => r.label === 'Krea 2 Edit (local)');
  assert.ok(krea, 'Krea must appear in the capability summary');
  assert.equal(krea.ok, false);
  assert.equal(!!krea.pending, false, 'a real disk gap is not a "waiting" state');
  // It counts against the total instead of disappearing from it.
  assert.ok(rows.length >= 12);
  assert.ok(rows.filter((r) => r.ok).length < rows.length);
});

test('an installed Krea reads as ready like any other capability', () => {
  const rows = deriveCapabilitySummary({
    engines: { krea: true },
    comfyui: { dir_valid: true, reachable: true, krea_missing: [], krea_nodes_installed: true },
  });
  assert.equal(rows.find((r) => r.label === 'Krea 2 Edit (local)').ok, true);
});

test('everything installed but ComfyUI not restarted reads as pending, not missing', () => {
  const rows = deriveCapabilitySummary({
    engines: { krea: false },
    comfyui: { dir_valid: true, reachable: true, krea_missing: [],
      krea_nodes_missing: ['Krea2EditModelPatch'], krea_nodes_installed: true },
  });
  const krea = rows.find((r) => r.label === 'Krea 2 Edit (local)');
  assert.equal(krea.pending, true);
  assert.match(krea.note, /restart ComfyUI/i);
});

test('the node pack never shows "Installed" while ComfyUI has not loaded it', () => {
  // Both lies are refused: green would certify what the app cannot see, and
  // "Not installed" would invite a re-install of a folder that is already right.
  const cat = Object.fromEntries(installCatalog({
    comfyui: { dir_valid: true, reachable: true, krea_missing: [],
      krea_nodes_missing: ['Krea2EditModelPatch'], krea_nodes_installed: true },
  }).map((c) => [c.action, c]));
  assert.equal(cat.krea_nodes.present, false);
  assert.equal(cat.krea_nodes.state, 'restart');
  assert.match(cat.krea_nodes.stateLabel, /Restart ComfyUI/i);
  assert.equal(cat.krea_nodes.available, true);   // a repair path always stays
});

test('the per-piece repair menu covers Krea too, pack included', () => {
  const cat = Object.fromEntries(installCatalog({
    comfyui: { dir_valid: true, reachable: true, klein_missing: [], krea_missing: ['krea_vae'],
      krea_nodes_missing: ['Krea2EditModelPatch'], krea_nodes_installed: true },
  }).map((c) => [c.action, c]));
  assert.equal(cat.krea_vae.present, false);
  assert.equal(cat.krea_vae.available, true);
  assert.equal(cat.krea_model.present, true);
  // The pack has its own test above (the restart state); here just check the row
  // exists at all — it was missing entirely, which is what made Krea invisible.
  assert.ok(cat.krea_nodes);
});
