/* Memory-saving levers (Advanced options → Memory saving) — the pure bits, kept
   out of TrainingPanel.jsx so `node --test` can exercise the copy directly (the
   test runner does not parse JSX).

   Community request: GitHub issue #14 (bobba84) — "Disable Quant / LowVRAM for
   Z-Image Training. I have a 5090."

   Two rules this module exists to keep honest:

   1. ADVISING IS NOT DECIDING. The detected VRAM only chooses a SENTENCE. It
      never writes a setting, never blocks a launch, and a machine that cannot be
      probed (no NVIDIA card, nvidia-smi missing, a laptop that only drives cloud
      runs) gets a generic line rather than a guess.
   2. THE FAILURE MODE IS SLOWNESS, NOT A CRASH. On Windows/WDDM there is no
      clean out-of-memory error when a run no longer fits: the driver pages to
      system RAM and training creeps for hours. A vague "you may run out of
      memory" would send people hunting for an error that never arrives, so the
      "too small" and "unknown" branches both name the symptom explicitly. */

export const MEMORY_KEYS = ['quantize', 'quantize_te', 'low_vram'];

export const MEMORY_LABELS = {
  quantize: 'Quantise base model',
  quantize_te: 'Quantise text encoder',
  low_vram: 'Low-VRAM streaming',
};

/* "NVIDIA GeForce RTX 5090 · 31.4 GB", or whichever half we actually know. */
export function memoryCardLabel(advice) {
  const a = advice || {};
  return [a.gpu, a.vram_gb ? `${a.vram_gb} GB` : null].filter(Boolean).join(' · ');
}

/* The How: sentence under the checkboxes, indexed on the real card. */
export function memoryAdviceText(advice) {
  const a = advice || {};
  const need = a.unquantised_vram_gb;
  const card = memoryCardLabel(a);
  if (a.verdict === 'can_disable' && card) {
    return `${card} detected — this family needs roughly ${need} GB without them, `
      + 'so you can switch them off for better precision and a much faster run.';
  }
  if (a.verdict === 'keep_on' && card) {
    return `${card} detected — this family needs roughly ${need} GB without them. `
      + 'Leave them on: with them off the run does not fail cleanly, it slows to a '
      + 'crawl for hours while the driver pages to system RAM.';
  }
  return 'Card not detected — keep the defaults unless you know your VRAM. This '
    + `family needs roughly ${need || 24} GB with the savers off; below that a run `
    + 'does not fail cleanly, it slows to a crawl while the driver pages to system RAM.';
}

/* The one-line state next to the "Memory saving" label. A mixed state must NOT
   read as "on": two of the three switched off is exactly when a user needs to see
   that they have left the calibrated recipe. */
export function memoryStateLabel(effective) {
  const on = MEMORY_KEYS.filter((k) => (effective || {})[k]).length;
  if (on === MEMORY_KEYS.length) return 'on — fits a 24 GB card';
  if (on === 0) return 'all off — full precision, needs a big card';
  return `partly off (${MEMORY_KEYS.length - on} of ${MEMORY_KEYS.length})`;
}

/* What a checkbox change should send. Returning to the family's calibrated
   default sends 'auto' so the key is REMOVED — a dataset toggled off and back on
   is then byte-identical to one that was never touched, which is what makes the
   "defaults never changed" promise verifiable rather than merely intended. */
export function memoryPatchFor(key, checked, defaults) {
  const dflt = Boolean((defaults || {})[key]);
  return { [key]: checked === dflt ? 'auto' : checked };
}

/* Has the user overridden anything (→ show "Reset to default")? A null/undefined
   stored value means "Auto"; `false` is an override, not an absence. */
export function memoryIsOverridden(stored) {
  const s = stored || {};
  return MEMORY_KEYS.some((k) => s[k] === true || s[k] === false);
}
