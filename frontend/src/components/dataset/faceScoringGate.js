// Face-similarity gate, UI side (PURE JS, JSX-free so node --test can import it).
//
// The RULE lives on the server (face_dataset_service.face_scoring_block_reason) and
// arrives in the dataset payload as `face_scoring_blocked`: a sentence explaining
// why the pass is refused, or null. Nothing here knows about subject types — this
// module only decides how to PRESENT a reason it is handed. That is deliberate: the
// same rule then drives the button, the batch endpoint and the two other scoring
// lanes, and a second reason added server-side is honoured here for free.

const SCORE_TITLE = "Scores each image's facial resemblance vs the reference — kept images AND "
  + 'the ones still waiting for a ✓/✕ (deletes nothing)';
const NO_REF_TITLE = 'Set a reference photo first';
// insightface + onnxruntime + antelopev2 are an OPTIONAL extra, absent from
// plenty of installs. Without them the pass can only fail after the click, with
// a toast — so the button says what is missing and where to get it instead.
const NOT_INSTALLED = 'Face scoring is not installed — Setup ▸ Quality tools '
  + '(installs InsightFace + the antelopev2 model)';

/** {disabled, title, blocked, reason, setupRoute} for the 🎭 Analyze faces button.
 *
 *  `capable` is the `face_scoring` capability (CapabilitiesContext). It is
 *  deliberately ranked BELOW the server's own refusal (`blockedReason`, e.g. an
 *  anime dataset) — installing InsightFace would not help there — and ABOVE the
 *  missing-reference hint, which is pointless advice on a machine that cannot
 *  score anything anyway. `capsLoading` keeps the button quiet rather than
 *  flashing "not installed" for the split second before capabilities land. */
export function faceAnalysisState({ blockedReason, hasRef, busy,
                                    capable = true, capsLoading = false } = {}) {
  // A pass that stops running without a word is exactly the failure being fixed —
  // so the reason is the tooltip, and it outranks the missing-reference hint (which
  // would send the user to fix something that would not have helped).
  if (blockedReason) return { disabled: true, title: blockedReason, blocked: true,
                              reason: blockedReason, setupRoute: null };
  if (!capsLoading && !capable) {
    return { disabled: true, title: NOT_INSTALLED, blocked: true,
             reason: NOT_INSTALLED, setupRoute: '#/setup?step=quality' };
  }
  if (!hasRef) return { disabled: true, title: NO_REF_TITLE, blocked: false,
                        reason: null, setupRoute: null };
  return { disabled: !!busy, title: SCORE_TITLE, blocked: false,
           reason: null, setupRoute: null };
}

/** Label for the 🎭 button: names the scope so the pass is not a mystery.
 *  `scope` = the payload's `face_scoring_scope` ({total, unscored}) or null on an
 *  older/absent payload — in which case we fall back to the bare label rather
 *  than inventing a count. */
export function faceAnalysisLabel(scope) {
  const total = Number(scope?.total);
  if (!Number.isFinite(total) || total <= 0) return '🎭 Analyze faces';
  const unscored = Number(scope?.unscored);
  if (Number.isFinite(unscored) && unscored > 0 && unscored < total) {
    return `🎭 Analyze faces (${total} · ${unscored} new)`;
  }
  return `🎭 Analyze faces (${total})`;
}

/** Can 🎯 Auto-triage act on the stored face scores?
 *
 * Scores written BEFORE a dataset was marked anime are kept (nothing is deleted —
 * flip the subject type back and they are all still there), but auto-triage is the
 * one surface that ACTS on them: it batch-flips keep/reject. Replaying a threshold
 * over numbers the scorer could not actually measure is silent damage, so the
 * panel withdraws while the dataset is blocked. */
export function autoTriageAvailable(blockedReason) {
  return !blockedReason;
}
