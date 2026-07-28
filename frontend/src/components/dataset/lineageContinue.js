/* ▶ "Continue from here" gate for a ◉ Graph checkpoint pill. JSX-free so
   `node --test` exercises the real rule (RunLineageGraph.jsx imports this exact
   function), and so both mounts of the graph share ONE definition of "this save
   can be resumed" instead of drifting apart.

   Two lanes, chosen by the mount:
     - 'cloud' (DEFAULT — the Runs hub): cloud runs only. That hub's Continue
       flow relaunches a cloud run, so a local run has nothing to offer there.
     - 'any' (the dataset ▸ Checkpoints panel): the same gesture, served by the
       panel's LOCAL resume flow, so a local run's save qualifies too.

   In both lanes the pill must be a save that really exists — the graph never
   offers an action the backend would refuse. */

// A cloud run that failed (e.g. 'pod did not become ready in time') can still
// hold a valid harvested save, hence the download check on the non-'done' path.
const TERMINAL_FAILED = ['error', 'error_pod_kept', 'stopped', 'failed'];

/* canContinueFromCheckpoint(node, pill, { continueSource, hasHandler }) -> bool.
   `hasHandler` is false when the mount passed no onContinueCheckpoint (the
   popover then shows Download/Import only). */
export function canContinueFromCheckpoint(node, pill, opts = {}) {
  const { continueSource = 'cloud', hasHandler = true } = opts;
  if (!hasHandler || !node) return false;
  if (node.source === 'cloud') {
    // The Runs hub's rule, unchanged: a terminal cloud run with a run id, and —
    // when it didn't finish cleanly — a pill that is actually downloadable.
    return node.run_id != null
      && (node.status === 'done'
          || (TERMINAL_FAILED.includes(node.status) && !!pill?.download_url));
  }
  // Local run: the lineage records no terminal status for it (only a currently
  // failed local run is flagged), so the SAVE decides — present and downloadable.
  // The panel wires the handler only when nothing is training and the checkpoint
  // selection matches, which is the in-flight gate.
  return continueSource === 'any' && pill?.present !== false && !!pill?.download_url;
}

/* Which checkpoint the ▶ Continue dialog opens ON. A pill click asks for a
   specific step (`requested`); honour it only when it is a REAL save of the run
   being resumed, otherwise keep the historical default — the newest. Never
   invents a step the run doesn't have. `steps` is ascending. */
export function initialResumeStep(requested, steps) {
  const list = Array.isArray(steps) ? steps : [];
  const latest = list.length ? list[list.length - 1] : 0;
  return requested != null && list.includes(requested) ? requested : latest;
}

/* Which lane the ▶ Continue dialog opens on when it offers the choice.
   `where` is the SOURCE run's lane (a local run defaults to Local, a cloud run to
   Cloud) — but a lane the user cannot use is never pre-selected: if the default
   one is unavailable and the other isn't, open on the one that works. Both
   unavailable → keep `where` (the dialog then shows each lane's reason and the
   submit button stays disabled). `lanes` is null when the mount doesn't offer the
   picker, in which case the answer is simply `where`. */
export function resolveInitialLane(where, lanes) {
  const lane = where === 'cloud' ? 'cloud' : 'local';
  if (!lanes) return lane;
  const other = lane === 'cloud' ? 'local' : 'cloud';
  if (lanes[lane]?.available === false && lanes[other]?.available) return other;
  return lane;
}

/* Krea's raw recipe answers to two names ('base' in the run rows, 'raw' in the
   labels): comparing the strings blindly would call one run two runs. */
const sameVariant = (a, b) => {
  const norm = (v) => (String(v || '').trim().toLowerCase() === 'raw' ? 'base'
    : String(v || '').trim().toLowerCase());
  return norm(a) === norm(b);
};

/* WHY a ◉ Graph pill cannot be resumed from the dataset panel — the true reason,
   or null when it can be. The panel's ▶ Continue dialog resumes the ACTIVE
   checkpoint set (the family/base/variant chosen under "Browse results"), so:

     - a pill from ANOTHER run identity is a selection problem — say so;
     - a pill of the SAME identity that the active set simply doesn't hold is
       NOT: that save is not on this machine (a cloud epoch never mirrored
       here). Blaming the selection sent the user to change a setting that was
       already right — the reported "switch the Checkpoints selection" message
       on a run whose family, base and variant all matched.

   `active` = { steps, trainType, variant, base, familyLabel, variantLabel }. */
export function graphContinueRefusal(node, pill, active = {}) {
  const step = pill?.step ?? null;
  const steps = Array.isArray(active.steps) ? active.steps : [];
  if (step == null || steps.includes(step)) return null;
  const here = `${active.familyLabel || 'this family'} · ${active.variantLabel || 'this variant'}`;
  const differs = (node?.train_type && node.train_type !== active.trainType)
    || (node?.variant && !sameVariant(node.variant, active.variant))
    || (node?.base_model != null && (node.base_model || '') !== (active.base || ''));
  if (differs) {
    return `Step ${step} comes from a run trained with a different family, base or variant `
      + `than the checkpoints selected here (${here}) — switch the Checkpoints selection to `
      + 'that run’s family, base and variant to continue it.';
  }
  const highest = steps.length ? Math.max(...steps) : null;
  return `Step ${step} is not among the checkpoints this machine holds for ${here}`
    + (highest ? ` (they stop at step ${highest})` : ' (none are on this machine)')
    + ' — that save lives in its cloud run: continue it from the Runs page.';
}

/* Why the ▶ Continue dialog's submit button is disabled, as text to SHOW — or
   null when it is live. A greyed button that explains nothing is read as a
   broken one ("I click Continue and nothing happens"), so every state that
   disables it owes the user a sentence: a blocked lane has its own reason (it is
   printed with the lane picker, so we don't repeat it), but "this run has no
   checkpoint" printed nothing at all. */
export function submitBlockedReason({ latest, laneBlocked, laneReason, lane } = {}) {
  if (!(latest > 0)) {
    return 'This run has no checkpoint to resume from on this machine — continue it from '
      + 'the Runs page (where its own saves live), or start a fresh run.';
  }
  if (laneBlocked && !laneReason) {
    return `Continuing on the ${lane === 'cloud' ? 'cloud' : 'local'} GPU is unavailable right now.`;
  }
  return null;      // live, or a blocked lane that already states its reason
}

export default canContinueFromCheckpoint;
