// A rented pod bills whether or not the run is getting anywhere. The backend
// reports, on every active run, how long it has said NOTHING (idle_seconds)
// and how long it is allowed to (idle_limit_seconds, 0 = the freeze watchdog
// is off). This turns that pair into the one-line warning the run card shows —
// so a frozen run is visible long before the watchdog acts, and visible at all
// when the user configured the watchdog never to act.
//
// Thresholds differ by phase for the same reason the backend's do: a training
// run writes progress every ~10 s, while booting, uploading and downloading are
// silent by design for long stretches.

const TRAINING_WARN_SECONDS = 10 * 60;
const OTHER_WARN_SECONDS = 45 * 60;
const ACTIVE = new Set(['preparing', 'provisioning', 'uploading', 'training',
  'downloading', 'terminating']);

const minutes = (seconds) => Math.round(seconds / 60);

export function runSilenceWarning(run) {
  if (!run || !ACTIVE.has(run.status)) return null;
  const idle = Number(run.idle_seconds);
  if (!Number.isFinite(idle)) return null;
  const threshold = run.status === 'training' ? TRAINING_WARN_SECONDS : OTHER_WARN_SECONDS;
  if (idle < threshold) return null;
  const limit = Number(run.idle_limit_seconds) || 0;
  // Past the limit the supervisor is already terminating the pod (it ticks every
  // minute) — say that rather than promise a future that has arrived.
  const critical = limit > 0 && idle >= limit;
  const head = `No progress reported for ${minutes(idle)} min — the pod is still being paid for.`;
  let tail;
  if (critical) tail = 'The pod is being terminated automatically.';
  else if (limit > 0) tail = `It is terminated automatically at ${minutes(limit)} min of silence.`;
  else tail = 'Automatic termination is off — stop the run yourself if it is stuck.';
  return { level: critical ? 'critical' : 'warn', text: `${head} ${tail}` };
}

// The Stop button answers with what ACTUALLY happened, never a courtesy ok:
// 'graceful' (a healthy monitor is winding the pod down), 'forced' (the pod was
// terminated on the spot because nothing was left to do it), or a failure that
// must name the instance the user has to destroy by hand.
export function stopOutcomeMessage(res) {
  if (!res || res.ok === false) {
    return {
      level: 'error',
      text: (res && res.error)
        || 'Could not stop the run — check the vast.ai console for a live instance.',
    };
  }
  if (res.mode === 'forced') {
    return {
      level: 'warn',
      text: 'Pod terminated. The run had stopped reporting, so it was shut down '
        + 'directly — checkpoints already downloaded are kept.',
    };
  }
  return { level: 'info', text: 'Stopping the run — the pod is winding down…' };
}
