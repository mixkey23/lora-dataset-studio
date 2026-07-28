/* Wording and arithmetic for the concept face-mask preview's progress.

   PURE (no React, no fetch) so the honest-message logic is testable on its own —
   node --test cannot parse JSX, so anything worth asserting lives here.

   The point of naming phases: the per-image counter alone LIES about the wait.
   InsightFace imports onnxruntime and prepares antelopev2 before image 1, which
   is tens of seconds, and on a fresh install it first downloads ~350 MB. A bar
   sitting at 0/40 through all of that is read as "it crashed" at exactly the
   moment nothing at all was wrong. So each stage says what it is, and only the
   stages that really cost are named. */

export const FACE_MASK_PHASES = ['starting', 'downloading', 'loading', 'detecting'];

const PHASE_LABELS = {
  starting: 'Starting the face detector…',
  // First run only, and worth its own sentence: several hundred megabytes over
  // the network is not a slow detection, and telling the user it is would earn
  // exactly the "is this broken?" it is meant to answer.
  downloading: 'Downloading the face-detection model (~350 MB, first run only)…',
  loading: 'Loading the face-detection model…',
  detecting: 'Analyzing images…',
};

/** The one line under the button. Never a bare spinner: it always names either a
 *  stage, a count, or a failure. */
export function previewStatusLabel(job) {
  if (!job) return '';
  if (job.error) return job.error;
  if (job.finished) return 'Done.';
  if (job.phase === 'detecting' && job.total > 0) {
    return `Analyzing image ${Math.min(job.done + 1, job.total)} of ${job.total}…`;
  }
  return PHASE_LABELS[job.phase] || 'Working…';
}

/** {done, total} once there is a real count to show, else null = indeterminate.
 *  Returning null rather than 0 matters: a determinate bar pinned at 0% for the
 *  whole model load is the misleading state this replaces. */
export function previewProgressValue(job) {
  if (!job || job.error || job.finished) return null;
  if (job.phase !== 'detecting' || !(job.total > 0)) return null;
  return { done: Math.max(0, Math.min(job.done || 0, job.total)), total: job.total };
}

/** Bar width, 0-100. Only meaningful when previewProgressValue is non-null. */
export function previewPercent(job) {
  const v = previewProgressValue(job);
  return v ? Math.round((v.done / v.total) * 100) : 0;
}

/** True while a pass is in flight — drives both the disabled button and the poll. */
export function previewRunning(job) {
  return Boolean(job) && !job.finished && !job.error;
}

/** What went wrong, or ''. A finished job with an error is the failure case that
 *  used to look like an endless wait. */
export function previewError(job) {
  return (job && job.error) || '';
}
