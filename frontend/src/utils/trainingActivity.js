/* 🏋️ Wording for the "something is training" indicator in the nav.
   Pure on purpose: node --test cannot parse JSX, so the phrasing — the part
   that is easy to get subtly wrong — lives here and is covered without a
   browser. The component only decides whether to paint a dot. */

export const EMPTY_ACTIVITY = { running: false, local: false, cloud: 0 }

/* Tolerate anything the endpoint may hand back (missing keys, a stale payload,
   a negative count from a bad merge) rather than rendering "undefined training
   runs" next to the icon. */
export function normalizeActivity(raw) {
  if (!raw || typeof raw !== 'object') return EMPTY_ACTIVITY
  const local = Boolean(raw.local)
  const n = Number(raw.cloud)
  const cloud = Number.isFinite(n) && n > 0 ? Math.floor(n) : 0
  // `running` is DERIVED, never trusted: a payload claiming running:true with
  // nothing running would leave the dot lit forever with no way to explain it.
  return { local, cloud, running: local || cloud > 0 }
}

/* The accessible label — also the tooltip. Says WHERE it runs, because that is
   what changes what the user can do next: a local training holds their GPU, a
   cloud one costs money while it lives. */
export function activityLabel(raw) {
  const { local, cloud, running } = normalizeActivity(raw)
  if (!running) return ''
  const parts = []
  if (local) parts.push('1 training running on this machine')
  if (cloud > 0) parts.push(`${cloud} training${cloud > 1 ? 's' : ''} running in the cloud`)
  return parts.join(' · ')
}
