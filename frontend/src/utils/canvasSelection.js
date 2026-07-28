/* Which datasets the ◉ LoRA Canvas is showing — a display preference, not data.

   JSX-free so `node --test` runs it. The rules are small but every one of them
   was a bug waiting to happen:

   • Nothing stored yet = show EVERYTHING. The canvas' promise is "all your
     datasets on one surface"; opening on an empty board and asking the user to
     go find the filter would be the opposite.
   • A stored id whose dataset is gone is dropped silently, never drawn as an
     empty lane and never resurrected if an id is reused.
   • Deselecting every dataset is a legitimate choice and survives a reload — it
     must not be "helpfully" reset to all, or the filter would fight the user.

   ⚠️ The stored ids are DATABASE dataset ids: stable, never renamed. Nothing in
   here may start persisting names instead — those DO change, and a rename would
   silently empty someone's board. */

export const CANVAS_SELECTION_KEY = 'lds.canvasDatasets';

// Database ids are positive integers. Anything else (null, '', a boolean, a
// stray string) is not an id — Number() would turn several of them into 0 and
// quietly add a lane for "dataset 0".
const asIdList = (v) => (Array.isArray(v) ? v : [])
  .map((n) => Number(n))
  .filter((n) => Number.isInteger(n) && n > 0);

/** Read the stored selection. `null` means "never chosen" (≠ an empty array,
 *  which means "the user unticked everything"). A corrupt value reads as null:
 *  a bad localStorage entry must not be able to blank the board forever. */
export function readSelection(store, key = CANVAS_SELECTION_KEY) {
  try {
    const raw = store?.getItem(key);
    if (raw == null) return null;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return null;
    return asIdList(parsed);
  } catch {
    return null;
  }
}

/** Persist a selection. Best-effort: a full or blocked localStorage must never
 *  break the canvas over a display preference. */
export function writeSelection(store, ids, key = CANVAS_SELECTION_KEY) {
  try {
    store?.setItem(key, JSON.stringify(asIdList(ids)));
    return true;
  } catch {
    return false;
  }
}

/**
 * The datasets actually drawn: the stored choice intersected with what exists,
 * in the order the index gave (newest-trained first), or everything when there
 * is no stored choice.
 */
export function resolveSelection(availableIds, stored) {
  const available = asIdList(availableIds);
  if (stored == null) return available;
  const want = new Set(asIdList(stored));
  return available.filter((id) => want.has(id));
}

/** Toggle one dataset in/out, keeping the list in `available` order so the board
 *  never reorders itself just because of the order things were ticked. */
export function toggleSelection(selected, id, availableIds) {
  const available = asIdList(availableIds);
  const cur = new Set(asIdList(selected));
  const n = Number(id);
  if (cur.has(n)) cur.delete(n); else cur.add(n);
  return available.filter((a) => cur.has(a));
}

/** "3 of 7" style summary for the collapsed filter button on a narrow screen,
 *  where the checkbox list itself cannot be on screen permanently. */
export function selectionSummary(selectedCount, totalCount) {
  const sel = Math.max(0, Number(selectedCount) || 0);
  const total = Math.max(0, Number(totalCount) || 0);
  if (!total) return 'No trained datasets';
  if (sel === 0) return `None of ${total}`;
  if (sel === total) return total === 1 ? '1 dataset' : `All ${total} datasets`;
  return `${sel} of ${total}`;
}
