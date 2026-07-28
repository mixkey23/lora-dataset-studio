/**
 * Grid ORDERING — the companion of `gridStatusFilter.js` / `tagFilter.js`, which
 * only ever narrow a list. Asked for by nofaceman (Discord): both the bank and a
 * dataset already MEASURE things (aesthetic rating, sharpness, face similarity)
 * and let you filter on them, but nothing could put the best — or the worst —
 * in front of your eyes first, which is what makes a review fast.
 *
 * Two surfaces, two mechanics, ONE contract:
 *  • Bank — the grid is paginated over thousands of rows, so the order is a SQL
 *    `sort=` parameter (see image_bank_service._sort_order). This file only owns
 *    the MENU: which entries exist and which ones are honestly usable right now.
 *  • Dataset — the workspace already holds every image of the dataset in memory
 *    (dataset_payload ships them all, and both existing grid filters are
 *    client-side), so sorting here is a pure array transform. Sending it to SQL
 *    would mean a refetch per click for no gain and a second, divergent order.
 *
 * The contract both share:
 *  • A value the matching pass never produced (NULL / undefined) SINKS TO THE END
 *    in BOTH directions. "Worst first" must open on the worst MEASURED image, not
 *    on the pile nobody measured — that is the same decision the resolution sort
 *    already took ("Unscanned images sink to the end").
 *  • An entry whose data does not exist yet is OFFERED BUT DISABLED, with the
 *    reason in its own label — same precedent as the 👥 Group by person button
 *    ("Install the Quality tools (Setup) to sort by person"). A sort that
 *    silently reorders nothing reads as a broken app.
 *  • Sorting NEVER changes membership: it composes with every filter, and the
 *    filtered list keeps exactly the same images.
 *
 * `id`s are stored (bank query string, dataset localStorage) — never rename one
 * without an alias, per the repo rule on stored identifiers.
 */

/** Bank sort entries, in menu order. `needs` names the payload count that must
 *  be > 0 for the entry to do anything; `missing` is the reason shown when it is
 *  0 (appended to the label, because a disabled <option>'s title is invisible on
 *  most platforms). */
export const BANK_SORTS = [
  { id: 'default', label: 'Default',
    title: 'The bank\'s own order (or the worst-first order of the active flag chip)' },
  { id: 'res_desc', label: 'Resolution ↓', needs: 'scanned', missing: 'run 🔎 Scan quality',
    title: 'Biggest first, in megapixels (width × height)' },
  { id: 'res_asc', label: 'Resolution ↑', needs: 'scanned', missing: 'run 🔎 Scan quality',
    title: 'Smallest first — surfaces the thumbnails hiding in a big dump' },
  { id: 'aesthetic_desc', label: 'Aesthetic ↓', needs: 'scored', missing: 'run ✨ Score',
    title: 'Best-looking first — the keepers, from the ✨ Score pass' },
  { id: 'aesthetic_asc', label: 'Aesthetic ↑', needs: 'scored', missing: 'run ✨ Score',
    title: 'Worst-looking first — the fastest way to prune, from the ✨ Score pass' },
  { id: 'sharp_desc', label: 'Sharpness ↓', needs: 'scanned', missing: 'run 🔎 Scan quality',
    title: 'Sharpest first, from the 🔎 Scan quality pass' },
  { id: 'sharp_asc', label: 'Sharpness ↑', needs: 'scanned', missing: 'run 🔎 Scan quality',
    title: 'Blurriest first — the misses come to you, from the 🔎 Scan quality pass' },
];

export const DEFAULT_BANK_SORT = 'default';

/**
 * The bank's Sort menu for a given payload `counts` ({scanned, scored, …}).
 * An entry is disabled ONLY when we positively know its pass reached 0 images:
 * while the payload is still loading (`counts` absent) nothing is greyed out,
 * because "I don't know yet" must never be shown as "you can't".
 */
export function bankSortOptions(counts) {
  return BANK_SORTS.map((s) => {
    const known = counts && typeof counts[s.needs] === 'number';
    const empty = Boolean(s.needs && known && counts[s.needs] <= 0);
    return {
      id: s.id,
      label: empty ? `${s.label} — ${s.missing} first` : s.label,
      title: empty ? `Nothing measured yet — ${s.missing} to sort by this.` : s.title,
      disabled: empty,
    };
  });
}

/* -------------------------------------------------------------------------- */

/** Dataset sort entries. Only face similarity is offered: it is the one quantity
 *  a dataset row actually carries per image (`face_score`, the ArcFace cosine vs
 *  the reference). Aesthetics/sharpness are BANK columns — a dataset image has
 *  never been through those passes, so offering them here would be a menu entry
 *  that can only ever be greyed out. */
export const DATASET_SORTS = [
  { id: 'default', label: 'Newest first',
    title: 'The order images were added — most recent first' },
  { id: 'face_desc', label: 'Face similarity ↓',
    title: 'Closest to your reference photo first — from 🎭 Analyze faces' },
  { id: 'face_asc', label: 'Face similarity ↑',
    title: 'Least like your reference first — the ones to cut, from 🎭 Analyze faces' },
];

export const DEFAULT_DATASET_SORT = 'default';

const DATASET_IDS = new Set(DATASET_SORTS.map((s) => s.id));

/** Normalise anything (stale localStorage, undefined…) to a usable sort id. */
export function normalizeDatasetSort(value) {
  return (typeof value === 'string' && DATASET_IDS.has(value))
    ? value : DEFAULT_DATASET_SORT;
}

const faceScore = (img) => {
  const v = img?.face_score;
  return (typeof v === 'number' && Number.isFinite(v)) ? v : null;
};

const DATASET_SORT_SPECS = {
  face_desc: { value: faceScore, direction: 'desc' },
  face_asc: { value: faceScore, direction: 'asc' },
};

/**
 * Reorder the dataset grid. Returns the SAME array reference for 'default' and
 * for an unknown id (cheap no-op, like filterImagesByStatus) — the payload
 * already arrives newest-first. Never adds or drops an image.
 * Unscored rows (no face_score: the pass never ran, or the face was not scorable)
 * go last in both directions; ties fall back to the default newest-first order,
 * so the result is deterministic.
 */
export function sortDatasetImages(images, sortId) {
  const list = images || [];
  const spec = DATASET_SORT_SPECS[normalizeDatasetSort(sortId)];
  if (!spec || list.length < 2) return list;
  const dir = spec.direction === 'asc' ? 1 : -1;
  return [...list].sort((a, b) => {
    const va = spec.value(a);
    const vb = spec.value(b);
    if ((va === null) !== (vb === null)) return va === null ? 1 : -1;
    if (va !== null && va !== vb) return (va - vb) * dir;
    return (b.id || 0) - (a.id || 0);
  });
}

/** The dataset's Sort menu. Disabled — with the reason — while no image of the
 *  set carries a face score, so "sort by similarity" can never look broken. */
export function datasetSortOptions(images) {
  const anyScored = (images || []).some((i) => faceScore(i) !== null);
  return DATASET_SORTS.map((s) => {
    const empty = s.id !== DEFAULT_DATASET_SORT && !anyScored;
    return {
      id: s.id,
      label: empty ? `${s.label} — run 🎭 Analyze faces first` : s.label,
      title: empty
        ? 'No image has a face score yet — run 🎭 Analyze faces (Curation) to sort by this.'
        : s.title,
      disabled: empty,
    };
  });
}
