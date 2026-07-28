/* What a press on a run CARD means — decided in one place, because the board
 * overloads that one gesture three ways and two of them are invisible in a
 * screenshot.
 *
 * A card can be dragged to a new position ("Drag a run to move it"), and a drop
 * lands ON the card it moved, so the browser fires a click straight after the
 * gesture. Without a guard, every rearrangement of the board would also open a
 * panel — and the panel is a bottom sheet on a phone, so the user would end each
 * drag looking at something they never asked for. That is the whole reason this
 * function exists rather than a `&&` inside a handler: `node --test` cannot
 * parse the canvas JSX, and "a drag does not open the panel" must be a test, not
 * a promise in a comment.
 *
 * Returns:
 *   'ignored' — the press was the tail of a drag: swallow it, change nothing;
 *   'compare' — ⇧ Shift-click, the two-run compare selection;
 *   'open'    — a plain click: open this run's gallery (its images by step, its
 *               notes, its settings).
 */
export function cardClickAction({ dragged = false, shiftKey = false } = {}) {
  if (dragged) return 'ignored';
  return shiftKey ? 'compare' : 'open';
}

/** The target the board hands the gallery panel for a run card. `kind: 'run'` is
 *  explicit rather than inferred from a missing step, so a malformed pill target
 *  can never be mistaken for a whole run. */
export function runGalleryTarget(node) {
  if (!node || node.record_id == null) return null;
  return { kind: 'run', recordId: node.record_id, node };
}
