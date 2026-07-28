/* The confirmation text for "remove this run from the graph".

   A destructive action has to say what it takes BEFORE it takes it. Deleting a
   run used to promise "the run entry and its notes"; it now also clears preview
   links, the card's position on the canvas, and the archived source images no
   other run still uses — so the dialog counts each of them out loud.

   Two things it promises will SURVIVE, because they are the ones a user would
   fear losing: the generated images (unlinked from the run, still in the Test
   Studio) and the runs that continued from this one (re-rooted, not deleted).

   Pure string builder, no React — unit-tested with node:test. */

function plural(n, one, many) {
  return `${n} ${n === 1 ? one : many}`;
}

/* Lines describing what disappears. Exported for the panel's inline hint, which
   shows the same truth without opening a dialog. Zero counts are omitted rather
   than printed as "0 notes" — noise that hides the lines that matter. */
export function runDeletionLosses(impact) {
  const n = (k) => {
    const v = Number(impact?.[k]);
    return Number.isFinite(v) && v > 0 ? v : 0;
  };
  const lines = [];
  if (n('notes')) lines.push(plural(n('notes'), 'checkpoint note', 'checkpoint notes'));
  if (n('previews')) lines.push(plural(n('previews'), 'preview link', 'preview links'));
  if (n('canvas_positions')) lines.push('its saved position on the canvas');
  if (n('archived_images_released')) {
    lines.push(`${plural(n('archived_images_released'), 'archived source image',
      'archived source images')} (kept only for this run)`);
  }
  return lines;
}

/* What survives — stated positively so the user is not left guessing. */
export function runDeletionKeeps(impact) {
  const n = (k) => {
    const v = Number(impact?.[k]);
    return Number.isFinite(v) && v > 0 ? v : 0;
  };
  const lines = [];
  const images = n('images_unlinked');
  if (images) {
    lines.push(images === 1
      ? '1 generated image stays in the Test Studio — it only loses the link to this run.'
      : `${images} generated images stay in the Test Studio — they only lose the link to this run.`);
  }
  const children = n('children_detached');
  if (children) {
    lines.push(children === 1
      ? '1 run that continued from it stays in the graph, as its own root.'
      : `${children} runs that continued from it stay in the graph, as their own roots.`);
  }
  return lines;
}

/* The full window.confirm body. `impact` may be null — when the preview call
   fails we still let the user delete, with the honest generic wording rather
   than invented numbers. */
export function runDeletionMessage(recordId, impact) {
  const head = `Remove run #${recordId} from the graph?\n\n`
    + 'Its checkpoints are already gone from disk. No LoRA file is deleted.';
  if (!impact) {
    return `${head}\n\nThis clears the leftover run entry, its notes, its previews `
      + 'and its place on the canvas. Generated images are kept.';
  }
  const losses = runDeletionLosses(impact);
  const keeps = runDeletionKeeps(impact);
  const parts = [head];
  parts.push(losses.length
    ? `This also removes:\n${losses.map((l) => `• ${l}`).join('\n')}`
    : 'Nothing else is attached to it.');
  if (keeps.length) parts.push(keeps.join('\n'));
  return parts.join('\n\n');
}
