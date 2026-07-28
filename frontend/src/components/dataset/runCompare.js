/* Pure helpers for the two-run compare panel (LineageDiffPanel.jsx).

   JSX-free on purpose so `node --test` can exercise the logic directly — the
   panel is a thin renderer over these functions. Everything here is derived from
   the /api/dataset/train/runs/compare payload; nothing fetches, nothing mutates.

   The whole point of this module is that "3 captions changed" becomes readable:
   `captionWordDiff` is what turns two caption strings into the words that were
   actually added and removed, which is the difference between a comparison you
   can act on and a number you have to trust. */

/* Word-level diff of two captions, as a flat list of
   { type: 'same' | 'removed' | 'added', text } segments in reading order.

   A classic LCS over WORDS (captions run to a few dozen words, so the quadratic
   table is nothing) — character-level would shred words into noise, and
   line-level would just restate "it changed". Adjacent segments of the same type
   are merged so the rendered caption reads as prose rather than confetti.

   Either side missing (a run that predates snapshots, an image that had no
   caption) yields a pure add or a pure remove — never an empty result that would
   read as "identical". */
export function captionWordDiff(before, after) {
  const a = splitWords(before);
  const b = splitWords(after);
  if (!a.length && !b.length) return [];
  if (!a.length) return merge([{ type: 'added', text: b.join(' ') }]);
  if (!b.length) return merge([{ type: 'removed', text: a.join(' ') }]);

  // Words are matched on a NORMALISED key (lower-case, outer punctuation
  // stripped) but rendered as written. Without it, appending a clause turns the
  // previous last word into a struck-through/added pair purely because it gained
  // a comma — noise that buries the one word that actually changed. A caption is
  // prose; "red → blue" is the signal, "hair → hair," never is.
  const ka = a.map(wordKey);
  const kb = b.map(wordKey);

  // lcs[i][j] = length of the longest common subsequence of a[i..] and b[j..]
  const lcs = Array.from({ length: a.length + 1 }, () => new Array(b.length + 1).fill(0));
  for (let i = a.length - 1; i >= 0; i -= 1) {
    for (let j = b.length - 1; j >= 0; j -= 1) {
      lcs[i][j] = ka[i] === kb[j]
        ? lcs[i + 1][j + 1] + 1
        : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }
  const out = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (ka[i] === kb[j]) {
      // Show the LATER spelling for a shared word: the panel reads as the caption
      // the run actually trained on, with only the edits marked up.
      out.push({ type: 'same', text: b[j] });
      i += 1; j += 1;
    } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
      out.push({ type: 'removed', text: a[i] });
      i += 1;
    } else {
      out.push({ type: 'added', text: b[j] });
      j += 1;
    }
  }
  while (i < a.length) { out.push({ type: 'removed', text: a[i] }); i += 1; }
  while (j < b.length) { out.push({ type: 'added', text: b[j] }); j += 1; }
  return merge(out);
}

function splitWords(text) {
  return String(text || '').trim().split(/\s+/).filter(Boolean);
}

/* Matching key of one word: lower-cased with leading/trailing punctuation
   stripped. A token made ONLY of punctuation keeps itself, so an em-dash or a
   lone "—" is still a word that can be added or removed. */
function wordKey(word) {
  const stripped = word.toLowerCase().replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');
  return stripped || word.toLowerCase();
}

function merge(segments) {
  const out = [];
  for (const seg of segments) {
    const last = out[out.length - 1];
    if (last && last.type === seg.type) last.text += ` ${seg.text}`;
    else out.push({ ...seg });
  }
  return out;
}

/* The headline of the comparison: one chip per KIND of dataset change, with the
   counts the lists below expand. Empty array = the two runs trained on exactly
   the same images with exactly the same captions, which the panel states in
   words rather than showing four zeroes. */
export function datasetChangeChips(images) {
  if (!images) return [];
  const chips = [];
  // The plural belongs on the NOUN, not on the trailing participle: "5 images
  // added", never "5 image addeds". Both forms are spelled out rather than
  // derived, so a future label with an irregular plural cannot go wrong.
  const push = (key, n, one, many) => {
    if (n > 0) chips.push({ key, count: n, label: `${n} ${n > 1 ? many : one}` });
  };
  push('added', (images.added || []).length + (images.added_withheld || 0),
    'image added', 'images added');
  push('removed', (images.removed || []).length + (images.removed_withheld || 0),
    'image removed', 'images removed');
  push('caption_changed',
    (images.caption_changed || []).length + (images.caption_withheld || 0),
    'caption edited', 'captions edited');
  push('content_changed',
    (images.content_changed || []).length + (images.content_withheld || 0),
    'image re-edited', 'images re-edited');
  return chips;
}

/* True when the payload proves the two runs saw an identical dataset. Anything
   unknown (a run with no snapshot) is NOT identical — the panel must not claim
   sameness it cannot demonstrate. */
export function datasetIsIdentical(data) {
  if (!data || !data.images) return false;
  if ((data.notes || []).length > 0) return false;
  return datasetChangeChips(data.images).length === 0;
}

/* Rows of the environment table, sorted so the DIFFERENCES come first — the
   whole reason the machine is recorded is that a changed ai-toolkit revision
   explains a gap the dataset does not. Stable within each group. */
export function sortedEnvRows(rows) {
  const list = [...(rows || [])];
  list.sort((x, y) => Number(Boolean(y.changed)) - Number(Boolean(x.changed)));
  return list;
}

/* Short human label for one side of the comparison, e.g. "v3 · Marion · #117". */
export function sideLabel(side) {
  if (!side) return '';
  const bits = [];
  if (side.version) bits.push(`v${side.version}`);
  if (side.dataset_name) bits.push(side.dataset_name);
  bits.push(`#${side.record_id}`);
  return bits.join(' · ');
}
