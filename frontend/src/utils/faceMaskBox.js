/* Face -> head box arithmetic for the concept face-masking preview (issue #15,
   reported by shivdbz2010 on GitHub).

   THIS IS A MIRROR. The authoritative copy is `dilate_box` in
   backend/infer/face_mask_infer.py, which is what actually paints the mask the
   trainer is given. Both are asserted against the same numbers (this file's test
   and backend/tests/test_concept_face_masking.py) so the preview cannot drift into
   showing one thing while the export does another.

   Why the arithmetic lives on BOTH sides instead of the server returning grown
   boxes: the server returns the RAW detected boxes once, and the browser grows
   them. Moving the expand slider then redraws instantly, with no second InsightFace
   pass — the knob is only worth exposing if you can see it act.

   Pure JS, no JSX, so `node --test` can execute it. */

// The detected box runs eyes-to-chin. Growing it around its centre turns it into a
// head box; the upward bias buys the hair, which sits above that centre. Same two
// numbers as the Python side — change one, change both.
export const SHIFT_UP = 0.10;

/** Grow a normalised [x1,y1,x2,y2] face box into the head box the mask will cover.
 *  Coordinates may fall outside 0-1: a face at the edge of the frame legitimately
 *  grows past it, and the caller clamps for DISPLAY only (clampBox) so the drawn
 *  rectangle never reports coverage the mask does not actually have. */
export function dilateBox(box, expand) {
  const [x1, y1, x2, y2] = box;
  const cx = (x1 + x2) / 2;
  const cy = (y1 + y2) / 2 - (y2 - y1) * SHIFT_UP;
  const hw = ((x2 - x1) * expand) / 2;
  const hh = ((y2 - y1) * expand) / 2;
  return [cx - hw, cy - hh, cx + hw, cy + hh];
}

/** The visible part of a box, for positioning an overlay inside the image. */
export function clampBox([x1, y1, x2, y2]) {
  return [Math.max(0, x1), Math.max(0, y1), Math.min(1, x2), Math.min(1, y2)];
}

/** Fraction of the frame the grown boxes cover (clamped, so off-frame growth is
 *  not counted). Above ~0.5 the export refuses to mask that image: ai-toolkit
 *  renormalises the mask to mean 1.0, so masking most of the frame multiplies the
 *  loss on what little is left — a silent learning-rate bump for that sample.
 *  Overlapping faces are counted once each, so this OVER-estimates rather than
 *  under-estimates; erring toward "we would skip this one" is the safe direction. */
export function coverageFraction(boxes, expand) {
  let total = 0;
  for (const b of boxes || []) {
    const [x1, y1, x2, y2] = clampBox(dilateBox(b, expand));
    total += Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  }
  return Math.min(1, total);
}

// Kept in sync with _MAX_COVERAGE in backend/infer/face_mask_infer.py.
export const MAX_COVERAGE = 0.5;

/** CSS percentages for an absolutely-positioned overlay, same idiom as the
 *  watermark region editor (normalised coords -> left/top/width/height in %). */
export function boxStyle(box, expand) {
  const [x1, y1, x2, y2] = clampBox(dilateBox(box, expand));
  // Rounded: binary floating point turns 0.3 into "30.000000000000004%", which is
  // both noise in the DOM and a needless source of test churn.
  const pct = (n) => `${Math.round(n * 1e6) / 1e4}%`;
  return {
    left: pct(x1),
    top: pct(y1),
    width: pct(x2 - x1),
    height: pct(y2 - y1),
  };
}
