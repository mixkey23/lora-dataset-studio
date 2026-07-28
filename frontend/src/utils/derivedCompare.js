/**
 * Derived image → the original it came from.
 *
 * Two flows produce a CANDIDATE that sits next to an image already in the
 * dataset instead of replacing it: the manual "Klein upscale & improve"
 * (`klein_image_improve`) and the automatic rescue of small scraped images
 * (`klein_small_image`). Both record `parent_image_id` + `derivation_kind` at
 * generation time, and `dataset_payload` publishes both columns — but nothing
 * on screen used them, so judging a candidate meant memorising the original and
 * going back and forth in the grid.
 *
 * `small_image_source` is deliberately absent: it marks a PARENT, not a
 * candidate, so there is nothing above it to compare against.
 *
 * Pure JS on purpose (node --test cannot parse JSX): the lightbox renders what
 * this decides, and the decision itself is unit-tested.
 */

export const DERIVED_COMPARISONS = {
  klein_image_improve: {
    beforeLabel: 'Original',
    afterLabel: 'Improved',
  },
  klein_small_image: {
    beforeLabel: 'Original (small)',
    afterLabel: 'Klein rescue',
  },
};

const unavailable = (spec, reason) => ({ ...spec, parent: null, available: false, reason });

/**
 * @param img     the image currently inspected
 * @param images  the dataset's image rows (the payload list)
 * @returns null when `img` is not a candidate at all — the caller must then
 *          behave exactly as before. Otherwise a descriptor:
 *          `{ beforeLabel, afterLabel, parent, available, reason }`, with
 *          `available: false` + a plain-English `reason` when the original is
 *          gone. Never a half-usable object: `available` alone decides.
 */
export function describeDerivedComparison(img, images = []) {
  const spec = DERIVED_COMPARISONS[img?.derivation_kind];
  if (!spec) return null;

  if (!img.parent_image_id) {
    // Legacy rows written before the link was recorded, and archives restored
    // without their source. Nothing is broken — there is just no original.
    return unavailable(spec, 'No original recorded for this image — nothing to compare.');
  }
  const rows = Array.isArray(images) ? images : [];
  const parent = rows.find((row) => row && row.id === img.parent_image_id) || null;
  if (!parent) {
    return unavailable(spec, 'The original is no longer in this dataset — nothing to compare.');
  }
  if (!parent.filename) {
    return unavailable(spec, 'The original file is missing — nothing to compare.');
  }
  return { ...spec, parent, available: true, reason: '' };
}
