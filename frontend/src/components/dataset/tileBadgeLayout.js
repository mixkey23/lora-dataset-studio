/* Where the provenance badge ("generated · face · Nano Banana Pro") sits on a
   dataset tile — and the ONE place that number is written down.

   PURE JS (no JSX) so `node --test` can import and exercise it directly.

   THE BUG THIS ENCODES
   --------------------
   The badge was pinned top-LEFT and the action buttons (🔄 ✏️ ⇆ ✂ 🗑) top-RIGHT,
   with nothing arbitrating between them. Measured in a browser, the button row
   is ~147 px wide and the longest badge ~172 px ("generated · face · Nano Banana
   Pro"): they need ~336 px of tile to coexist. A 400-px phone at tile size M
   gives 188 px, so the badge ran 113 px UNDER the buttons and the engine name —
   the whole point of the badge — became unreadable. It only shows on touch
   devices: `@media (hover: hover)` hides the buttons at rest, so a mouse user
   never sees the collision.

   WHY A CONTAINER QUERY AND NOT A VIEWPORT MEDIA QUERY
   ----------------------------------------------------
   Tile width is a product of BOTH the S/M/L setting (2/2/1 columns on a phone,
   6/4/3 on a desktop) AND the window width. A viewport media query gets it
   backwards at both ends: size L on a 400-px phone is a 384-px tile with room
   to spare, size S on a 1400-px desktop is a 228-px tile with none. Only the
   tile's own inline size answers the question, so `.dataset-grid-item` is a
   size-query container (the repo already does this in WatermarkReviewLightbox)
   and the badge reads that container, not the window. The alternative — keying
   off the `tileSize` prop in JS — would have needed a second, viewport-based
   rule bolted on to cover the window axis, i.e. the same wrong tool with extra
   steps, plus a re-render on every resize. CSS does it with zero JS.

   The JS below is the SPEC of that CSS rule; tileBadgeLayout.test.js asserts the
   stylesheet still matches it, so the two cannot drift apart. */

/** Tile inline size (px) from which the badge can share the top row with the
 *  action buttons: longest badge (~172) + gap (~8) + buttons (~147) + margins
 *  (~8) ≈ 335, rounded up to a round 21rem. Below it the badge drops to the
 *  bottom-RIGHT — bottom-LEFT is taken by the bulk-selection checkbox. */
export const TILE_BADGE_TOP_MIN_WIDTH_REM = 21;

/** Width (px) the top placement reserves for the button row, so that even in
 *  the top placement the badge is truncated rather than slid under the buttons. */
export const TILE_BADGE_ACTIONS_RESERVE_REM = 10.25;

/** Width (px) the bottom placement reserves for the selection checkbox. */
export const TILE_BADGE_CHECKBOX_RESERVE_REM = 2;

/**
 * @param {number} tileWidthPx inline size of the tile
 * @param {number} [rootFontSizePx] to honour a user's larger browser font
 * @returns {'top-left' | 'bottom-right'}
 */
export function provenanceBadgePlacement(tileWidthPx, rootFontSizePx = 16) {
  if (!Number.isFinite(tileWidthPx)) return 'bottom-right';
  return tileWidthPx >= TILE_BADGE_TOP_MIN_WIDTH_REM * rootFontSizePx
    ? 'top-left'
    : 'bottom-right';
}

/* Class names shared by the component and the stylesheet. Positioning lives
   ENTIRELY in index.css (no Tailwind position utilities on these elements) so
   the container query has nothing to fight. */
export const TILE_BADGE_STACK_CLASS = 'dataset-tile-badges';
export const PROVENANCE_BADGE_CLASS = 'dataset-tile-badge dataset-tile-badge--provenance';
export const FACE_BADGE_CLASS = 'dataset-tile-badge dataset-tile-badge--face';
export const WATERMARK_BADGE_CLASS = 'dataset-tile-badge dataset-tile-badge--watermark';
