/* The provenance badge ("generated · face · Nano Banana Pro") used to be pinned
   top-LEFT while the action buttons sat top-RIGHT, so on a narrow tile the
   engine name ran under the icons and became unreadable — reported from a
   tablet, where the buttons are always visible (no hover pointer).

   Two halves, because neither catches the other:
   - the pure placement rule, exercised directly;
   - a CONTRACT grep over index.css and the JSX, because the fix is CSS and a
     rewrite of either file would silently undo it (a class is not behaviour,
     nothing throws). node --test cannot parse JSX, hence the grep. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import {
  FACE_BADGE_CLASS,
  PROVENANCE_BADGE_CLASS,
  TILE_BADGE_ACTIONS_RESERVE_REM,
  TILE_BADGE_CHECKBOX_RESERVE_REM,
  TILE_BADGE_STACK_CLASS,
  TILE_BADGE_TOP_MIN_WIDTH_REM,
  WATERMARK_BADGE_CLASS,
  provenanceBadgePlacement,
} from './tileBadgeLayout.js';

const css = fs.readFileSync(new URL('../../index.css', import.meta.url), 'utf8');
const jsx = fs.readFileSync(new URL('./DatasetGridItem.jsx', import.meta.url), 'utf8');

test('the badge only keeps the top row when the tile can hold it next to the buttons', () => {
  const min = TILE_BADGE_TOP_MIN_WIDTH_REM * 16;
  assert.equal(provenanceBadgePlacement(min), 'top-left');
  assert.equal(provenanceBadgePlacement(min + 200), 'top-left');
  assert.equal(provenanceBadgePlacement(min - 1), 'bottom-right');
  // The three widths that actually occur on a 400-px phone: S and M are two
  // columns (~188 px), L is one (~384 px). Measured in a browser.
  assert.equal(provenanceBadgePlacement(188), 'bottom-right');
  assert.equal(provenanceBadgePlacement(384), 'top-left');
  // ...and on a ~820-px tablet: S ~195, M ~263, L ~398.
  assert.equal(provenanceBadgePlacement(195), 'bottom-right');
  assert.equal(provenanceBadgePlacement(263), 'bottom-right');
  assert.equal(provenanceBadgePlacement(398), 'top-left');
});

test('a larger browser font raises the threshold instead of ignoring it', () => {
  // Every length in the rule is in rem, so a user at 20px root gets the bottom
  // placement on a tile that would have kept the top row at 16px.
  assert.equal(provenanceBadgePlacement(350, 16), 'top-left');
  assert.equal(provenanceBadgePlacement(350, 20), 'bottom-right');
});

test('a missing width falls back to the placement that can never collide', () => {
  assert.equal(provenanceBadgePlacement(undefined), 'bottom-right');
  assert.equal(provenanceBadgePlacement(NaN), 'bottom-right');
});

test('the stylesheet asks the TILE, not the window, and at the documented size', () => {
  // A viewport media query is the wrong tool: tile width is a product of the
  // S/M/L setting AND the window. If this ever becomes @media, the bug is back.
  assert.match(css, /\.dataset-grid-item\s*\{[^}]*container-type:\s*inline-size/);
  assert.match(css, new RegExp(`@container \\(min-width: ${TILE_BADGE_TOP_MIN_WIDTH_REM}rem\\)`));
  // The top placement reserves the button row's width, so even above the
  // threshold the badge is truncated rather than slid under the icons.
  assert.match(css, new RegExp(`max-width:\\s*calc\\(100% - ${TILE_BADGE_ACTIONS_RESERVE_REM}rem\\)`));
  // The bottom placement reserves the bulk-selection checkbox at bottom-LEFT.
  assert.match(css, new RegExp(`max-width:\\s*calc\\(100% - ${TILE_BADGE_CHECKBOX_RESERVE_REM}rem\\)`));
});

test('the badges are laid out by the stylesheet alone, with no Tailwind rival', () => {
  for (const cls of [TILE_BADGE_STACK_CLASS, ...PROVENANCE_BADGE_CLASS.split(' '),
    ...FACE_BADGE_CLASS.split(' '), ...WATERMARK_BADGE_CLASS.split(' ')]) {
    assert.ok(css.includes(`.${cls}`), `index.css must style .${cls}`);
    assert.ok(jsx.includes(cls) || jsx.includes('BADGE_CLASS'), `${cls} must reach the tile`);
  }
  // The old hard-coded corners must be gone, or they would win over the
  // container query depending on stylesheet order.
  assert.doesNotMatch(jsx, /absolute top-1 left-1/);
  assert.doesNotMatch(jsx, /absolute top-6 left-1/);
  assert.doesNotMatch(jsx, /absolute bottom-1 right-1/);
  // The selection checkbox stays bottom-LEFT — that is WHY the badge drops to
  // the bottom-RIGHT rather than the bottom-left.
  assert.match(jsx, /absolute bottom-1 left-1/);
});

test('the full provenance label survives the visual truncation', () => {
  // Clamped to two lines on a narrow tile, so hover and screen readers must
  // still get the whole thing — engine name included.
  assert.match(jsx, /title=\{provenanceTitle\}\s+aria-label=\{provenanceTitle\}/);
  assert.match(jsx, /made with \$\{engineLabel\}/);
});
