/**
 * Contract test for the two Sort controls. node --test cannot parse JSX, so the
 * ORDERING LOGIC lives in src/utils/gridSort.js (unit-tested there) and this file
 * greps the JSX for the wiring that logic depends on — the part a rewrite or a
 * "quick tidy" silently drops, leaving a menu that renders but sorts nothing.
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

import { BANK_SORTS } from '../src/utils/gridSort.js'

const read = (rel) => readFileSync(new URL(rel, import.meta.url), 'utf8')
const BANK = read('../src/components/bank/BankWorkspace.jsx')
const WORKSPACE = read('../src/components/dataset/DatasetWorkspace.jsx')

test('the bank Sort menu is built from the registry, not hard-coded options', () => {
  assert.match(BANK, /import \{ bankSortOptions \} from '\.\.\/\.\.\/utils\/gridSort\.js'/)
  assert.match(BANK, /bankSortOptions\(counts\)\.map\(/,
    'the menu must be rendered from bankSortOptions(payload counts)')
  assert.match(BANK, /disabled=\{o\.disabled\}/,
    'an option with no data behind it must actually be disabled')
  // No option may be written by hand any more: a literal <option value="res_desc">
  // would silently escape the enabled/disabled rules.
  for (const s of BANK_SORTS) {
    assert.doesNotMatch(BANK, new RegExp(`<option value="${s.id}"`),
      `${s.id} is hard-coded in JSX instead of coming from BANK_SORTS`)
  }
})

test('the bank sort rides to the server (whole filter) and to fetchAllIds', () => {
  // The sort is a query parameter, so SQL orders the WHOLE selection; the same
  // params object feeds fetchAllIds, so "Select all" / Review walk that order.
  assert.match(BANK, /if \(f\.sort && f\.sort !== 'default'\) params\.sort = f\.sort/)
  assert.match(BANK, /fetchAllIds\(bankId, filterParams\(filter\)\)/)
  // …and the page fetch uses the same filterParams(f).
  assert.match(BANK, /\{ \.\.\.filterParams\(f\), offset: String\(off\)/)
})

test('the dataset grid renders the sorted+filtered list, and sorts last', () => {
  assert.match(WORKSPACE, /sortDatasetImages, \}? ?from '\.\.\/\.\.\/utils\/gridSort'|from '\.\.\/\.\.\/utils\/gridSort'/)
  // Sort wraps the filters — membership stays the filters' business.
  assert.match(WORKSPACE, /const gridImages = sortDatasetImages\(filterImages\(/)
  assert.match(WORKSPACE, /\), gridSort\);/)
  // The very list that was sorted is what the grid (and thus its select-all) gets.
  assert.match(WORKSPACE, /<DatasetGrid images=\{gridImages\}/)
  assert.match(WORKSPACE, /<GridSortSelect value=\{gridSort\}/)
})

test('the persisted sort id is normalised on read (legacy/hand-edited values)', () => {
  assert.match(WORKSPACE, /const GRID_SORT_KEY = 'datasetGridSort'/)
  assert.match(WORKSPACE, /normalizeDatasetSort\(localStorage\.getItem\(GRID_SORT_KEY\)\)/)
})

test('both selects stay inside a 400 px toolbar', () => {
  // A <select> with long option labels stretches its own box; without a bound it
  // pushes the toolbar into a horizontal scroll on a phone.
  assert.match(BANK, /max-w-\[11rem\][^"]*rounded-md border border-border bg-surface[^"]*text-xs text-content"\n?\s*>\n?\s*\{bankSortOptions/)
  assert.match(WORKSPACE, /max-w-\[13rem\]/)
  // The dataset control wraps onto its own line rather than squeezing the chips.
  assert.match(WORKSPACE, /flex flex-wrap items-center gap-x-3 gap-y-1\.5/)
  assert.match(WORKSPACE, /flex shrink-0 items-center gap-1 text-xs/)
})

test('both selects are labelled for screen readers', () => {
  assert.match(BANK, /aria-label="Sort the grid"/)
  assert.match(WORKSPACE, /aria-label="Sort the grid"/)
})
