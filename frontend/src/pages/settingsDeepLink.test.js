import test from 'node:test'
import assert from 'node:assert/strict'
import { shouldScrollToSection } from './settingsDeepLink.js'

const base = { hasSection: true, hasFocus: false, loading: false, panelTop: 900, viewportHeight: 800 }

test('a section deep link below the fold scrolls to the panel', () => {
  // The reported case: "Settings › Image engines →" on a phone, where the rail
  // stacks above the panel and the section lands off-screen.
  assert.equal(shouldScrollToSection(base), true)
})

test('a panel already on screen never jumps', () => {
  // Desktop: the grid puts the panel beside the rail, already at the top.
  assert.equal(shouldScrollToSection({ ...base, panelTop: 120, viewportHeight: 800 }), false)
})

test('a panel scrolled just off the top still gets pulled back', () => {
  assert.equal(shouldScrollToSection({ ...base, panelTop: -300 }), true)
})

test('a ?focus= link keeps the scroll for its own field', () => {
  // Two scrolls would fight, and the exact field is the better target.
  assert.equal(shouldScrollToSection({ ...base, hasFocus: true }), false)
})

test('bare /settings does not move the page', () => {
  assert.equal(shouldScrollToSection({ ...base, hasSection: false }), false)
})

test('nothing moves while the settings are still loading', () => {
  // The panel is not rendered yet, so its position is meaningless.
  assert.equal(shouldScrollToSection({ ...base, loading: true }), false)
})

test('missing measurements are treated as "do not move"', () => {
  assert.equal(shouldScrollToSection({ ...base, panelTop: undefined }), false)
  assert.equal(shouldScrollToSection({ ...base, viewportHeight: undefined }), false)
  assert.equal(shouldScrollToSection(), false)
})
