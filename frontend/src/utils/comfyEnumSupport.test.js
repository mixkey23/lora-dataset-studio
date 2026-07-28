import test from 'node:test'
import assert from 'node:assert/strict'
import {
  comfyEnumUnavailableReason, enumGapPacks, uniqueEnumGaps,
} from './comfyEnumSupport.js'

const BETA57 = {
  node_id: '77', class_type: 'KSampler', input: 'scheduler', value: 'beta57',
  pack: 'RES4LYF', url: 'https://github.com/ClownsharkBatwing/RES4LYF',
}

test('a capable install says nothing at all', () => {
  assert.equal(comfyEnumUnavailableReason([]), null)
  assert.equal(comfyEnumUnavailableReason(undefined), null)
  // Unreachable ComfyUI: the server fails open and sends an empty list. Silence
  // is the correct output — we could not verify, so we claim nothing.
  assert.equal(comfyEnumUnavailableReason(null), null)
})

test('an attributable value points at the pack, not at a ComfyUI update', () => {
  const reason = comfyEnumUnavailableReason([BETA57])
  assert.match(reason, /scheduler "beta57"/)
  assert.match(reason, /RES4LYF node pack/)
  assert.match(reason, /restart it/)
  // core ComfyUI has never shipped beta57, so "update ComfyUI" would be a dead
  // end — the exact wrong advice this branch exists to avoid.
  assert.doesNotMatch(reason, /update ComfyUI/)
})

test('an unattributable value falls back to the generic fix, with no invented version', () => {
  const reason = comfyEnumUnavailableReason([{ ...BETA57, value: 'zzz', pack: null, url: null }])
  assert.match(reason, /scheduler "zzz"/)
  assert.match(reason, /update ComfyUI and its node packs/)
  assert.doesNotMatch(reason, /\d+\.\d+/)          // no fabricated version number
})

test('the same value on several nodes is one problem, not three', () => {
  const items = [BETA57, { ...BETA57, node_id: '88' }, { ...BETA57, node_id: '99' }]
  assert.equal(uniqueEnumGaps(items).length, 1)
  assert.equal(comfyEnumUnavailableReason(items).match(/beta57/g).length, 1)
})

test('several distinct gaps are all named, each pack listed once', () => {
  const items = [BETA57, { ...BETA57, node_id: '78', input: 'sampler_name', value: 'res_2s' }]
  const reason = comfyEnumUnavailableReason(items)
  assert.match(reason, /scheduler "beta57"/)
  assert.match(reason, /sampler_name "res_2s"/)
  assert.deepEqual(enumGapPacks(items), ['RES4LYF'])
})
