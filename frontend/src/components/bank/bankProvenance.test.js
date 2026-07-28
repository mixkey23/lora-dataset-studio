import test from 'node:test'
import assert from 'node:assert/strict'

import {
  DETAIL_CALIBRATION, DETAIL_SPEAK_BELOW, ORIGIN_CHIPS, PROVENANCE_FLAG_LABEL,
  barsSummary, detailSummary, isGeneratorSize, jpegQualitySummary, originHint,
  originLabel, trueFraction,
} from './bankProvenance.js'

// --- effective resolution ---------------------------------------------------
test('a full-detail image is never described as enlarged', () => {
  const got = detailSummary({ detail_ratio: 1, width: 2048, height: 2048 })
  assert.equal(got.soft, false)
  assert.equal(got.real, null)
  assert.match(got.text, /2048 px/)
  assert.doesNotMatch(got.text, /real detail/)
})

test('a measured enlargement is reported in pixels, de-biased', () => {
  // 0.54 measured is what a true 1/4 enlargement reads (see the backend module).
  const got = detailSummary({ detail_ratio: 0.54, width: 2048, height: 1024 })
  assert.equal(got.soft, true)
  assert.equal(got.stored, 2048)
  // ~1/4 of 2048 = ~512, NOT 0.54*2048 = 1106. Reporting the raw ratio as pixels
  // would understate every enlargement by a factor of two.
  assert.ok(got.real >= 450 && got.real <= 600, `got ${got.real}`)
  assert.match(got.text, /2048 px stored/)
  assert.match(got.text, /real detail/)
})

test('the calibration is monotone across its whole range', () => {
  let previous = Infinity
  for (let m = 1; m >= 0.4; m -= 0.02) {
    const f = trueFraction(m)
    assert.ok(f <= previous + 1e-9, `not monotone at ${m}`)
    previous = f
  }
})

test('the calibration anchors map to themselves', () => {
  for (const [measured, fraction] of DETAIL_CALIBRATION) {
    assert.ok(Math.abs(trueFraction(measured) - fraction) < 1e-6)
  }
})

test('readings too close to a sharp photo stay silent', () => {
  // One genuinely full-resolution image in ten measures ~0.78, so anything above
  // that must NOT be presented as reduced detail.
  const got = detailSummary({ detail_ratio: DETAIL_SPEAK_BELOW + 0.01,
                              width: 1600, height: 1600 })
  assert.equal(got.soft, false)
})

test('an unmeasured image says nothing at all', () => {
  assert.equal(detailSummary({ width: 1024, height: 1024 }), null)
  assert.equal(detailSummary({ detail_ratio: null, width: 1024, height: 1024 }), null)
  assert.equal(detailSummary({ detail_ratio: 0.5 }), null)
})

// --- origin -----------------------------------------------------------------
test('origin has exactly three states and unknown is one of them', () => {
  assert.deepEqual(ORIGIN_CHIPS.map((c) => c.id), ['ai', 'camera', 'unknown'])
})

test('unknown is explained as an absence, never as "not AI"', () => {
  const got = originLabel({ origin: 'unknown' })
  assert.equal(got.state, 'unknown')
  assert.match(got.detail, /NOT evidence either way/)
  assert.doesNotMatch(got.detail, /not AI|real photo|genuine/i)
})

test('ai and camera name the evidence that proved them', () => {
  assert.match(originLabel({ origin: 'ai', origin_evidence: 'png-prompt' }).detail,
               /ComfyUI/)
  assert.match(originLabel({ origin: 'camera', origin_evidence: 'exif-camera' }).detail,
               /Camera EXIF/)
})

test('an unscanned image has no origin label', () => {
  assert.equal(originLabel({}), null)
  assert.equal(originLabel({ origin: null }), null)
})

// --- the bucket-size presumption -------------------------------------------
test('generator bucket sizes are recognised', () => {
  assert.equal(isGeneratorSize(1024, 1024), true)
  assert.equal(isGeneratorSize(832, 1216), true)
  assert.equal(isGeneratorSize(4032, 3024), false)  // a phone photo
  assert.equal(isGeneratorSize(0, 0), false)
})

test('the size hint is only ever offered when metadata said nothing', () => {
  assert.ok(originHint({ origin: 'unknown', width: 832, height: 1216 }))
  // Never over a state we actually proved...
  assert.equal(originHint({ origin: 'ai', width: 832, height: 1216 }), null)
  assert.equal(originHint({ origin: 'camera', width: 832, height: 1216 }), null)
  // ...and never for a size that proves nothing.
  assert.equal(originHint({ origin: 'unknown', width: 1913, height: 2551 }), null)
})

test('the size hint calls itself a hint', () => {
  const got = originHint({ origin: 'unknown', width: 1024, height: 1024 })
  assert.match(got, /hint, not a finding/)
})

// --- small facts ------------------------------------------------------------
test('black bars are reported as a percentage and compared to the threshold', () => {
  assert.equal(barsSummary({ bars_ratio: 0.25 }, 0.04).text, '25% black bars')
  assert.equal(barsSummary({ bars_ratio: 0.25 }, 0.04).over, true)
  assert.equal(barsSummary({ bars_ratio: 0.02 }, 0.04).over, false)
  assert.equal(barsSummary({ bars_ratio: 0 }, 0.04), null)
  assert.equal(barsSummary({}, 0.04), null)
})

test('jpeg quality is a fact, not a verdict', () => {
  assert.equal(jpegQualitySummary({ jpeg_quality: 71.9 }).text, 'JPEG q72')
  assert.equal(jpegQualitySummary({ jpeg_quality: null }), null)
})

test('the two new flags have labels', () => {
  assert.deepEqual(Object.keys(PROVENANCE_FLAG_LABEL).sort(), ['bars', 'soft_detail'])
  for (const label of Object.values(PROVENANCE_FLAG_LABEL)) {
    assert.ok(label.length > 2)
  }
})
