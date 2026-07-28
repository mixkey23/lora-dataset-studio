import test from 'node:test'
import assert from 'node:assert/strict'
import { normalizeActivity, activityLabel, EMPTY_ACTIVITY } from './trainingActivity.js'

test('nothing running yields no label at all', () => {
  assert.equal(activityLabel({ running: false, local: false, cloud: 0 }), '')
  assert.equal(activityLabel(null), '')
  assert.equal(activityLabel(undefined), '')
})

test('a garbage payload degrades to "nothing running" instead of throwing', () => {
  assert.deepEqual(normalizeActivity(null), EMPTY_ACTIVITY)
  assert.deepEqual(normalizeActivity('boom'), EMPTY_ACTIVITY)
  assert.deepEqual(normalizeActivity({}), EMPTY_ACTIVITY)
})

test('a nonsense cloud count never reaches the label', () => {
  assert.equal(normalizeActivity({ cloud: -3 }).cloud, 0)
  assert.equal(normalizeActivity({ cloud: 'many' }).cloud, 0)
  assert.equal(normalizeActivity({ cloud: 2.7 }).cloud, 2)
})

test('running is derived, never taken from the payload', () => {
  // A server claiming running:true with nothing running would otherwise leave
  // the dot lit forever, with nothing to explain it.
  assert.equal(normalizeActivity({ running: true, local: false, cloud: 0 }).running, false)
  assert.equal(normalizeActivity({ running: false, local: true }).running, true)
  assert.equal(normalizeActivity({ running: false, cloud: 1 }).running, true)
})

test('the label says WHERE it runs', () => {
  assert.equal(activityLabel({ local: true, cloud: 0 }),
    '1 training running on this machine')
  assert.equal(activityLabel({ local: false, cloud: 1 }),
    '1 training running in the cloud')
  assert.equal(activityLabel({ local: false, cloud: 3 }),
    '3 trainings running in the cloud')
})

test('local and cloud at once are both named', () => {
  assert.equal(activityLabel({ local: true, cloud: 2 }),
    '1 training running on this machine · 2 trainings running in the cloud')
})
