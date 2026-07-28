import test from 'node:test';
import assert from 'node:assert/strict';
import {
  FACE_DETECTION_ACTION, FACE_DETECTION_LABEL, faceDetectionInstallState,
} from './faceDetectionInstall.js';

// THE central assertion of this pass: the face-masking surface must be able to
// name AND install the thing it needs, without the user having to know the
// capability is filed under "face scoring" somewhere else.
test('an install-capable state points at the existing face_scoring install action', () => {
  const s = faceDetectionInstallState({ capable: false, capsLoading: false,
                                        python: { ml_supported: true, version: '3.11.9' } });
  assert.equal(s.status, 'installable');
  assert.equal(s.canInstall, true);
  assert.equal(s.action, 'face_scoring');   // stored key — never renamed
  assert.equal(FACE_DETECTION_ACTION, 'face_scoring');
});

test('the visible label names InsightFace, not "face scoring"', () => {
  assert.match(FACE_DETECTION_LABEL, /InsightFace/);
  const s = faceDetectionInstallState({ capable: false, capsLoading: false,
                                        python: { ml_supported: true } });
  assert.match(s.headline, /InsightFace/);
});

// "Option, never imposed": the cost is announced BEFORE the click.
test('the installable state announces download size and duration up front', () => {
  const s = faceDetectionInstallState({ capable: false, capsLoading: false,
                                        python: { ml_supported: true } });
  assert.match(s.detail, /MB/);
  assert.match(s.detail, /minute/i);
});

test('an unsupported Python explains itself instead of offering a doomed install', () => {
  const s = faceDetectionInstallState({
    capable: false, capsLoading: false,
    python: { ml_supported: false, version: '3.14.0', ml_range: '3.10–3.12' } });
  assert.equal(s.status, 'unsupported_python');
  assert.equal(s.canInstall, false);        // no button that can only fail
  assert.match(s.detail, /3\.14\.0/);       // the version it actually runs on
  assert.match(s.detail, /3\.10–3\.12/);    // the range that would work
  assert.match(s.detail, /face_scoring\.python/);  // the documented way out
});

test('capabilities still loading stays quiet — no "not installed" flash', () => {
  const s = faceDetectionInstallState({ capable: false, capsLoading: true, python: {} });
  assert.equal(s.status, 'loading');
  assert.equal(s.canInstall, false);
});

test('an install that already happened reports ready and offers nothing', () => {
  const s = faceDetectionInstallState({ capable: true, capsLoading: false,
                                        python: { ml_supported: true } });
  assert.equal(s.status, 'ready');
  assert.equal(s.canInstall, false);
});

// A missing `python` block (older backend, or caps not yet shaped) must not be
// read as "unsupported" — that would hide the install button for everyone.
test('an absent python probe degrades to installable, not to unsupported', () => {
  const s = faceDetectionInstallState({ capable: false, capsLoading: false });
  assert.equal(s.status, 'installable');
});
