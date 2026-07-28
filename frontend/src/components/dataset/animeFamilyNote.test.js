import test from 'node:test';
import assert from 'node:assert/strict';
import { animeFamilyNote } from './animeFamilyNote.js';

const ON = { subjectType: 'anime', trainType: 'zimage', animaSupported: true };

test('it shows for an anime dataset on a non-Anima family', () => {
  const note = animeFamilyNote(ON);
  assert.ok(note && note.includes('Anima'));
  assert.ok(/anime character/i.test(note));
});

test('it vanishes the moment the family IS Anima (no contradiction)', () => {
  assert.equal(animeFamilyNote({ ...ON, trainType: 'anima' }), null);
});

test('it never shows for a non-anime subject, whatever the family', () => {
  for (const subjectType of ['human', 'animal', 'creature', 'object', 'other']) {
    for (const trainType of ['zimage', 'sdxl', 'krea', 'flux', 'flux2klein', 'anima']) {
      assert.equal(animeFamilyNote({ ...ON, subjectType, trainType }), null,
        `${subjectType}/${trainType}`);
    }
  }
});

test('it stays quiet when Anima cannot run on this machine', () => {
  // Anima is local-only and needs an up-to-date ai-toolkit. Pointing at an option
  // the user cannot take would be worse than saying nothing.
  assert.equal(animeFamilyNote({ ...ON, animaSupported: false }), null);
  // Unknown (base-info not loaded / older server) is treated as "cannot" — the
  // note is a nicety, so silence is the safe default.
  assert.equal(animeFamilyNote({ ...ON, animaSupported: undefined }), null);
  assert.equal(animeFamilyNote({ ...ON, animaSupported: null }), null);
});

test('a missing subject type reads as human (legacy datasets stay silent)', () => {
  assert.equal(animeFamilyNote({ ...ON, subjectType: undefined }), null);
  assert.equal(animeFamilyNote({}), null);
});

test('it is a NOTE: it never claims something is wrong or blocked', () => {
  const note = animeFamilyNote(ON);
  assert.doesNotMatch(note, /⚠|warning|error|must|required|cannot|unsupported/i);
  // ... and it says out loud that nothing is being forced.
  assert.match(note, /optional/i);
});
