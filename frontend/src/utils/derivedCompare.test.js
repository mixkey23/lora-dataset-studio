import test from 'node:test';
import assert from 'node:assert/strict';

import {
  DERIVED_COMPARISONS,
  describeDerivedComparison,
} from './derivedCompare.js';

const original = { id: 1, filename: 'orig.png', status: 'keep' };
const improved = {
  id: 2, filename: 'improved.png', status: 'pending',
  derivation_kind: 'klein_image_improve', parent_image_id: 1,
};

test('a plain image offers no comparison at all (unchanged lightbox)', () => {
  assert.equal(describeDerivedComparison({ id: 9, filename: 'a.png' }, [original]), null);
  assert.equal(describeDerivedComparison(null, []), null);
  // The rescue SOURCE is a parent, never a child: nothing to compare it against.
  assert.equal(
    describeDerivedComparison({ id: 3, derivation_kind: 'small_image_source' }, []),
    null,
  );
});

test('an improve candidate whose original is still there compares against it', () => {
  const c = describeDerivedComparison(improved, [original, improved]);
  assert.equal(c.available, true);
  assert.equal(c.parent, original);
  assert.equal(c.beforeLabel, 'Original');
  assert.equal(c.afterLabel, 'Improved');
  assert.equal(c.reason, '');
});

test('the same gesture serves the small-image rescue candidate', () => {
  const source = { id: 5, filename: 'small.png', derivation_kind: 'small_image_source' };
  const candidate = {
    id: 6, filename: 'rescued.png',
    derivation_kind: 'klein_small_image', parent_image_id: 5,
  };
  const c = describeDerivedComparison(candidate, [source, candidate]);
  assert.equal(c.available, true);
  assert.equal(c.parent, source);
  assert.equal(c.beforeLabel, 'Original (small)');
  assert.equal(c.afterLabel, 'Klein rescue');
});

test('a missing original says why instead of offering a dead control', () => {
  // Parent row deleted/purged from the dataset.
  const gone = describeDerivedComparison(improved, [improved]);
  assert.equal(gone.available, false);
  assert.equal(gone.parent, null);
  assert.match(gone.reason, /no longer in this dataset/i);

  // Legacy row with no link recorded at all.
  const unlinked = describeDerivedComparison({ ...improved, parent_image_id: null }, []);
  assert.equal(unlinked.available, false);
  assert.match(unlinked.reason, /no original recorded/i);

  // Parent row still listed, but its file is gone (nothing to render).
  const fileless = describeDerivedComparison(improved, [{ ...original, filename: null }, improved]);
  assert.equal(fileless.available, false);
  assert.match(fileless.reason, /file is missing/i);
});

test('every declared derivation names both sides and its own action', () => {
  for (const [kind, spec] of Object.entries(DERIVED_COMPARISONS)) {
    assert.ok(spec.beforeLabel.trim(), `${kind}: empty beforeLabel`);
    assert.ok(spec.afterLabel.trim(), `${kind}: empty afterLabel`);
    assert.notEqual(spec.beforeLabel, spec.afterLabel);
  }
});
