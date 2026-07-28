import assert from 'node:assert/strict';
import test from 'node:test';
import { buildLineageGraph, CARD_W, PAD } from './lineageGraph.js';
import {
  NEW_NODE_GAP, applyPlacement, clampNodePosition, freeSpotBelow, pinSnapshot,
  toOverrideMap,
} from './canvasPlacement.js';

/* The placement layer is tested against the REAL automatic layout, not against
   a hand-written fake of it. Rule 3 ("a new run moves nothing") only means
   something if the thing it defends against actually happens — so the first
   test below proves that the automatic tree really does shove existing cards
   around when a fork arrives. Every rule-3 assertion after it rests on that. */

// A lineage tree the way the API sends one. `links` is [childId, parentId].
const tree = (ids, links, checkpoints = {}) => ({
  root_id: ids[0],
  current_id: null,
  nodes: ids.map((id) => ({
    record_id: id,
    parent_record_id: (links.find(([c]) => c === id) || [])[1] ?? null,
    created_at: `2026-07-2${id}T00:00:00`,
    train_type: 'zimage',
    checkpoints: (checkpoints[id] || []).map((step) => ({ step, present: true })),
  })),
  edges: links.map(([c, p]) => ({ parent: p, child: c })),
});

const posOf = (graph) => Object.fromEntries(
  graph.nodes.map((n) => [n.node.record_id, { x: n.x, y: n.y }]));

// 1 ─┬─ 2 ─── 3   (a trunk with one fork off the root)
//    └─ 4
const BEFORE = tree([1, 2, 3, 4], [[2, 1], [3, 2], [4, 1]], { 1: [500, 1000], 2: [500] });
// …and the same lineage after a SECOND fork off #2 finished training.
const AFTER = tree([1, 2, 3, 4, 5], [[2, 1], [3, 2], [4, 1], [5, 2]], { 1: [500, 1000], 2: [500] });

// ---- the premise ----------------------------------------------------------

test('PREMISE: the automatic tree really does move existing cards when a run arrives', () => {
  const before = posOf(buildLineageGraph(BEFORE));
  const after = posOf(buildLineageGraph(AFTER));
  const moved = Object.keys(before).filter((id) => before[id].y !== after[id].y);
  assert.ok(moved.length > 0,
    'if this ever passes with an empty list, rule 3 is testing nothing');
});

// ---- rule 1: no remembered position -> the automatic one ------------------

test('rule 1 — with nothing remembered, every card keeps its automatic position', () => {
  const g = buildLineageGraph(BEFORE);
  const placed = applyPlacement(g, {});
  assert.deepEqual(posOf(placed), posOf(g));
  assert.equal(placed.width, g.width);
  assert.equal(placed.height, g.height);
});

test('rule 1 — a null/garbage override map behaves like an empty one', () => {
  const g = buildLineageGraph(BEFORE);
  for (const bad of [null, undefined, 'nope', 42]) {
    assert.deepEqual(posOf(applyPlacement(g, bad)), posOf(g), `override ${bad}`);
  }
});

// ---- rule 2: a remembered position wins ------------------------------------

test('rule 2 — a dragged card sits exactly where it was dropped', () => {
  const g = buildLineageGraph(BEFORE);
  const placed = applyPlacement(g, { 3: { x: 1234, y: 777 } });
  const p = posOf(placed);
  assert.deepEqual(p[3], { x: 1234, y: 777 });
});

test('rule 2 — the automatic layout does not push a dragged card back', () => {
  // #3 is parked exactly on top of where #1 sits automatically. The automatic
  // layout has an opinion about that spot; the user's wins.
  const g = buildLineageGraph(BEFORE);
  const auto = posOf(g);
  const placed = applyPlacement(g, { 3: { x: auto[1].x, y: auto[1].y } });
  assert.deepEqual(posOf(placed)[3], auto[1]);
});

test('rule 2 — a card dropped past the lane origin is clamped, never negative', () => {
  const g = buildLineageGraph(BEFORE);
  const placed = applyPlacement(g, { 2: { x: -400, y: -90 } });
  assert.deepEqual(posOf(placed)[2], { x: 0, y: 0 });
});

// ---- rule 3: a new run moves nothing ---------------------------------------

test('rule 3 — a new run moves NO card that was already on an arranged board', () => {
  // Exactly the page's sequence: open the board, drag #4, store the snapshot,
  // then reload with a run that did not exist before.
  const g = buildLineageGraph(BEFORE);
  const stored = toOverrideMap(pinSnapshot(applyPlacement(g, {}), 4, 900, 40));
  const arranged = applyPlacement(g, stored);

  const after = applyPlacement(buildLineageGraph(AFTER), stored);
  const beforePos = posOf(arranged);
  const afterPos = posOf(after);
  for (const id of [1, 2, 3, 4]) {
    assert.deepEqual(afterPos[id], beforePos[id], `card #${id} moved`);
  }
  assert.ok(afterPos[5], 'the new run is on the board');
});

test('rule 3 — the new run lands in free space, overlapping nothing', () => {
  // #4 parked well clear of the tree, so any overlap found below is the new
  // run's fault and nobody else's.
  const g = buildLineageGraph(BEFORE);
  const stored = toOverrideMap(pinSnapshot(applyPlacement(g, {}), 4, 1400, 900));
  const after = applyPlacement(buildLineageGraph(AFTER), stored);

  const rects = after.nodes.map((n) => ({ id: n.node.record_id, x: n.x, y: n.y, w: CARD_W, h: n.cellH }));
  for (const a of rects) {
    for (const b of rects) {
      if (a.id >= b.id) continue;
      const hit = a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
      assert.ok(!hit, `#${a.id} and #${b.id} overlap`);
    }
  }
});

test('rule 2 beats rule 3 — the board does not police an overlap the user chose', () => {
  // Dropping a card on top of another is a legitimate arrangement (stacking two
  // dead branches out of the way). Only ARRIVALS are slid aside; a deliberate
  // drop is never "corrected".
  const g = buildLineageGraph(BEFORE);
  const auto = posOf(g);
  const stored = toOverrideMap(pinSnapshot(applyPlacement(g, {}), 4, auto[3].x, auto[3].y));
  const placed = posOf(applyPlacement(g, stored));
  assert.deepEqual(placed[4], auto[3]);
  assert.deepEqual(placed[3], auto[3]);
});

test('rule 3 — an UNPINNED lane is left to the automatic tree (nothing to protect)', () => {
  // No arrangement was ever made, so a new run reshuffling the tree is correct.
  const g = buildLineageGraph(AFTER);
  assert.deepEqual(posOf(applyPlacement(g, {})), posOf(g));
  assert.deepEqual(applyPlacement(g, {}).pendingPins, []);
});

test('rule 3 — the new run is reported as a pin, so it survives the next reload', () => {
  const stored = { 1: { x: 0, y: 0 }, 2: { x: 400, y: 0 }, 3: { x: 800, y: 0 }, 4: { x: 400, y: 300 } };
  const placed = applyPlacement(buildLineageGraph(AFTER), stored);
  assert.deepEqual(placed.pendingPins.map((p) => p.record_id), [5]);
  const pin = placed.pendingPins[0];
  assert.deepEqual(posOf(placed)[5], { x: pin.x, y: pin.y });
});

test('rule 3 — two runs arriving at once are placed deterministically', () => {
  const stored = { 1: { x: 0, y: 0 }, 2: { x: 400, y: 0 } };
  const two = tree([1, 2, 3, 4, 5], [[2, 1], [3, 2], [4, 1], [5, 2]], {});
  const a = applyPlacement(buildLineageGraph(two), stored);
  const b = applyPlacement(buildLineageGraph(two), stored);
  assert.deepEqual(posOf(a), posOf(b));
  assert.deepEqual(a.pendingPins, b.pendingPins);
});

// ---- pinning ---------------------------------------------------------------

test('the first drag pins the whole lane, at the positions the cards already had', () => {
  const g = buildLineageGraph(BEFORE);
  const auto = posOf(g);
  const rows = pinSnapshot(applyPlacement(g, {}), 4, 900, 40);
  assert.deepEqual(rows.map((r) => r.record_id).sort(), [1, 2, 3, 4]);
  const map = toOverrideMap(rows);
  assert.deepEqual(map[4], { x: 900, y: 40 }, 'the dragged card takes its new spot');
  for (const id of [1, 2, 3]) {
    assert.deepEqual(map[id], auto[id],
      `#${id} must be pinned where it already was — a drop never shoves a settled card`);
  }
  // …and re-placing with that map is a no-op for everyone but the dragged card.
  const after = posOf(applyPlacement(g, map));
  for (const id of [1, 2, 3]) assert.deepEqual(after[id], auto[id]);
  assert.deepEqual(after[4], { x: 900, y: 40 });
});

test('a second drag keeps the earlier arrangement instead of re-reading the tree', () => {
  const g = buildLineageGraph(BEFORE);
  const first = toOverrideMap(pinSnapshot(applyPlacement(g, {}), 4, 900, 40));
  const second = toOverrideMap(pinSnapshot(applyPlacement(g, first), 2, 120, 600));
  assert.deepEqual(second[4], { x: 900, y: 40 }, 'the first move survives the second');
  assert.deepEqual(second[2], { x: 120, y: 600 });
});

test('pinSnapshot clamps the drop like everything else', () => {
  const rows = pinSnapshot(applyPlacement(buildLineageGraph(BEFORE), {}), 1, -50, -50);
  assert.deepEqual(toOverrideMap(rows)[1], { x: 0, y: 0 });
});

test('a fully pinned lane asks for no further writes', () => {
  const g = buildLineageGraph(BEFORE);
  const stored = posOf(g);
  assert.deepEqual(applyPlacement(g, stored).pendingPins, []);
});

// ---- ✦ Tidy up -------------------------------------------------------------

test('✦ Tidy up — clearing the remembered positions restores the automatic tree', () => {
  const g = buildLineageGraph(AFTER);
  const messy = applyPlacement(g, { 1: { x: 2000, y: 900 }, 5: { x: 0, y: 1500 } });
  assert.notDeepEqual(posOf(messy), posOf(g));
  assert.deepEqual(posOf(applyPlacement(g, {})), posOf(g));
});

// ---- geometry the board depends on ----------------------------------------

test('a card dragged out to the right GROWS the lane, so the fit never crops it', () => {
  const g = buildLineageGraph(BEFORE);
  const placed = applyPlacement(g, { 3: { x: 3000, y: 2000 } });
  assert.equal(placed.width, 3000 + CARD_W + PAD);
  assert.ok(placed.height >= 2000 + PAD);
});

test("a moved card takes its checkpoint pills with it", () => {
  const g = buildLineageGraph(BEFORE);
  const autoNode = g.nodes.find((n) => n.node.record_id === 1);
  assert.ok(autoNode.checkpoints.length, 'precondition: #1 has pills');
  const offsets = autoNode.checkpoints.map((c) => ({ dx: c.x - autoNode.x, dy: c.y - autoNode.y }));
  const placed = applyPlacement(g, { 1: { x: 640, y: 480 } });
  const moved = placed.nodes.find((n) => n.node.record_id === 1);
  moved.checkpoints.forEach((c, i) => {
    assert.equal(c.x - moved.x, offsets[i].dx);
    assert.equal(c.y - moved.y, offsets[i].dy);
  });
});

test('an edge is re-drawn from the moved endpoints, not left hanging', () => {
  const g = buildLineageGraph(BEFORE);
  const edge = g.edges.find((e) => e.parentId === 1 && e.childId === 2);
  assert.ok(edge, 'precondition: 1 → 2 exists');
  const child = g.nodes.find((n) => n.node.record_id === 2);
  const placed = applyPlacement(g, { 2: { x: child.x + 100, y: child.y + 60 } });
  const out = placed.edges.find((e) => e.parentId === 1 && e.childId === 2);
  assert.equal(out.x1, edge.x1, "the parent did not move, so the edge's start must not");
  assert.equal(out.y1, edge.y1);
  assert.equal(out.x2, edge.x2 + 100);
  assert.equal(out.y2, edge.y2 + 60);
  assert.notEqual(out.d, edge.d);
  assert.ok(out.d.startsWith(`M${out.x1},${out.y1} `));
});

test('an empty graph places nothing and asks for nothing', () => {
  const placed = applyPlacement(buildLineageGraph({ nodes: [], edges: [] }), { 7: { x: 1, y: 2 } });
  assert.deepEqual(placed.nodes, []);
  assert.deepEqual(placed.pendingPins, []);
  assert.equal(placed.width, 0);
});

// ---- helpers ---------------------------------------------------------------

test('freeSpotBelow only ever moves DOWN, and stops as soon as it is clear', () => {
  const taken = [{ x: 0, y: 0, w: 100, h: 50 }, { x: 0, y: 60, w: 100, h: 50 }];
  const spot = freeSpotBelow({ x: 0, y: 10, w: 100, h: 50 }, taken);
  assert.equal(spot.x, 0, 'the horizontal axis carries the generation — never touched');
  assert.equal(spot.y, 110 + NEW_NODE_GAP);
});

test('freeSpotBelow leaves an already-free rect exactly where it is', () => {
  const taken = [{ x: 0, y: 0, w: 100, h: 50 }];
  assert.deepEqual(freeSpotBelow({ x: 400, y: 10, w: 100, h: 50 }, taken), { x: 400, y: 10 });
});

test('clampNodePosition keeps a card inside its lane and survives nonsense', () => {
  assert.deepEqual(clampNodePosition(12, 34), { x: 12, y: 34 });
  assert.deepEqual(clampNodePosition(-5, -5), { x: 0, y: 0 });
  assert.deepEqual(clampNodePosition(NaN, 'x'), { x: 0, y: 0 });
  assert.deepEqual(clampNodePosition(Infinity, 3), { x: 0, y: 3 });
});

test('toOverrideMap drops rows it cannot trust rather than parking them at (0,0)', () => {
  assert.deepEqual(toOverrideMap([
    { record_id: 1, x: 10, y: 20 },
    { record_id: 2, x: null, y: 5 },
    { record_id: 3, x: 'abc', y: 5 },
    { x: 4, y: 4 },
  ]), { 1: { x: 10, y: 20 } });
  assert.deepEqual(toOverrideMap(null), {});
});
