/* Placement — the layer that lets the ◉ LoRA Canvas be ARRANGED.

   It sits between the automatic layout (utils/lineageGraph.js, which knows
   nothing about the user) and the board geometry (utils/canvasLayout.js, which
   only ever consumes a finished lane size). Pure functions, no JSX and no DOM,
   so the delicate part — what happens to an arrangement when a new run lands —
   is exercised by `node --test` rather than by dragging things in a browser.

   Three rules, and the third is the one that decides whether the canvas ages
   well:

     1. a card with no remembered position takes the automatic one;
     2. a card with one takes it, and the automatic layout does not push it;
     3. a NEW run is placed by the automatic layout in free space, WITHOUT
        moving any card that is already on the board.

   Rule 3 is not free, because the automatic tree is global: it re-centres a
   parent over the span of its children, so one new fork slides every ancestor
   of that fork sideways. Nothing about "the user dragged card #7" tells the
   layout to leave #3 alone.

   So the lane has two states, and only two:

     • UNPINNED (no remembered position at all) — there is no arrangement to
       protect. The automatic tree wins outright, exactly as in slice 1, and a
       new run reshuffles it freely. That is the right answer: the user never
       expressed anything.
     • PINNED (at least one remembered position) — every card of the lane is
       expected to have a row. `pendingPins` below names the cards that don't
       have one yet, at the position they currently occupy, and the page writes
       them. From then on a new run is the only thing that can move: it takes
       its automatic spot, and if that spot is already taken it slides DOWN
       until it is clear.

   The transition is an EVENT, not something inferred: `pinSnapshot` below turns
   one drag into a complete map of the lane — the dragged card at its new spot,
   every other card at the spot it already occupied. Inferring it instead
   ("no row yet, so treat it as an arrival") would have nudged perfectly settled
   cards out of the way of the very first drag.

   ✦ Tidy up deletes the rows, which puts the lane back into the UNPINNED state
   and hands it to the automatic tree. That is the whole escape hatch — no
   special case anywhere. */

import { CARD_W, PAD, V_GAP, edgePath } from './lineageGraph.js';

// Vertical air left between a newly-arrived run and whatever it had to slide
// past. Same rhythm as the automatic tree's sibling gap, so a nudged card does
// not read as belonging to a different board.
export const NEW_NODE_GAP = V_GAP;

// A dragged card is clamped to this corner. Negative coordinates would place a
// card outside its own lane — visually on top of the lane above it, and
// invisible to the fit, which measures the lane from its origin.
const MIN_X = 0;
const MIN_Y = 0;

/** Clamp a dropped position into its lane. */
export function clampNodePosition(x, y) {
  const nx = Number(x);
  const ny = Number(y);
  return {
    x: Math.max(MIN_X, Number.isFinite(nx) ? nx : MIN_X),
    y: Math.max(MIN_Y, Number.isFinite(ny) ? ny : MIN_Y),
  };
}

/** Normalise whatever the API / a drag hands us into a plain {id: {x,y}} map.
 *  Rows with unusable coordinates are DROPPED, not defaulted to (0,0): a card
 *  silently teleported to the corner is worse than a card the automatic layout
 *  still owns. */
export function toOverrideMap(rows) {
  const out = {};
  for (const r of (Array.isArray(rows) ? rows : [])) {
    const id = r?.record_id ?? r?.id;
    // typeof first: Number(null) and Number('') are both 0, which would park a
    // card in the corner instead of leaving it to the automatic layout.
    if (id == null || typeof r.x !== 'number' || typeof r.y !== 'number') continue;
    const x = Number(r.x);
    const y = Number(r.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    out[id] = clampNodePosition(x, y);
  }
  return out;
}

const rectsOverlap = (a, b) => (
  a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h
);

/**
 * Find a free spot for ONE new card, starting from where the automatic layout
 * put it and only ever moving it DOWN.
 *
 * Down and not sideways on purpose: the horizontal axis of a lineage carries
 * meaning (one column = one generation), so a run pushed right would read as a
 * continuation of something it never continued. Vertical space is the only free
 * dimension, and it is exactly what the automatic layout uses for forks.
 *
 * `taken` are the rects already final. Bounded by construction: each step jumps
 * below the lowest blocker, so it terminates after at most `taken.length` moves.
 */
export function freeSpotBelow(rect, taken) {
  let y = rect.y;
  for (let guard = 0; guard <= taken.length; guard += 1) {
    const hits = taken.filter((t) => rectsOverlap({ ...rect, y }, t));
    if (!hits.length) return { x: rect.x, y };
    y = Math.max(...hits.map((t) => t.y + t.h)) + NEW_NODE_GAP;
  }
  return { x: rect.x, y };
}

/**
 * Apply the remembered positions to one dataset's automatic graph.
 *
 * `graph`     — buildLineageGraph() output (nodes, edges, width, height, …).
 * `overrides` — {record_id: {x, y}} for that dataset, lane-local coordinates of
 *               the card's top-left corner.
 *
 * Returns a graph of the same shape, with every node moved to its final spot,
 * its checkpoint pills carried along, its edges re-drawn, and width/height
 * recomputed from the result — a card dragged to the right MUST grow the lane,
 * or the board's fit would crop it and there would be no way back to it.
 *
 * Also returns `pendingPins`: the rows the caller should persist so the lane
 * stays whole (see the header — empty for an unpinned lane).
 */
export function applyPlacement(graph, overrides) {
  const nodes = Array.isArray(graph?.nodes) ? graph.nodes : [];
  const ov = overrides && typeof overrides === 'object' ? overrides : {};
  const pinned = Object.keys(ov).length > 0;
  if (!nodes.length) {
    return { ...(graph || {}), nodes: [], edges: [], width: 0, height: 0, pendingPins: [] };
  }

  // Pass 1 — the cards the user owns. They are final before anything else is
  // considered, which is what "the automatic layout does not push it" means.
  const finalPos = new Map();
  const taken = [];
  const rectOf = (n, p) => ({ x: p.x, y: p.y, w: CARD_W, h: n.cellH });
  for (const n of nodes) {
    const id = n.node.record_id;
    const o = ov[id];
    if (!o) continue;
    const p = clampNodePosition(o.x, o.y);
    finalPos.set(id, p);
    taken.push(rectOf(n, p));
  }

  // Pass 2 — everything else.
  const pendingPins = [];
  const rest = nodes.filter((n) => !finalPos.has(n.node.record_id));
  if (!pinned) {
    // Unpinned lane: the automatic tree owns it, untouched.
    for (const n of rest) finalPos.set(n.node.record_id, { x: n.x, y: n.y });
  } else {
    // Pinned lane: these are runs that arrived after the arrangement was made.
    // Placed top-to-bottom for determinism (two new runs must not depend on the
    // order the API happened to list them in), each into free space.
    const arrivals = [...rest].sort((a, b) => (a.y - b.y) || (a.x - b.x)
      || (a.node.record_id - b.node.record_id));
    for (const n of arrivals) {
      const spot = freeSpotBelow(rectOf(n, { x: Math.max(MIN_X, n.x), y: Math.max(MIN_Y, n.y) }), taken);
      finalPos.set(n.node.record_id, spot);
      taken.push(rectOf(n, spot));
      pendingPins.push({ record_id: n.node.record_id, x: spot.x, y: spot.y });
    }
  }

  // Pass 3 — move each node's whole cell (card + its pills) by its delta, and
  // re-draw the edges from the endpoints the layout handed us.
  const delta = new Map();
  const outNodes = nodes.map((n) => {
    const p = finalPos.get(n.node.record_id);
    const dx = p.x - n.x;
    const dy = p.y - n.y;
    delta.set(n.node.record_id, { dx, dy });
    if (!dx && !dy) return n;
    return {
      ...n,
      x: p.x,
      y: p.y,
      checkpoints: (n.checkpoints || []).map((c) => ({ ...c, x: c.x + dx, y: c.y + dy })),
    };
  });

  const zero = { dx: 0, dy: 0 };
  const outEdges = (graph.edges || []).map((e) => {
    const dp = delta.get(e.parentId) || zero;
    const dc = delta.get(e.childId) || zero;
    if (!dp.dx && !dp.dy && !dc.dx && !dc.dy) return e;
    const x1 = e.x1 + dp.dx;
    const y1 = e.y1 + dp.dy;
    const x2 = e.x2 + dc.dx;
    const y2 = e.y2 + dc.dy;
    return { ...e, x1, y1, x2, y2, d: edgePath(x1, y1, x2, y2) };
  });

  let maxX = 0;
  let maxY = 0;
  for (const n of outNodes) {
    maxX = Math.max(maxX, n.x + CARD_W);
    maxY = Math.max(maxY, n.y + n.cellH);
  }

  return {
    ...graph,
    nodes: outNodes,
    edges: outEdges,
    width: maxX + PAD,
    height: maxY + PAD,
    pendingPins,
  };
}

/**
 * The COMPLETE remembered map for a lane after one card was dropped.
 *
 * Called on every drop, and it is what turns an unpinned lane into a pinned
 * one: the dragged card takes its new spot, every other card is written down
 * exactly where it already sat. Nothing is nudged — the cards that are not
 * being dragged have not arrived, they were already settled, and shoving one
 * aside to make room for a drop the user chose would be the canvas overruling
 * them on their own board.
 *
 * `placed` is the graph AS DISPLAYED (applyPlacement's output), so the snapshot
 * records what the user can see, never the raw automatic tree underneath it.
 * Returns {record_id, x, y} rows, ready to PUT.
 */
export function pinSnapshot(placed, recordId, x, y) {
  const dropped = clampNodePosition(x, y);
  const rows = [];
  for (const n of (placed?.nodes || [])) {
    const id = n.node.record_id;
    rows.push(id === recordId
      ? { record_id: id, x: dropped.x, y: dropped.y }
      : { record_id: id, x: n.x, y: n.y });
  }
  return rows;
}
