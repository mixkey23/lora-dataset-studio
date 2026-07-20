/** Small bubble anchored on a generated tile: pick a camera angle (azimuth ×
 *  elevation × distance) and rotate that tile's angle with Qwen Multi-angle
 *  (Qwen-Image-Edit-2511 + the community multi-angle LoRA). The source pixels
 *  are never touched — a new derived candidate is generated for review, same
 *  shape as the Klein "Improve" action. Presentational only: the parent wires
 *  onSubmit (which calls multiangleImage) and onClose. Mirrors
 *  PromptEditPopover's structure/positioning. */
import { useState } from 'react';

// Fixed vocabulary from the multi-angle LoRA's model card — byte-exact with
// qwen_multiangle_helper.QWEN_MA_AZIMUTHS/ELEVATIONS/DISTANCES server-side.
const AZIMUTHS = [
  { value: 'front view', label: 'Front' },
  { value: 'front-right quarter view', label: 'Front-right 3/4' },
  { value: 'right side view', label: 'Right profile' },
  { value: 'back-right quarter view', label: 'Back-right 3/4' },
  { value: 'back view', label: 'Back' },
  { value: 'back-left quarter view', label: 'Back-left 3/4' },
  { value: 'left side view', label: 'Left profile' },
  { value: 'front-left quarter view', label: 'Front-left 3/4' },
];
const ELEVATIONS = [
  { value: 'eye-level shot', label: 'Eye level' },
  { value: 'high-angle shot', label: 'High angle' },
  { value: 'low-angle shot', label: 'Low angle' },
  { value: 'elevated shot', label: 'Elevated' },
];
const DISTANCES = [
  { value: 'close-up', label: 'Close-up' },
  { value: 'medium shot', label: 'Medium shot' },
  { value: 'wide shot', label: 'Wide shot' },
];

// One-click combos that set all three selects at once — the common asks.
const PRESETS = [
  { label: 'Front', azimuth: 'front view', elevation: 'eye-level shot', distance: 'medium shot' },
  { label: '3/4 left', azimuth: 'front-left quarter view', elevation: 'eye-level shot', distance: 'medium shot' },
  { label: 'Profile right', azimuth: 'right side view', elevation: 'eye-level shot', distance: 'medium shot' },
  { label: 'Back', azimuth: 'back view', elevation: 'eye-level shot', distance: 'medium shot' },
  { label: 'Top-down', azimuth: 'front view', elevation: 'high-angle shot', distance: 'wide shot' },
  { label: 'Low angle', azimuth: 'front view', elevation: 'low-angle shot', distance: 'close-up' },
];

export default function QwenAnglePopover({ onSubmit, onClose }) {
  const [azimuth, setAzimuth] = useState(AZIMUTHS[0].value);
  const [elevation, setElevation] = useState(ELEVATIONS[0].value);
  const [distance, setDistance] = useState(DISTANCES[1].value);

  const submit = () => { onSubmit({ azimuth, elevation, distance }); onClose(); };

  return (
    // Backdrop closes on outside click; stopPropagation on the bubble keeps clicks
    // inside from bubbling to the tile (mirrors PromptEditPopover).
    <div className="absolute inset-0 z-30 flex items-center justify-center bg-black/50 p-2"
      onClick={(e) => { e.stopPropagation(); onClose(); }}>
      <div className="w-full max-w-[16rem] rounded-lg border border-border bg-surface p-2 shadow-xl flex flex-col gap-2"
        onClick={(e) => e.stopPropagation()}>
        <span className="text-[0.625rem] uppercase text-content-muted">🎥 Rotate camera angle</span>
        <div className="flex flex-wrap gap-1">
          {PRESETS.map((p) => (
            <button key={p.label} type="button"
              onClick={() => { setAzimuth(p.azimuth); setElevation(p.elevation); setDistance(p.distance); }}
              className="px-1.5 py-0.5 rounded text-[10px] bg-app/60 border border-border text-content-muted hover:text-content">
              {p.label}
            </button>
          ))}
        </div>
        <label className="flex flex-col gap-0.5 text-[10px] text-content-muted">
          Azimuth
          <select value={azimuth} onChange={(e) => setAzimuth(e.target.value)}
            className="text-[11px] bg-app/60 border border-border rounded p-1 text-content">
            {AZIMUTHS.map((a) => <option key={a.value} value={a.value}>{a.label}</option>)}
          </select>
        </label>
        <label className="flex flex-col gap-0.5 text-[10px] text-content-muted">
          Elevation
          <select value={elevation} onChange={(e) => setElevation(e.target.value)}
            className="text-[11px] bg-app/60 border border-border rounded p-1 text-content">
            {ELEVATIONS.map((el) => <option key={el.value} value={el.value}>{el.label}</option>)}
          </select>
        </label>
        <label className="flex flex-col gap-0.5 text-[10px] text-content-muted">
          Distance
          <select value={distance} onChange={(e) => setDistance(e.target.value)}
            className="text-[11px] bg-app/60 border border-border rounded p-1 text-content">
            {DISTANCES.map((d) => <option key={d.value} value={d.value}>{d.label}</option>)}
          </select>
        </label>
        <div className="text-[9px] text-content-subtle font-mono truncate"
          title={`<sks> ${azimuth} ${elevation} ${distance}`}>
          {`<sks> ${azimuth} ${elevation} ${distance}`}
        </div>
        <div className="flex gap-1.5 justify-end">
          <button type="button" onClick={onClose}
            className="px-2 py-1 rounded text-[11px] bg-surface border border-border text-content-muted">
            Cancel
          </button>
          <button type="button" onClick={submit}
            className="px-3 py-1 rounded text-[11px] bg-gradient-primary text-white font-semibold">
            Generate
          </button>
        </div>
      </div>
    </div>
  );
}
