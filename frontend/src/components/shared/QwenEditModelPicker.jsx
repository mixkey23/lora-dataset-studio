import { useEffect, useState } from 'react';
import { useCapabilities } from '../../context/CapabilitiesContext';

const STORAGE_KEY = 'editPage_qwenEditModel_v1';

/**
 * Base diffusion model picker for the Qwen Edit "Generate variations" engine —
 * same shape as Flux2KleinModelPicker, for the same reason: some installs keep
 * several Qwen-Image-Edit-2511 checkpoints side by side (e.g. an SFW build and
 * an uncensored/community fine-tune), and the app can't guess which one a
 * given run should use.
 *
 * Renders nothing when fewer than 2 models are available (no choice to make).
 * The selected filename is persisted to localStorage and reported to the
 * parent via `onChange` so it can be forwarded to the backend at submit time
 * (the wire field is `klein_model` regardless of engine — see
 * face_dataset_service.generate_variations, which takes it as a generic
 * "local engine model override").
 *
 * Model list is sourced from `caps.comfyui.models.qwen_edit` (the
 * capabilities probe scans any 'qwen'+'edit'-named model folder — shared with
 * Qwen Multi-angle, same checkpoint pool) rather than a dedicated endpoint.
 */
export default function QwenEditModelPicker({ onChange }) {
  const { caps } = useCapabilities();
  const models = caps.comfyui.models.qwen_edit || [];
  const [selected, setSelected] = useState(() => {
    try { return localStorage.getItem(STORAGE_KEY) || ''; } catch { return ''; }
  });

  useEffect(() => {
    // Reconcile persisted choice with the current available list.
    const valid = models.includes(selected) ? selected : (models[0] || '');
    if (valid !== selected) setSelected(valid);
    onChange?.(valid || null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models.join('|')]);

  function handleChange(e) {
    const next = e.target.value;
    setSelected(next);
    try { localStorage.setItem(STORAGE_KEY, next); } catch { /* quota / private mode */ }
    onChange?.(next || null);
  }

  if (models.length < 2) return null;

  return (
    <div className="flex flex-col gap-1">
      <label className="text-content-muted text-sm font-medium" htmlFor="qwen-edit-model">
        Base model
      </label>
      <select
        id="qwen-edit-model"
        value={selected}
        onChange={handleChange}
        className="w-full bg-white/[0.03] border border-white/10 rounded-lg px-3 py-2 text-sm text-content focus:outline-none focus:border-primary/60"
      >
        {models.map((m) => (
          <option key={m} value={m} className="bg-surface-overlay">
            {m}
          </option>
        ))}
      </select>
    </div>
  );
}
