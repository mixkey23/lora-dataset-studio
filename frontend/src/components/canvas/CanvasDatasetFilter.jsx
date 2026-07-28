import { useEffect, useState } from 'react';
import { selectionSummary } from '../../utils/canvasSelection';

/* Which datasets sit on the board.

   The canvas' whole promise is "all your datasets at once", so this is a
   SUBTRACTIVE control: everything is on the board until you take something off.

   It folds. On a phone a permanent checkbox list would eat the screen the board
   needs, so below `sm` it opens collapsed behind a button that says what is
   currently shown ("3 of 7") — the state is never hidden behind a mystery icon.
   On a wide screen it opens expanded, because there it costs one row. */

const FAMILY_LABEL = {
  zimage: 'Z-Image', krea: 'Krea 2', sdxl: 'SDXL',
  flux: 'FLUX.1', flux2klein: 'FLUX.2 Klein', anima: 'Anima',
};
const familyLabel = (f) => FAMILY_LABEL[f] || f;

export default function CanvasDatasetFilter({ datasets, selected, onToggle, onAll, onNone }) {
  const wide = typeof window !== 'undefined' && window.matchMedia
    ? window.matchMedia('(min-width: 640px)').matches
    : true;
  const [open, setOpen] = useState(wide);
  // A phone rotated to landscape (or a resized window) should not be stuck with
  // the phone's fold — follow the breakpoint until the user decides for himself.
  const [userSet, setUserSet] = useState(false);
  useEffect(() => {
    if (userSet || typeof window === 'undefined' || !window.matchMedia) return undefined;
    const mq = window.matchMedia('(min-width: 640px)');
    const apply = () => setOpen(mq.matches);
    mq.addEventListener?.('change', apply);
    return () => mq.removeEventListener?.('change', apply);
  }, [userSet]);

  const sel = new Set(selected || []);
  const total = (datasets || []).length;

  return (
    <section className="mb-3 rounded-xl border border-border bg-surface-raised p-2.5">
      <div className="flex flex-wrap items-center gap-2">
        <button type="button"
          onClick={() => { setUserSet(true); setOpen((v) => !v); }}
          aria-expanded={open}
          className="flex h-9 items-center gap-1.5 rounded-md border border-border bg-app/60 px-2.5 text-content text-[0.75rem] font-semibold hover:border-indigo-400/50">
          <span aria-hidden>{open ? '▾' : '▸'}</span> Datasets
          <span className="font-normal text-content-muted">· {selectionSummary(sel.size, total)}</span>
        </button>
        {open && total > 0 && (
          <div className="flex items-center gap-1.5">
            <button type="button" onClick={onAll}
              className="flex h-9 items-center rounded-md border border-border bg-app/60 px-2.5 text-content-muted text-[0.6875rem] hover:text-content">
              Select all
            </button>
            <button type="button" onClick={onNone}
              className="flex h-9 items-center rounded-md border border-border bg-app/60 px-2.5 text-content-muted text-[0.6875rem] hover:text-content">
              Clear
            </button>
          </div>
        )}
      </div>
      {open && (
        total === 0 ? (
          <p className="mt-2 text-content-subtle text-[0.75rem]">
            No dataset has been trained yet — the canvas draws training runs, so it fills up
            after your first run finishes.
          </p>
        ) : (
          <ul className="mt-2 grid max-h-56 list-none grid-cols-1 gap-1 overflow-y-auto p-0 sm:grid-cols-2 lg:grid-cols-3">
            {datasets.map((d) => (
              <li key={d.id}>
                <label className="flex min-h-[2.25rem] cursor-pointer items-center gap-2 rounded-md border border-transparent px-2 py-1 hover:border-border hover:bg-app/50">
                  <input type="checkbox" checked={sel.has(d.id)}
                    onChange={() => onToggle(d.id)}
                    className="h-4 w-4 shrink-0 accent-indigo-500" />
                  <span className="min-w-0 flex-1 truncate text-content text-[0.75rem]" title={d.name}>
                    {d.name}
                  </span>
                  <span className="shrink-0 text-content-subtle text-[0.625rem] tabular-nums"
                    title={(d.families || []).map(familyLabel).join(', ')}>
                    {d.runs}
                  </span>
                </label>
              </li>
            ))}
          </ul>
        )
      )}
    </section>
  );
}
