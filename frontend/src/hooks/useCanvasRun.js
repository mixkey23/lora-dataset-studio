/**
 * useCanvasRun — the ◉ LoRA Canvas' generation, tracked by the BOARD.
 *
 * It used to be tracked by the settings panel: `useCanvasStudio` held the run id
 * in its own state, and that hook is mounted by CanvasGenerationPanel. Closing
 * the panel — or leaving the page — unmounted it, and the run id went with it.
 * ComfyUI kept rendering, but reopening the panel showed a blank form: the only
 * moment you could watch a launch was the moment you launched it.
 *
 * So the run moves up to the board, and is remembered in localStorage. The
 * consequences are the point:
 *   • the progress is visible on the board itself, panel open or closed;
 *   • reopening the panel finds the run in flight, not the form;
 *   • a reload — or coming back to the page later — finds it too;
 *   • the checkpoints it was launched on are remembered with it, which is what
 *     lets the board say WHERE the images landed once they are done.
 *
 * ONE poller. The board owns it and hands the run object down to the panel, so
 * an open panel does not double the polling of the same run.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { useStudioRun } from './useStudioRun';
import { readCanvasRun, writeCanvasRun } from '../utils/canvasRunResults';

const store = () => (typeof localStorage !== 'undefined' ? localStorage : null);

export function useCanvasRun() {
  const [remembered, setRemembered] = useState(() => readCanvasRun(store()));
  const runId = remembered?.runId ?? null;
  const run = useStudioRun(runId);

  /** Adopt a freshly launched run, with the checkpoints it was launched on. */
  const adopt = useCallback((id, targets) => {
    const next = id ? { runId: String(id), targets: targets || [] } : null;
    setRemembered(next);
    writeCanvasRun(store(), next);
  }, []);

  const forget = useCallback(() => {
    setRemembered(null);
    writeCanvasRun(store(), null);
  }, []);

  // A remembered run the server no longer knows (its history was cleared, the
  // database was replaced) would otherwise haunt the board forever. useStudioRun
  // answers 404 by leaving `data` null, so a run that never materialises after a
  // first poll is dropped. Guarded by a ref so this only ever fires once per id.
  const checked = useRef(null);
  useEffect(() => {
    if (!runId || checked.current === runId) return undefined;
    const t = setTimeout(() => {
      checked.current = runId;
      if (!run.data) forget();
    }, 8000);
    return () => clearTimeout(t);
  }, [runId, run.data, forget]);

  return { runId, run, targets: remembered?.targets || [], adopt, forget };
}

export default useCanvasRun;
