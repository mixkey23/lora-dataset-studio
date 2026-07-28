import { useMemo } from 'react';
import RunSetupPanel from '../dataset/studio/RunSetupPanel';
import { useCanvasStudio } from '../../hooks/useCanvasStudio';
import { useStudioForm } from '../../hooks/useStudioForm';
import {
  canvasFamily, canvasSelectionSummary, describeCanvasLaunch,
} from '../../utils/canvasGeneration';
import { famLabel } from '../../utils/familyLabels';
import { DEPLOY_BAR_CLASS } from '../../utils/checkpointDeployState';
import { runNumber } from '../../utils/runIdentity';
import { HelpBadge } from '../../help/HelpMode';

/* 🎨 Generating from the board.

   This panel is deliberately thin. It mounts the Test Studio's own hooks and
   renders the Test Studio's own settings panel — same prompt field, same seed
   controls, same model/format/cfg/steps axes, same global generation settings.
   The ONE difference is where the checkpoints come from: ticked pills on the
   canvas instead of a picker, possibly from several datasets at once.

   Because it is the same code, a setting added to the Test Studio appears here
   without anyone touching this file, and the two screens cannot drift.

   Layout: a bottom sheet under `sm`, a side drawer above it. A full settings
   panel on a 400-px canvas is the hard case of this screen — a side drawer
   there would leave a sliver of board, and the board is what you are picking
   from. */

/** The picks, as the panel shows them back: one removable chip per checkpoint,
 *  grouped so a cross-dataset run reads as one. Replaces the CheckpointPicker. */
function CanvasCheckpointRecap({ selection, onToggle, onClear }) {
  const lanes = useMemo(() => {
    const by = new Map();
    for (const e of selection) {
      if (!by.has(e.datasetId)) by.set(e.datasetId, { name: e.datasetName, picks: [] });
      by.get(e.datasetId).picks.push(e);
    }
    return [...by.entries()];
  }, [selection]);

  return (
    <div id="st-loras" className="scroll-mt-16 rounded-lg border border-border bg-app/40 p-2">
      <div className="mb-1.5 flex flex-wrap items-center gap-x-2 gap-y-1">
        <span className="text-[0.6875rem] font-semibold text-content">
          <span aria-hidden>🧬</span> Checkpoints
        </span>
        <span className="text-[0.625rem] text-content-muted">{canvasSelectionSummary(selection)}</span>
        <HelpBadge topic="canvas-generate" />
        {selection.length > 0 && (
          <button type="button" onClick={onClear}
            className="ml-auto text-content-subtle text-[0.625rem] underline decoration-dotted hover:text-content">
            Clear
          </button>
        )}
      </div>
      {selection.length === 0 ? (
        <p className="m-0 text-content-subtle text-[0.6875rem]">
          Tick the ✓ box on a checkpoint pill to add it here. Picks from several datasets
          run together — that is what the board is for.
        </p>
      ) : (
        <div className="flex flex-col gap-1.5">
          {lanes.map(([datasetId, lane]) => (
            <div key={datasetId} className="min-w-0">
              <div className="truncate text-content-subtle text-[0.5625rem] font-semibold uppercase tracking-wide">
                {lane.name || `Dataset ${datasetId}`}
              </div>
              <div className="mt-0.5 flex flex-wrap gap-1">
                {/* Same code as the board's pills: a solid sky edge means
                    deployed, a dashed slate one means on disk only. The chip
                    used to say "· to deploy" in a register of its own, which
                    made the panel and the board two vocabularies for one fact.
                    The words stay too — the bar is never the only signal. */}
                {lane.picks.map((e) => (
                  <button key={`${e.recordId}:${e.step}`} type="button"
                    onClick={() => onToggle(e)}
                    title={`Remove step ${e.step} of run ${runNumber({ record_id: e.recordId })} from this run`
                      + (e.deployed ? ' — deployed to ComfyUI'
                        : ' — on disk but not deployed yet; the launch deploys it first')}
                    className={'flex max-w-full items-center gap-1 rounded-md border px-1.5 py-0.5 text-[0.625rem] tabular-nums '
                      + (e.deployed
                        ? 'border-indigo-400/60 bg-indigo-500/15 text-indigo-100 '
                        : 'border-amber-400/50 bg-amber-500/10 text-amber-100 ')
                      + DEPLOY_BAR_CLASS[e.deployed ? 'deployed' : 'on-disk'] + ' '}>
                    <span className="truncate">
                      #{e.recordId} · {e.step}
                      {!e.deployed && <span className="opacity-80"> · not deployed yet</span>}
                    </span>
                    <span aria-hidden className="opacity-70">✕</span>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function CanvasGenerationPanel({ selection, onToggle, onClear, onDeploy, onClose, tracker }) {
  const family = canvasFamily(selection);
  const verdict = describeCanvasLaunch(selection);
  const studio = useCanvasStudio(selection, family, { onDeploy, tracker });
  const pinned = useMemo(
    () => (studio.data?.checkpoints || []).map((c) => c.filename), [studio.data]);
  // Namespaced per FAMILY, never per dataset: a canvas run is cross-dataset by
  // design, so "the settings of dataset 7" would be the wrong thing to restore.
  const form = useStudioForm(studio.data, 'canvas', family, { pinnedCheckpoints: pinned });

  const recap = (
    <CanvasCheckpointRecap selection={selection} onToggle={onToggle} onClear={onClear} />
  );

  return (
    <aside
      data-testid="canvas-generation-panel"
      aria-label="Generate from the canvas"
      className="fixed inset-x-0 bottom-0 z-50 flex max-h-[78vh] flex-col overflow-hidden border-t border-border bg-surface-overlay shadow-xl
                 sm:inset-x-auto sm:right-0 sm:top-0 sm:h-full sm:max-h-none sm:w-[26rem] sm:border-l sm:border-t-0">
      <header className="flex shrink-0 items-center gap-2 border-b border-border px-3 py-2">
        <h3 className="min-w-0 flex-1 truncate text-sm font-semibold text-content">
          <span aria-hidden>🎨</span> Generate from the board
          {family && <span className="font-normal text-content-muted"> · {famLabel(family)}</span>}
        </h3>
        <button type="button" onClick={onClose} aria-label="Close"
          className="shrink-0 text-content-subtle hover:text-content">✕</button>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        {!studio.data ? (
          // The recap stays even with nothing loaded: it is where the picks land,
          // and an empty panel would read as "the board did not register my click".
          <div className="flex flex-col gap-2">
            {recap}
            <p className="m-0 text-content-subtle text-[0.75rem]">
              {selection.length
                ? 'Loading this family’s settings…'
                : 'Tick a checkpoint on the board to set up a run.'}
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            <RunSetupPanel
              d={studio.data}
              studio={studio}
              form={form}
              datasetId={selection[0]?.datasetId ?? null}
              checkpointSlot={recap}
              launchBlocked={verdict.blocked}
              launchLabel={verdict.label}
              launchHint={verdict.reason}
              genStoragePrefix={`studioGen_canvas_${family || 'default'}`}
              // The Studio's fixed bottom bar would sit ON this sheet at 400 px.
              actionBar={false}
            />
          </div>
        )}
      </div>
    </aside>
  );
}
