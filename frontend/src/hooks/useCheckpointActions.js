/**
 * useCheckpointActions — deploying and deleting ONE checkpoint, for whichever
 * surface is showing it.
 *
 * The two writes behind the checkpoint popover (📦 Deploy → loras/…, and the
 * single 🗑/⏏ action that aims at either the ComfyUI copy or the training save)
 * used to live inside RunLineageGraph.jsx. The LoRA Canvas needed exactly the
 * same two, which left the usual choice: copy them, or share them. They are
 * shared — a second copy of "which file does this button delete" is precisely
 * the drift the popover extraction exists to prevent.
 *
 * Every rule stays where it already was: `lineageImportPayload` builds the
 * deploy body, `checkpointDeleteTarget` picks the route AND the file, and
 * `describeCheckpointDelete` writes the confirmation that names it. This hook
 * only carries them out.
 *
 * `postJson` (never a bare fetch): a state-changing POST without the X-CSRFToken
 * header is rejected 400 by Flask-WTF — the trap that once broke browser
 * Generate. It also THROWS on 400/409, so the server's own words (e.g. "this
 * dataset is training right now") reach the toast instead of being swallowed.
 *
 * `onChanged(datasetId)` is awaited after a success so the caller can re-read
 * its lineage: a just-deployed pill has to flip to ✓ Deployed, a just-undeployed
 * one back to offering Deploy, a deleted save has to disappear. Without it the
 * popover would keep making a claim about the disk that stopped being true.
 * Both actions answer `true` on success so the host can close the popover.
 */
import { useCallback, useState } from 'react';
import { postJson } from '../api/fetchClient';
import { useToast } from '../components/common/Toast';
import { loraFolderLabel } from '../utils/checkpointBrowser';
import {
  checkpointDeleteTarget, describeCheckpointDelete, lineageImportPayload,
} from '../components/dataset/lineagePreview.js';

export function useCheckpointActions({ onChanged, bestSettingsLora = null } = {}) {
  const toast = useToast();
  const [importing, setImporting] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const deployCheckpoint = useCallback(async (datasetId, node, pill) => {
    const body = lineageImportPayload(node, pill);
    if (datasetId == null || !body) return false;
    setImporting(true);
    try {
      const d = await postJson(`/api/dataset/${datasetId}/train/import`, body);
      toast.success(d?.note || `LoRA deployed to ${loraFolderLabel(node.train_type)}: ${d?.dest || pill.filename}`);
      try { await onChanged?.(datasetId); } catch { /* it already landed server-side */ }
      return true;
    } catch (e) {
      toast.error(e?.message || 'Deploy failed');
      return false;
    } finally {
      setImporting(false);
    }
  }, [onChanged, toast]);

  const deleteCheckpoint = useCallback(async (datasetId, node, pill) => {
    const target = checkpointDeleteTarget(node, pill);
    if (datasetId == null || !target) return false;
    const { message } = describeCheckpointDelete(node, pill, { bestSettingsLora }) || {};
    if (message && !window.confirm(message)) return false;
    setDeleting(true);
    try {
      await postJson(`/api/dataset/${datasetId}/${target.path}`, target.body);
      toast.success(target.kind === 'deployed'
        ? `Undeployed from ComfyUI — the training save is kept, you can deploy it again: ${target.filename}`
        : `Training save moved to the trash: ${target.filename}`);
      try { await onChanged?.(datasetId); } catch { /* it is deleted server-side */ }
      return true;
    } catch (e) {
      toast.error(e?.message || 'Delete failed');
      return false;
    } finally {
      setDeleting(false);
    }
  }, [bestSettingsLora, onChanged, toast]);

  return { importing, deleting, deployCheckpoint, deleteCheckpoint };
}

export default useCheckpointActions;
