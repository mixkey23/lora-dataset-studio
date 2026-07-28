"""🗑 Delete a run AND everything it produced — the explicit, opt-in cascade.

WHY THIS IS A SEPARATE MODE, NOT THE NEW DEFAULT
------------------------------------------------
``cloud_training.delete_run_record`` removes a GONE run: a row the graph already
badges as having nothing on disk. It refuses (``has_saves``) a run whose
checkpoints still exist and it never deletes a generated image. That behaviour is
load-bearing — the ◉ Graph's "Remove this run" and every existing caller lean on
it — so it is left EXACTLY as it was. Changing its meaning under those callers
would turn a tidy-up button into a file shredder without anyone touching them.

The cascade is therefore its own function, reached only through an explicit
``?cascade=1`` on the delete route. What it takes:

  • every checkpoint this run produced, to the recycle bin / app trash (local
    run-dir saves through ``lora_training.delete_checkpoint``, cloud staging
    saves + the harvested final through ``trash.send_to_trash``);
  • every generated image linked to the run, through
    ``cloud_training.delete_checkpoint_images`` — the SAME function the gallery's
    🗑 uses, so the recycle-bin promise, the shared-file rule and the
    "a generating row is never cancelled" rule are kept in one place;
  • the run row and its notes / preview links / canvas position, through the same
    row sweep the conservative delete uses.

What it deliberately does NOT take, and why:

  • **Runs that continued from it** — detached (``parent_record_id`` → NULL), never
    deleted. They are real work with their own checkpoints; the parent's removal
    re-roots them and nothing else.
  • **LoRAs deployed into ComfyUI** — kept. A deployed copy lives in the ComfyUI
    models folder, outside the run, and the user's saved workflows reference it by
    filename. Deleting a run is a lineage decision; silently breaking generation
    graphs that have nothing to do with lineage is not what the click asked for.
    The confirmation counts them and points at Checkpoints & LoRAs → Undeploy.
  • **Images the user gave 👍** — kept, and unlinked like the conservative path.
    A thumbs-up is the one explicit "I am keeping this" signal the app records;
    destroying a picture the user marked as a keeper is worse than leaving an
    orphan behind. The confirmation says how many survive.

Refusals, never a best-effort mess:
  • a run whose dataset is training RIGHT NOW → ``training`` (409). Deleting
    files ai-toolkit is about to rewrite invites corruption — the same guard
    ``delete_checkpoint`` already enforces, checked up front so the cascade does
    not half-run;
  • a checkpoint that could not be moved → ``partial`` (409): the files that DID
    go are reported, the run row STAYS. A run whose record vanished while its
    weights sat on disk would be unreachable from the graph forever.

Every count returned is what actually happened, so a partial success can never be
displayed as a clean one."""
import os

from ..extensions import db
from ..utils.redact import redact_user_paths

import logging

logger = logging.getLogger(__name__)

# Rows in these states are still being produced; they are never counted as
# deletable images (delete_checkpoint_images skips them too, by design).
_LIVE_IMAGE_STATES = ('pending', 'running', 'queued')


def _rec(record_id):
    from ..models import TrainingRunRecord
    return db.session.get(TrainingRunRecord, int(record_id))


def _cloud_run(rec):
    from ..models import CloudTrainingRun
    if rec.source != 'cloud' or not rec.cloud_run_id:
        return None
    return db.session.get(CloudTrainingRun, rec.cloud_run_id)


def run_checkpoint_files(rec) -> list:
    """Absolute paths of the checkpoints THIS run owns on disk, with their size.

    Local: the run-dir saves ``list_checkpoints`` attributes to this record —
    the same attribution the ◉ Graph pills draw, so the cascade removes exactly
    what the card shows. Cloud: the harvested staging saves plus the final LoRA
    copy the run downloaded.

    ``[{'filename', 'path', 'bytes'}]``. Best-effort: a scan we cannot run
    yields an empty list — an unprovable file must never be claimed, and the
    caller's own "still on disk?" check is what decides a partial failure."""
    from . import cloud_training as ct
    from . import lora_training as lt
    from .. import config as cfg
    out = []
    try:
        crun = _cloud_run(rec)
        if crun is not None:
            for c in ct._run_staging_checkpoints(crun):
                out.append({'filename': c['filename'], 'path': c['path']})
            final = crun.checkpoint_local_path
            if final and os.path.isfile(final) \
                    and not any(os.path.normcase(e['path']) == os.path.normcase(final)
                                for e in out):
                out.append({'filename': os.path.basename(final), 'path': final})
        elif rec.source != 'cloud':
            run_dir = lt._run_dir(cfg.LOCAL_USER, rec.dataset_id, rec.base_model or '',
                                  rec.family, rec.variant)
            for c in lt.list_checkpoints(cfg.LOCAL_USER, rec.dataset_id,
                                         rec.base_model or '', rec.family, rec.variant):
                if c.get('run_source') != 'local' or c.get('run_id') != rec.id:
                    continue
                out.append({'filename': c['filename'],
                            'path': os.path.join(run_dir, c['filename'])})
    except Exception:
        logger.debug('checkpoint scan failed for the cascade preview', exc_info=True)
        return []
    for e in out:
        try:
            e['bytes'] = os.path.getsize(e['path'])
        except OSError:
            e['bytes'] = 0
    return out


def _image_split(record_id) -> tuple:
    """(ids to delete, kept count) for a run's generated images.

    Kept = the 👍 rows and anything still generating. Everything else is a test
    render the run produced and the cascade removes."""
    from ..models import LoraTestImage
    try:
        rows = LoraTestImage.query.filter_by(record_id=record_id).all()
    except Exception:
        logger.debug('image split failed', exc_info=True)
        return [], 0
    doomed, kept = [], 0
    for r in rows:
        if r.rating == 1 or (r.status or '') in _LIVE_IMAGE_STATES:
            kept += 1
        else:
            doomed.append(r.id)
    return doomed, kept


def _deployed_kept(rec, checkpoints) -> int:
    """How many of this run's saves have a copy deployed in ComfyUI. Those copies
    survive the cascade; the number exists so the dialog can say so out loud."""
    from . import cloud_training as ct
    try:
        rows = [{'step': c.get('step'), 'filename': c.get('filename')}
                for c in (checkpoints or [])]
        ct.annotate_deployed_checkpoints(rec.dataset_id, rec.family, rows,
                                         run_tag=ct._deployed_run_tag(rec))
        return sum(1 for r in rows if r.get('deployed_filename'))
    except Exception:
        logger.debug('deployment annotation failed for the cascade preview',
                     exc_info=True)
        return 0


def training_block_reason(rec) -> str | None:
    """``'local'`` / ``'cloud'`` while this run must not be touched, else None.

    A local dataset mid-training is rewriting the very run dir the cascade would
    empty; a cloud run in an active state is still producing the staging saves.
    Checked BEFORE anything is deleted so the refusal is total, not halfway."""
    from . import cloud_training as ct
    from . import lora_training as lt
    try:
        crun = _cloud_run(rec)
        if crun is not None and crun.status in ct.ACTIVE_STATES:
            return 'cloud'
        if lt._local_training_active_for(rec.dataset_id):
            return 'local'
    except Exception:
        logger.debug('training guard probe failed', exc_info=True)
    return None


def cascade_impact(record_id) -> dict | None:
    """What the CASCADE would take, counted — the extra half of the deletion
    preview the confirmation dialog reads. None when the run is unknown.

    Every figure degrades to 0/False on its own rather than failing the preview:
    a fresh install, a run whose ai-toolkit is not configured, a staging folder
    already purged by hand."""
    rec = _rec(record_id)
    if rec is None:
        return None
    cks = run_checkpoint_files(rec)
    doomed, kept = _image_split(rec.id)
    return {
        'checkpoints': len(cks),
        'checkpoint_bytes': sum(int(c.get('bytes') or 0) for c in cks),
        'images_deleted': len(doomed),
        'images_kept_rated': kept,
        'deployed_kept': _deployed_kept(rec, [
            {'step': None, 'filename': c['filename']} for c in cks]),
        'training_active': training_block_reason(rec),
    }


def _trash_checkpoints(rec, entries) -> dict:
    """Move this run's saves to the trash. Returns {'deleted', 'failed', 'error'}.

    Local saves go through ``lora_training.delete_checkpoint`` so the anti
    path-traversal whitelist and the active-training refusal apply exactly as
    they do from the Checkpoints panel. Cloud staging files are not in that
    whitelist (they live outside the run dir) and go straight to the shared
    ``trash`` helper. Any error message is path-redacted before it can reach a
    response body."""
    from . import lora_training as lt
    from . import trash
    from .. import config as cfg
    out = {'deleted': 0, 'failed': 0, 'error': None}
    crun = _cloud_run(rec)
    for e in entries:
        try:
            if crun is not None:
                trash.send_to_trash(e['path'], context=f'run{rec.id}')
            else:
                lt.delete_checkpoint(cfg.LOCAL_USER, rec.dataset_id, e['filename'],
                                     rec.base_model or '', rec.family, rec.variant)
            out['deleted'] += 1
        except Exception as exc:                       # noqa: BLE001 — reported, not raised
            out['failed'] += 1
            if out['error'] is None:
                out['error'] = redact_user_paths(str(exc) or exc.__class__.__name__)[:200]
            logger.warning('cascade: could not trash a checkpoint of run %s', rec.id,
                           exc_info=True)
    return out


def delete_run_cascade(record_id) -> dict:
    """Delete a run AND what it produced. See the module docstring for the
    arbitrations; this is the order they are executed in.

    Returns ``{'status', 'checkpoints_deleted', 'checkpoints_failed',
    'images_deleted', 'images_kept', 'error'}`` with
    ``status`` in ``not_found | training | partial | conflict | deleted``.

    Files first, row last, on purpose: a run whose record was already gone while
    its weights sat on disk would be unreachable from the graph — no card, no
    pill, no delete. So a checkpoint that refuses to move aborts BEFORE the row
    is touched and the run stays exactly where the user can see it."""
    from . import cloud_training as ct
    out = {'status': 'not_found', 'checkpoints_deleted': 0, 'checkpoints_failed': 0,
           'images_deleted': 0, 'images_kept': 0, 'error': None}
    rec = _rec(record_id)
    if rec is None:
        return out

    blocked = training_block_reason(rec)
    if blocked:
        out['status'] = 'training'
        out['error'] = blocked
        return out

    files = _trash_checkpoints(rec, run_checkpoint_files(rec))
    out['checkpoints_deleted'] = files['deleted']
    out['checkpoints_failed'] = files['failed']
    if files['failed']:
        # Partial: the run row stays so the remaining weights are still reachable
        # from the card that owns them.
        out['status'] = 'partial'
        out['error'] = files['error']
        return out

    doomed, kept = _image_split(rec.id)
    out['images_kept'] = kept
    if doomed:
        try:
            res = ct.delete_checkpoint_images(rec.id, None, doomed)
            out['images_deleted'] = int(res.get('rows_removed') or 0)
            skipped = len(res.get('skipped') or [])
            if skipped:
                out['images_kept'] += skipped
        except Exception as exc:                       # noqa: BLE001
            out['status'] = 'partial'
            out['error'] = redact_user_paths(str(exc) or exc.__class__.__name__)[:200]
            return out

    # The row sweep is the conservative delete's own — one flush order, one
    # place where the "detach the children, unlink what survives" promise lives.
    # Its checkpoint guard cannot fire any more: the files just went to the trash.
    status = ct.delete_run_record(rec.id, cascade=True)
    out['status'] = 'deleted' if status == 'deleted' else status
    return out
