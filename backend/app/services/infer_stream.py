"""Run one of the `backend/infer/*.py` helper scripts in the dedicated ML
interpreter and READ ITS PROGRESS WHILE IT RUNS.

Extracted verbatim from face_similarity._run_scorer, which solved this first for
the face-scoring pass: `subprocess.run()` only hands the output back once the
child has exited, so every `[tag] i/N` line those scripts print all along was
unreadable until it no longer mattered — the bar sat at 0 for the whole pass and
then jumped to N/N, which reads exactly like a hang.

Both the scoring pass and the concept face-mask preview now share this one
mechanism instead of growing a second copy of it. Callers own their own line
grammar (one regex each); this module owns the plumbing:

* Popen, not run() — stdout is read here, stderr is drained by a thread (a full
  pipe would deadlock the child), both bounded: only the last few stderr lines
  are kept, for the error message.
* A watchdog kills the child on timeout and says so, rather than hanging forever.
* A callback that raises must never take the pass down with it: progress is a
  display concern, the work is the work.
"""
from __future__ import annotations
import logging
import subprocess
import threading
from collections import deque

logger = logging.getLogger(__name__)

# How long we wait for the child to die after a timeout kill before giving up on
# joining its reader thread. Short: the pipes are closed by then.
_JOIN_GRACE_S = 5

# Enough stderr to identify a crash (for a Python traceback the last non-empty
# line is the `SomeError: ...` one, which is what a human wants to read) without
# buffering a chatty run.
_TAIL_LINES = 5


def stderr_tail(lines) -> str:
    """Last non-empty stderr line — the useful half of a child's traceback."""
    return next((ln.strip() for ln in reversed(list(lines or ())) if ln.strip()), '')


def run_infer_script(python, script, payload, timeout, on_line=None):
    """Run ``python script``, feed it ``payload`` on stdin, stream its stderr
    lines to ``on_line`` as they arrive.

    Returns ``(stdout, stderr_lines, returncode, timed_out)``.
    """
    proc = subprocess.Popen(
        [python, script], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace',
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    lines: deque = deque(maxlen=_TAIL_LINES)

    def _drain():
        for line in proc.stderr:
            line = line.strip()
            if not line:
                continue
            lines.append(line)
            if on_line:
                try:
                    on_line(line)
                except Exception:
                    logger.debug('infer progress callback failed', exc_info=True)

    reader = threading.Thread(target=_drain, daemon=True)
    reader.start()
    state = {'timed_out': False}

    def _kill():
        state['timed_out'] = True
        try:
            proc.kill()
        except OSError:
            pass

    watchdog = threading.Timer(max(1, int(timeout)), _kill)
    watchdog.daemon = True
    watchdog.start()
    try:
        try:
            proc.stdin.write(payload)
            proc.stdin.close()
        except OSError:
            pass        # the child died early — the caller reports it from rc
        stdout = proc.stdout.read()
        proc.wait()
    finally:
        watchdog.cancel()
        reader.join(timeout=_JOIN_GRACE_S)
    return stdout, lines, proc.returncode, state['timed_out']
