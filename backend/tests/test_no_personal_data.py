"""The repo is public and the account behind it is pseudonymous.

CLAUDE.md forbids real names, machine paths, emails and tokens in code,
comments, commits and test fixtures. That rule was respected by intent and
broken by accretion: twelve comments had picked up the maintainer's first name
in ten days — several written the same night, by different agents and by me.
A rule nothing enforces is a rule that decays.

TWO KINDS OF CHECK, on purpose:

* PATTERNS always run and need no secret list — a Windows user path, an email,
  a bearer token or an API key shape has no business in this repo whoever you
  are. This is the half that protects a contributor who never read CLAUDE.md.

* NAMES are read from a list that is NOT in the repo, because writing the name
  down to forbid it would publish it. Point `LDS_PRIVACY_NAMES` at a file (one
  identifier per line) or drop `.privacy-names` next to the repo root — both are
  gitignored. With no list the name check SKIPS and says so: a silent pass would
  be worse than no test.
"""
import os
import re
import subprocess

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suffixes worth scanning. Binaries, lockfiles and the built bundle are excluded:
# dist is generated from sources this test already covers.
_SCANNED = ('.py', '.js', '.jsx', '.mjs', '.md', '.json', '.yml', '.yaml',
            '.html', '.css', '.bat', '.ps1', '.txt')
_SKIP_DIRS = ('frontend/dist/', 'docs/superpowers/', 'node_modules/')

_PATTERNS = {
    'a Windows user path': re.compile(r'[A-Za-z]:[\/]+Users[\/]+(?!<)[A-Za-z0-9._-]+', re.I),
    'an email address': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
    'an OpenAI-shaped key': re.compile(r'\bsk-[A-Za-z0-9]{20,}'),
    'a bearer token': re.compile(r'\bBearer\s+[A-Za-z0-9._-]{20,}'),
}
# Windows paths are only personal when the account name is one. A documented
# placeholder is what we WANT contributors to write.
_PLACEHOLDER_USERS = ('user', 'users', 'username', 'youruser', 'yourname',
                      'somebody', 'someone', 'me', 'you', 'public', 'default',
                      'all users', 'test', 'example')
# RFC 2606 / 6761 reserve these for documentation and tests; a fixture that
# uses one cannot belong to a person.
_RESERVED_DOMAINS = ('.example', '.test', '.invalid', '.localhost',
                     'example.com', 'example.net', 'example.org')
_ALLOWED_EMAILS = ('noreply@lora-dataset-studio.dev',)


def _is_personal_email(found, before):
    """`before` is the text immediately preceding the match.

    Three ways a match is NOT someone's address, each hit by a real fixture in
    this repo — the test is worthless if it cries wolf on all of them:
      * URL userinfo (`https://pexels.com@evil.example/`) — the domain-spoofing
        tests need exactly this shape;
      * a reserved documentation domain;
      * a stub too short to be anyone (`u@x.io`): a one-letter local part or a
        one-letter domain label is a placeholder, never a real mailbox.
    """
    if found in _ALLOWED_EMAILS:
        return False
    token = re.split(r'''[\s'"(\[<]''', before)[-1]    # the run glued to the match
    if '://' in token and '/' not in token.split('://', 1)[1]:
        return False                                   # userinfo inside a URL
    local, _, domain = found.rpartition('@')
    if any(domain.endswith(d) or domain == d for d in _RESERVED_DOMAINS):
        return False
    if len(local) < 3 or len(domain.split('.')[0]) < 3:
        return False
    return True


def _tracked_files():
    out = subprocess.run(['git', 'ls-files'], cwd=_REPO,
                         capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        pytest.skip('not a git checkout')
    for rel in out.stdout.splitlines():
        if not rel.endswith(_SCANNED):
            continue
        if any(rel.startswith(d) for d in _SKIP_DIRS):
            continue
        yield rel


def _read(rel):
    try:
        with open(os.path.join(_REPO, rel), encoding='utf-8', errors='replace') as fh:
            return fh.read()
    except OSError:
        return ''


def test_no_machine_path_email_or_token_in_tracked_files():
    r"""The half that needs no secret list — and would have caught a pasted
    diagnostic containing `C:\Users\<someone>` long before a human noticed."""
    hits = []
    for rel in _tracked_files():
        # This file necessarily contains the patterns it forbids.
        if rel.endswith('test_no_personal_data.py'):
            continue
        body = _read(rel)
        for label, rx in _PATTERNS.items():
            for m in rx.finditer(body):
                found = m.group(0)
                if label == 'an email address' and not _is_personal_email(
                        found, body[max(0, m.start() - 80):m.start()]):
                    continue
                if label == 'a Windows user path':
                    account = re.split(r'[\\/]+', found)[-1]
                    if account.lower() in _PLACEHOLDER_USERS:
                        continue
                line = body[:m.start()].count('\n') + 1
                hits.append(f'{rel}:{line} — {label}: {found[:60]}')
    assert not hits, (
        'personal data in a PUBLIC repo:\n  ' + '\n  '.join(hits[:20]))


def _name_list():
    path = os.environ.get('LDS_PRIVACY_NAMES') or os.path.join(_REPO, '.privacy-names')
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as fh:
        return [w.strip() for w in fh if w.strip() and not w.startswith('#')]


def test_no_forbidden_name_in_tracked_files():
    """Names come from a list kept OUT of the repo — writing them here to forbid
    them would publish them, which is the whole problem."""
    names = _name_list()
    if not names:
        pytest.skip('no name list — set LDS_PRIVACY_NAMES or add .privacy-names '
                    '(gitignored) to enable the name check')
    rx = re.compile(r'\b(' + '|'.join(re.escape(n) for n in names) + r')\b', re.I)
    hits = []
    for rel in _tracked_files():
        body = _read(rel)
        for m in rx.finditer(body):
            line = body[:m.start()].count('\n') + 1
            hits.append(f'{rel}:{line} — {m.group(0)}')
    assert not hits, (
        'a forbidden identifier is in the PUBLIC repo:\n  ' + '\n  '.join(hits[:20]))
