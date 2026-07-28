"""Deleting a bank whose folder lives on ANOTHER DRIVE must work.

Reported on 2026-07-26: removing a bank pointing at `A:\\...` while the
app runs from `C:\\` answered HTTP 500. The cause was `_is_imported_source`, whose
`os.path.commonpath([root, p])` call sat OUTSIDE its try/except — and on Windows
that call RAISES `ValueError: Paths don't have the same drive` rather than
returning a non-match. A bank on a second disk is the ordinary case, not an edge
one, so every such delete failed.

The question the function asks is "did WE create this folder under
bank_sources_root?". A path on a different drive is certainly not under it, so the
honest answer is False — never an exception.
"""
import os
from unittest.mock import patch

from app.services import image_bank_service as ibs


def test_a_folder_on_another_drive_is_not_ours_instead_of_raising():
    """The exact production failure, reproduced without needing a second drive:
    commonpath raises, and the caller must still get a plain False."""
    with patch.object(ibs.os.path, 'commonpath',
                      side_effect=ValueError("Paths don't have the same drive")):
        assert ibs._is_imported_source(os.path.join('somewhere', 'a bank')) is False


def test_a_folder_we_created_is_still_recognised(tmp_path):
    """Non-regression: the whole point of the function still works — a folder we
    made under bank_sources_root is reported as ours, so its copy is cleaned up."""
    root = tmp_path / 'bank_sources'
    mine = root / 'imported-bank'
    mine.mkdir(parents=True)
    with patch.object(ibs.cfg, 'bank_sources_root', return_value=str(root)):
        assert ibs._is_imported_source(str(mine)) is True
        # the root itself is not "a bank we made"
        assert ibs._is_imported_source(str(root)) is False


def test_a_users_own_folder_is_never_claimed(tmp_path):
    """The safety property this function exists for: a folder the user owns must
    never be reported as ours, or delete_bank would erase their images."""
    root = tmp_path / 'bank_sources'
    root.mkdir(parents=True)
    theirs = tmp_path / 'my photos'
    theirs.mkdir()
    with patch.object(ibs.cfg, 'bank_sources_root', return_value=str(root)):
        assert ibs._is_imported_source(str(theirs)) is False
        assert ibs._is_imported_source('') is False
        assert ibs._is_imported_source(None) is False
