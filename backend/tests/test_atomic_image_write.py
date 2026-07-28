"""A generated image must never be readable half-written.

Reported on 2026-07-26 after an OpenRouter generation: the tile was
black. The file on disk was a perfectly good 1024x1024 WEBP — caught mid-flight
it measured ZERO bytes, and the server answered it with HTTP 200.

Cause, and nothing about it was engine-specific: `open(path, 'wb')` truncates at
once, while `fh.write(normalize_to_webp(out))` only produces its bytes a second
or two later (decode + resize + WEBP encode). For that whole window an EMPTY
file sat under the image's final name — and the grid polls the dataset while a
batch runs, so the browser asked for it and got zero bytes back.

`write_image_atomic` writes beside the target and renames. os.replace is atomic
on one filesystem, so a reader sees the old state or the new one. A missing file
is already handled everywhere (the tile shows its pending state), which is the
truthful answer while the image is still encoding.
"""
import os

import pytest

from app.services.face_dataset_service import write_image_atomic

PAYLOAD = b'RIFF____WEBPVP8 ' + b'\x00' * 4096


def test_the_final_name_never_exists_empty(tmp_path):
    """The exact production symptom: while the bytes are being produced, the
    target must not exist at all — never exist at zero bytes."""
    target = tmp_path / 'local_ORFace_deadbeef.webp'
    seen = []

    class SlowBytes(bytes):
        pass

    # Observe the directory at the moment the temp file is being written.
    original_replace = os.replace

    def spy(src, dst):
        # Before the rename: the final name must be absent, and whatever is
        # there under another name is none of the reader's business.
        seen.append(os.path.exists(dst))
        return original_replace(src, dst)

    os.replace = spy
    try:
        write_image_atomic(str(target), PAYLOAD)
    finally:
        os.replace = original_replace

    assert seen == [False], 'the target existed before the write completed'
    assert target.read_bytes() == PAYLOAD


def test_a_failed_write_leaves_no_target_and_no_leftover(tmp_path):
    """If the bytes never arrive, the reader must see nothing — not an empty
    image, and not a stray .part file cluttering the dataset folder."""
    target = tmp_path / 'local_ORFace_cafe1234.webp'

    original_replace = os.replace

    def failing_replace(src, dst):
        raise OSError('disk went away')

    os.replace = failing_replace
    try:
        with pytest.raises(OSError):
            write_image_atomic(str(target), PAYLOAD)
    finally:
        os.replace = original_replace

    assert not target.exists(), 'a failed write must not publish the image'
    assert list(tmp_path.iterdir()) == [], 'a .part file was left behind'


def test_replacing_an_existing_image_is_all_or_nothing(tmp_path):
    """Re-cropping writes the same filename. The previous image must stay
    readable until the new one is complete."""
    target = tmp_path / 'local_datasetref_1234abcd.webp'
    target.write_bytes(b'the previous image')

    original_replace = os.replace
    during = {}

    def spy(src, dst):
        during['content'] = open(dst, 'rb').read()   # mid-write: still the old one
        return original_replace(src, dst)

    os.replace = spy
    try:
        write_image_atomic(str(target), PAYLOAD)
    finally:
        os.replace = original_replace

    assert during['content'] == b'the previous image'
    assert target.read_bytes() == PAYLOAD
