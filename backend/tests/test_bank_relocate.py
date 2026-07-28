"""🗃️ Moving a bank's folder — repoint the bank, keep every analysis.

A bank row stores `source_path`; every BankImage stores a relpath UNDER it, and
every analysis (quality scores, dhash, face state, captions, keep/reject
decisions) hangs off the row id, never off a path. So moving the folder to
another drive should cost nothing — provided (a) the bank can be repointed at
all, and (b) a pass that runs while the folder is elsewhere does not treat the
absence of the files as a defect of the images.

(b) is the dangerous half: an unreadable file is auto-rejected by the quality
pass, and "the folder moved" used to look exactly like "every file is
unreadable" — a silent mass-reject over a whole bank.
"""
import os

import pytest
from PIL import Image


def _save(path, im):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path)


def photo(size=128, seed=0):
    """Smooth structure — a stable dHash and non-degenerate quality metrics."""
    im = Image.new('L', (size, size))
    c, r2 = size / 2, (size / 3) ** 2
    im.putdata([min(255, int(150 * x / size + 50 * y / size) + seed
                    + (80 if (x - c) ** 2 + (y - c) ** 2 < r2 else 0))
                for y in range(size) for x in range(size)])
    return im.convert('RGB')


def _mkbank(client, root, rels, name='B'):
    src = root / 'src'
    for i, rel in enumerate(rels):
        _save(str(src / rel), photo(seed=i * 7))
    r = client.post('/api/bank/create', json={'name': name, 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id'], src


def _move(src, dest):
    """Simulate "the user moved the folder to another drive": the old path no
    longer exists, the same tree sits at a brand-new one."""
    os.makedirs(os.path.dirname(str(dest)), exist_ok=True)
    os.rename(str(src), str(dest))
    return dest


def _get(model, row_id):
    from app.models import db
    return db.session.get(model, row_id)


def _statuses(app, bank_id):
    from app.models import BankImage
    with app.app_context():
        return sorted((r.relpath.replace('\\', '/'), r.status,
                       r.reject_reason, r.quality_state)
                      for r in BankImage.query.filter_by(bank_id=bank_id))


# --- (b) the guard: a moved folder must not degrade the bank -----------------
def test_scan_on_a_moved_folder_rejects_nothing(client, app, tmp_path):
    """THE regression. Files absent (folder moved / drive unplugged) + quality
    pass: not one image may be auto-rejected, and the job must say the folder
    moved instead of quietly grading 30k images as unreadable."""
    rels = [f'a{i}.png' for i in range(6)]
    bank_id, src = _mkbank(client, tmp_path, rels)
    _move(src, tmp_path / 'moved')

    r = client.post(f'/api/bank/{bank_id}/scan')
    assert r.status_code in (202, 409), r.get_json()

    rows = _statuses(app, bank_id)
    assert len(rows) == len(rels)
    assert [x[1] for x in rows] == ['pending'] * len(rels), rows
    assert all(x[2] is None for x in rows), rows
    # ...and nothing was graded 'unreadable' either: absent is not a defect.
    assert all(x[3] != 'unreadable' for x in rows), rows

    payload = client.get(f'/api/bank/{bank_id}').get_json()
    job = payload.get('activity') or {}
    blurb = f"{job.get('error') or ''} {job.get('detail') or ''}".lower()
    assert 'moved' in blurb, job


def test_scan_stops_when_the_folder_is_there_but_empty(client, app, tmp_path):
    """The nastier shape of the same accident: the PATH still resolves (a drive
    letter got reused, a sync folder came back empty) so the root check passes,
    yet every file is gone. The pass must bail out on the pattern instead of
    walking 30 000 absences and grading each one."""
    rels = [f'a{i}.png' for i in range(25)]
    bank_id, src = _mkbank(client, tmp_path, rels)
    for rel in rels:
        os.remove(str(src / rel))

    assert client.post(f'/api/bank/{bank_id}/scan').status_code == 202
    rows = _statuses(app, bank_id)
    assert {x[1] for x in rows} == {'pending'}, rows
    assert {x[3] for x in rows} == {None}, rows
    job = client.get(f'/api/bank/{bank_id}').get_json().get('activity') or {}
    assert 'moved' in (job.get('error') or '').lower(), job


def test_scan_still_rejects_a_genuinely_corrupt_file(client, app, tmp_path):
    """The guard must not disarm the legitimate case: the folder is right where
    it should be and ONE file is garbage — that one is still auto-rejected."""
    bank_id, src = _mkbank(client, tmp_path, ['good.png'])
    (src / 'broken.png').write_bytes(b'not a png at all')
    client.get(f'/api/bank/{bank_id}')          # folder walk picks the new file up

    r = client.post(f'/api/bank/{bank_id}/scan')
    assert r.status_code == 202, r.get_json()
    rows = dict((x[0], (x[1], x[3])) for x in _statuses(app, bank_id))
    assert rows['broken.png'] == ('reject', 'unreadable'), rows
    assert rows['good.png'][0] == 'pending', rows


# --- (a) relocate ------------------------------------------------------------
def _analyse(client, app, bank_id):
    """Run the quality pass, then stamp a decision and a caption by hand, so the
    relocate assertions cover BOTH computed analysis and user decisions."""
    from app.models import BankImage
    assert client.post(f'/api/bank/{bank_id}/scan').status_code == 202
    with app.app_context():
        rows = BankImage.query.filter_by(bank_id=bank_id).order_by(
            BankImage.id.asc()).all()
        rows[0].status = 'keep'
        rows[1].status, rows[1].reject_reason = 'reject', 'manual'
        rows[2].caption = 'a photo of a test pattern'
        from app.models import db
        db.session.commit()
        return {r.relpath.replace('\\', '/'): (
            r.status, r.reject_reason, r.quality_state, r.blur_score,
            r.noise_score, r.uniformity_score, r.dhash, r.caption)
            for r in rows}


def test_relocate_keeps_every_analysis_and_serves_from_the_new_path(
        client, app, tmp_path):
    rels = ['a.png', 'sub/b.png', 'sub/deep/c.png', 'd.png']
    bank_id, src = _mkbank(client, tmp_path, rels)
    before = _analyse(client, app, bank_id)
    dest = _move(src, tmp_path / 'other-drive' / 'bank')

    r = client.post(f'/api/bank/{bank_id}/relocate', json={'folder': str(dest)})
    body = r.get_json()
    assert r.status_code == 200, body
    assert body['applied'] is False and body['found'] == len(rels)
    assert body['missing'] == 0

    r = client.post(f'/api/bank/{bank_id}/relocate',
                    json={'folder': str(dest), 'confirm': True})
    body = r.get_json()
    assert r.status_code == 200, body
    assert body['applied'] is True

    from app.models import BankImage, ImageBank
    with app.app_context():
        bank = _get(ImageBank, bank_id)
        assert os.path.normcase(bank.source_path) == os.path.normcase(
            os.path.realpath(str(dest)))
        after = {r.relpath.replace('\\', '/'): (
            r.status, r.reject_reason, r.quality_state, r.blur_score,
            r.noise_score, r.uniformity_score, r.dhash, r.caption)
            for r in BankImage.query.filter_by(bank_id=bank_id)}
    assert after == before

    # ...and the bytes are served again, from the new location.
    with app.app_context():
        one = BankImage.query.filter_by(bank_id=bank_id).first().id
    assert client.get(f'/api/bank/{bank_id}/file/{one}').status_code == 200


def test_relocate_refuses_a_folder_that_is_not_this_bank(client, app, tmp_path):
    rels = ['a.png', 'b.png', 'c.png']
    bank_id, src = _mkbank(client, tmp_path, rels)
    before = _statuses(app, bank_id)
    stranger = tmp_path / 'someone-elses'
    _save(str(stranger / 'zzz.png'), photo())

    r = client.post(f'/api/bank/{bank_id}/relocate',
                    json={'folder': str(stranger), 'confirm': True})
    body = r.get_json()
    assert r.status_code == 400, body
    assert body['found'] == 0 and body['missing'] == len(rels)
    assert 'error' in body

    from app.models import ImageBank
    with app.app_context():
        assert os.path.normcase(_get(ImageBank, bank_id).source_path) \
            == os.path.normcase(os.path.realpath(str(src)))
    assert _statuses(app, bank_id) == before          # not one row lost


def test_relocate_needs_confirmation_when_files_are_missing(
        client, app, tmp_path):
    """A PARTIAL match is the ambiguous case: report both numbers and require an
    explicit confirmation — then keep every row, missing files included."""
    rels = ['a.png', 'b.png', 'c.png', 'd.png']
    bank_id, src = _mkbank(client, tmp_path, rels)
    dest = _move(src, tmp_path / 'moved')
    os.remove(str(dest / 'c.png'))
    os.remove(str(dest / 'd.png'))

    r = client.post(f'/api/bank/{bank_id}/relocate', json={'folder': str(dest)})
    body = r.get_json()
    assert r.status_code == 200, body
    assert (body['found'], body['missing']) == (2, 2)
    assert body['applied'] is False and body['needs_confirm'] is True
    assert body['missing_sample']

    r = client.post(f'/api/bank/{bank_id}/relocate',
                    json={'folder': str(dest), 'confirm': True})
    assert r.status_code == 200 and r.get_json()['applied'] is True
    assert len(_statuses(app, bank_id)) == len(rels)     # nothing deleted


def test_relocate_matches_case_insensitively_on_windows(client, app, tmp_path):
    """The same tree re-created with a different case (or reached through a
    differently-cased drive letter) is still the same bank on Windows."""
    from app.services import image_bank_service as banks
    bank_id, src = _mkbank(client, tmp_path, ['Photo.PNG', 'Sub/Two.png'])
    dest = tmp_path / 'dest'
    os.makedirs(str(dest / 'sub'), exist_ok=True)
    _save(str(dest / 'photo.png'), photo())
    _save(str(dest / 'sub' / 'two.png'), photo(seed=3))

    with app.app_context():
        preview = banks.relocate_preview('local', bank_id, str(dest))
    expected = 2 if os.path.normcase('A') == os.path.normcase('a') else 0
    assert preview['found'] == expected, preview


def test_relocate_rejects_an_unknown_folder(client, tmp_path):
    bank_id, _src = _mkbank(client, tmp_path, ['a.png'])
    r = client.post(f'/api/bank/{bank_id}/relocate',
                    json={'folder': str(tmp_path / 'nope'), 'confirm': True})
    assert r.status_code == 400
    assert 'not found' in (r.get_json().get('error') or '').lower()


@pytest.mark.parametrize('folder', ['', '   '])
def test_relocate_requires_a_folder(client, tmp_path, folder):
    bank_id, _src = _mkbank(client, tmp_path, ['a.png'])
    r = client.post(f'/api/bank/{bank_id}/relocate', json={'folder': folder})
    assert r.status_code == 400
