"""🗃️ Image bank — two banks over the same files.

A bank does not own its folder, so nothing stops one bank pointing at a folder
that sits inside another bank's. Triaging is unaffected (statuses are per bank),
but 🗑 Delete rejected removes the FILES — and those files belong to the other
bank too, which simply finds them missing. The confirmation has to be able to
say that before the click, and the creation flow says it once up front.
"""
import os

from PIL import Image


def _img(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new('RGB', (64, 64), (10, 20, 30)).save(path)


def _create(client, name, folder):
    r = client.post('/api/bank/create', json={'name': name, 'folder': str(folder)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()


def test_creating_a_bank_inside_another_ones_folder_says_so(client, tmp_path):
    parent = tmp_path / 'Telegram'
    child = parent / 'Export'
    _img(str(parent / 'loose.jpg'))
    _img(str(child / 'a.jpg'))

    _create(client, 'Everything', parent)
    out = _create(client, 'One export', child)

    assert [o['name'] for o in out['overlaps']] == ['Everything']
    assert out['overlaps'][0]['relation'] == 'parent'   # it contains us


def test_a_bank_that_swallows_an_existing_one_is_flagged_too(client, tmp_path):
    parent = tmp_path / 'Telegram'
    child = parent / 'Export'
    _img(str(child / 'a.jpg'))

    _create(client, 'One export', child)
    out = _create(client, 'Everything', parent)

    assert out['overlaps'][0]['relation'] == 'child'    # it sits inside us


def test_unrelated_folders_are_not_reported(client, tmp_path):
    _img(str(tmp_path / 'one' / 'a.jpg'))
    _img(str(tmp_path / 'two' / 'b.jpg'))
    _create(client, 'One', tmp_path / 'one')
    assert _create(client, 'Two', tmp_path / 'two')['overlaps'] == []


def test_the_delete_preview_counts_the_files_the_other_bank_would_lose(
        client, tmp_path):
    """The number that matters: how many of the files about to be destroyed are
    also inventoried by another bank."""
    parent = tmp_path / 'Telegram'
    child = parent / 'Export'
    _img(str(parent / 'loose.jpg'))
    for n in ('a.jpg', 'b.jpg', 'c.jpg'):
        _img(str(child / n))

    big = _create(client, 'Everything', parent)['id']
    small = _create(client, 'One export', child)['id']

    imgs = client.get(f'/api/bank/{small}/images').get_json()['images']
    doomed = [i['id'] for i in imgs if i['name'] in ('a.jpg', 'b.jpg')]
    client.post(f'/api/bank/{small}/images/status',
                json={'ids': doomed, 'status': 'reject'})

    out = client.get(f'/api/bank/{small}/delete-rejected/preview').get_json()
    assert out['rejected'] == 2
    assert len(out['shared']) == 1
    assert out['shared'][0]['id'] == big
    assert out['shared'][0]['name'] == 'Everything'
    assert out['shared'][0]['files'] == 2       # both doomed files are ITS files too
    assert out['mode'] in ('trash', 'app_trash', 'delete')


def test_the_preview_reports_nothing_shared_for_a_lone_bank(client, tmp_path):
    _img(str(tmp_path / 'solo' / 'a.jpg'))
    bank_id = _create(client, 'Solo', tmp_path / 'solo')['id']
    imgs = client.get(f'/api/bank/{bank_id}/images').get_json()['images']
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': [imgs[0]['id']], 'status': 'reject'})

    out = client.get(f'/api/bank/{bank_id}/delete-rejected/preview').get_json()
    assert out['rejected'] == 1 and out['shared'] == []


def test_the_preview_404s_on_an_unknown_bank(client):
    assert client.get('/api/bank/999/delete-rejected/preview').status_code == 404
