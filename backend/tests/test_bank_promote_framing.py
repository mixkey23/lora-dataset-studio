"""Bug report from axelf_ (Discord): images uploaded to a 🗃️ Bank, analysed, then
promoted into a dataset showed "Composition (0)" in 📸 Add images — the count only
came alive while a generation was running (the in-flight rows DO carry a framing)
and fell back to 0 the moment it was stopped (those rows are deleted).

Root cause: the composition tally only counts rows that HAVE a framing, and the
promotion dropped the framing the bank's own classify pass had already written.
These tests replay the scenario end to end, with NO generation involved.
"""
import io
import os

from PIL import Image


def _save(path, im):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path)


def _blocks(value, seed, size=256):
    """Distinct-dHash image (checker offset by ``seed``) carrying a top-left pixel
    marker the mocked classifier keys off — flat fills, or two patterns sharing a
    dHash, would perceptual-dedupe on promotion and silently shrink the dataset."""
    im = Image.new('L', (size, size))
    im.putdata([255 if (((x // 32) + (y // 32) * 3 + seed) % 4) < 2 else 0
                for y in range(size) for x in range(size)])
    im = im.convert('RGB')
    im.putpixel((0, 0), (value, value, value))
    return im


def _mkbank(client, tmp_path, files, name='B'):
    src = tmp_path / 'src'
    for rel, im in files.items():
        _save(str(src / rel), im)
    r = client.post('/api/bank/create', json={'name': name, 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id']


def _images(client, bank_id):
    return {i['name']: i for i in client.get(f'/api/bank/{bank_id}/images').get_json()['images']}


def _mock_framing(monkeypatch, verdict_by_pixel):
    from app import capabilities
    from app.services import vision_ollama
    monkeypatch.setattr(capabilities, 'probe_ollama_model', lambda *a, **k: {'ok': True})

    def fake_describe(image_bytes, *a, **k):
        v = Image.open(io.BytesIO(image_bytes)).convert('L').getpixel((0, 0))
        return verdict_by_pixel.get(v, '')

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)


def _bank_to_dataset(client, app, tmp_path, monkeypatch, files, verdicts):
    bank_id = _mkbank(client, tmp_path, files)
    _mock_framing(monkeypatch, verdicts)
    r = client.post(f'/api/bank/{bank_id}/framing', json={})
    assert r.status_code == 202, r.get_json()
    assert client.get(f'/api/bank/{bank_id}').get_json()['activity']['error'] is None

    by = _images(client, bank_id)
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': [i['id'] for i in by.values()], 'status': 'keep'})
    with app.app_context():
        from app.services import face_dataset_service as svc
        ds_id = svc.create_dataset('local', 'From bank', 'bnk').id
    r = client.post(f'/api/bank/{bank_id}/promote', json={'dataset_id': ds_id})
    assert r.status_code == 202, r.get_json()
    assert client.get(f'/api/bank/{bank_id}').get_json()['activity']['error'] is None
    return bank_id, ds_id


def test_promoted_images_land_counted_in_the_composition(client, tmp_path, app, monkeypatch):
    """axelf_'s exact path: bank → analyse (framing) → promote. The dataset payload
    the 📸 Add images bar reads must show the real counts, with nothing generating."""
    _bank, ds_id = _bank_to_dataset(
        client, app, tmp_path, monkeypatch,
        {'a.png': _blocks(128, seed=0), 'b.png': _blocks(60, seed=1),
         'c.png': _blocks(90, seed=2)},
        {128: '{"framing":"face","angle":"front"}',
         60: '{"framing":"body","angle":"three-quarter"}',
         90: '{"framing":"bust","angle":"front"}'})

    comp = client.get(f'/api/dataset/{ds_id}').get_json()['composition']
    assert comp == {'face': 1, 'bust': 1, 'body': 1, 'back': 0}
    assert sum(comp.values()) == 3      # the "Composition (N)" header — never 0 here


def test_unclassified_bank_images_stay_null_for_the_dataset_classifier(
        client, tmp_path, app, monkeypatch):
    """An 'unknown' verdict (or an image the bank never analysed) must NOT be
    invented into a bucket: the row stays NULL so a later classify can fill it."""
    _bank, ds_id = _bank_to_dataset(
        client, app, tmp_path, monkeypatch,
        {'a.png': _blocks(128, seed=0), 'b.png': _blocks(60, seed=1)},
        {128: '{"framing":"face","angle":"front"}',
         60: '{"framing":"nonsense"}'})

    comp = client.get(f'/api/dataset/{ds_id}').get_json()['composition']
    assert comp == {'face': 1, 'bust': 0, 'body': 0, 'back': 0}
    with app.app_context():
        from app.models import FaceDatasetImage
        framings = sorted(
            (r.framing or 'NULL')
            for r in FaceDatasetImage.query.filter_by(dataset_id=ds_id).all())
    assert framings == ['NULL', 'face']
