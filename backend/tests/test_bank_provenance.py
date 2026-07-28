"""🗃️ Bank ↔ provenance wiring: the scan persists the four signals, the flags and
filters read them back, the origin facet counts, and a bank scanned by an OLDER
build gets retrofitted instead of staying half-measured forever.

The unit behaviour of the signals themselves lives in test_image_provenance.py.
"""
import os

from PIL import Image, ImageChops, PngImagePlugin

from app.extensions import db
from app.models import BankImage
from test_image_bank import _mkbank, flat


def detailed(size=256, seed=7, falloff=1.5):
    """Photograph-like spectrum (see test_image_provenance._detailed)."""
    state = seed
    acc = Image.new('L', (size, size), 0)
    weight, total, layers, n = 1.0, 0.0, [], 4
    while n <= size:
        layer = Image.new('L', (n, n))
        px = layer.load()
        for y in range(n):
            for x in range(n):
                state = (state * 1103515245 + 12345) & 0x7FFFFFFF
                px[x, y] = (state >> 16) & 0xFF
        layers.append((layer.resize((size, size), Image.BICUBIC), weight))
        total += weight
        weight /= falloff
        n *= 2
    for layer, wgt in layers:
        acc = ImageChops.add(acc, layer.point(lambda v, k=wgt / total: int(v * k)))
    return acc.convert('RGB')


def enlarged(im, factor=4):
    w, h = im.size
    return (im.resize((w // factor, h // factor), Image.LANCZOS)
              .resize((w, h), Image.BICUBIC))


def letterboxed(size=256):
    im = Image.new('RGB', (size, size), 'black')
    im.paste(detailed(size).crop((0, 0, size, size // 2)), (0, size // 4))
    return im


def _scan(client, bank_id, rescan=False):
    r = client.post(f'/api/bank/{bank_id}/scan', json={'rescan': rescan})
    # 202: the pass is a background job (run inline under TESTING, see bank_jobs).
    assert r.status_code in (200, 202), r.get_json()
    return r.get_json()


def _rows(bank_id):
    return {os.path.basename(r.relpath): r
            for r in BankImage.query.filter_by(bank_id=bank_id).all()}


def test_scan_persists_every_provenance_signal(client, tmp_path, app):
    info = PngImagePlugin.PngInfo()
    info.add_text('prompt', '{"1": {"class_type": "KSampler"}}')
    gen = tmp_path / 'gen.png'
    os.makedirs(gen.parent, exist_ok=True)
    bank_id, _src = _mkbank(client, tmp_path, {
        'native.jpg': detailed(),
        'up.jpg': enlarged(detailed()),
        'bars.jpg': letterboxed(),
    })
    _scan(client, bank_id)
    with app.app_context():
        rows = _rows(bank_id)
        native, up = rows['native.jpg'], rows['up.jpg']
        assert native.detail_ratio is not None and up.detail_ratio is not None
        # The enlarged copy of the SAME picture must read lower.
        assert up.detail_ratio < native.detail_ratio
        # JPEG quality comes back for a JPEG, and every readable row gets one of
        # the three origin states — never NULL, never a fourth answer.
        assert 1.0 <= native.jpeg_quality <= 100.0
        assert {r.origin for r in rows.values()} <= {'ai', 'camera', 'unknown'}
        assert all(r.origin is not None for r in rows.values())
        assert rows['bars.jpg'].bars_ratio > 0.2


def test_stripped_images_are_unknown_not_camera(client, tmp_path, app):
    """The trap this feature exists to avoid: a bank of metadata-less files is a
    bank of UNKNOWNS. If any of them ever reports 'camera', the three-state
    contract is broken and users will trust a verdict that was never earned."""
    bank_id, _src = _mkbank(client, tmp_path, {f'a{i}.jpg': detailed(128, seed=i)
                                               for i in range(4)})
    _scan(client, bank_id)
    with app.app_context():
        assert {r.origin for r in _rows(bank_id).values()} == {'unknown'}


def test_comfyui_png_is_reported_as_ai(client, tmp_path, app):
    info = PngImagePlugin.PngInfo()
    info.add_text('prompt', '{"1": {"class_type": "KSampler"}}')
    src = tmp_path / 'src'
    os.makedirs(src, exist_ok=True)
    detailed(128).save(str(src / 'gen.png'), 'PNG', pnginfo=info)
    detailed(128, seed=3).save(str(src / 'plain.jpg'), 'JPEG', quality=92)
    r = client.post('/api/bank/create', json={'name': 'AI', 'folder': str(src)})
    bank_id = r.get_json()['id']
    _scan(client, bank_id)
    with app.app_context():
        rows = _rows(bank_id)
        assert rows['gen.png'].origin == 'ai'
        assert rows['gen.png'].origin_evidence == 'png-prompt'
        assert rows['plain.jpg'].origin == 'unknown'


def test_origin_facet_counts_and_filter(client, tmp_path, app):
    info = PngImagePlugin.PngInfo()
    info.add_text('parameters', 'Steps: 20')
    src = tmp_path / 'src'
    os.makedirs(src, exist_ok=True)
    detailed(128).save(str(src / 'gen.png'), 'PNG', pnginfo=info)
    for i in range(3):
        detailed(128, seed=i + 2).save(str(src / f'p{i}.jpg'), 'JPEG', quality=92)
    bank_id = client.post('/api/bank/create',
                          json={'name': 'F', 'folder': str(src)}).get_json()['id']
    _scan(client, bank_id)
    payload = client.get(f'/api/bank/{bank_id}').get_json()
    # Every state is present with a real count, 'unknown' included — it is an
    # answer, not a gap.
    assert set(payload['origins']) == {'ai', 'camera', 'unknown'}
    assert payload['origins']['ai'] == 1
    assert payload['origins']['unknown'] == 3
    assert payload['origins']['camera'] == 0
    got = client.get(f'/api/bank/{bank_id}/images?origin=ai').get_json()
    assert got['total'] == 1 and got['images'][0]['name'] == 'gen.png'
    got = client.get(f'/api/bank/{bank_id}/images?origin=unknown').get_json()
    assert got['total'] == 3


def test_soft_detail_flag_follows_the_threshold_without_a_rescan(client, tmp_path, app):
    """Raw score in, verdict at read time — the same contract as blur. Moving
    bank.detail_min must re-sort the bank with no second pass over the files."""
    from app import config as cfg
    bank_id, _src = _mkbank(client, tmp_path, {'native.jpg': detailed(),
                                               'up.jpg': enlarged(detailed())})
    _scan(client, bank_id)
    with app.app_context():
        up = _rows(bank_id)['up.jpg']
        measured = up.detail_ratio
    # A threshold under the measured value: nobody is flagged.
    cfg.save_config({'bank': {'detail_min': round(measured - 0.05, 4)}})
    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert payload['flags']['soft_detail'] == 0
    # ...and above it: the enlarged copy is, with the file untouched.
    cfg.save_config({'bank': {'detail_min': round(measured + 0.05, 4)}})
    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert payload['flags']['soft_detail'] == 1
    got = client.get(f'/api/bank/{bank_id}/images?flag=soft_detail').get_json()
    assert got['total'] == 1 and got['images'][0]['name'] == 'up.jpg'
    assert 'soft_detail' in got['images'][0]['flags']


def test_unmeasured_rows_are_never_flagged(client, tmp_path, app):
    """NULL is 'not measured', never 'below threshold'. A bank from an older
    build must not light up with flags it was never scored for."""
    from app import config as cfg
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': detailed()})
    _scan(client, bank_id)
    with app.app_context():
        for row in BankImage.query.filter_by(bank_id=bank_id).all():
            row.detail_ratio = None
            row.bars_ratio = None
        db.session.commit()
    # Thresholds that would flag EVERYTHING measurable, and neutral ones for the
    # older flags so the assertion below is about the two new signals alone.
    cfg.save_config({'bank': {'detail_min': 0.99, 'bars_max': 0.0, 'min_side': 64,
                              'sharpness_min': 0.0, 'noise_max': 9999.0,
                              'uniformity_min': 0.0}})
    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert payload['flags']['soft_detail'] == 0
    assert payload['flags']['bars'] == 0
    # ...and such a row still counts as CLEAN rather than vanishing from both chips.
    assert client.get(f'/api/bank/{bank_id}/images?flag=clean').get_json()['total'] == 1


def test_a_bank_scanned_by_an_older_build_is_retrofitted(client, tmp_path, app):
    """The rule that matters for people who already own a 36 000-image bank: a
    plain Scan (not a full rescan) picks the un-measured rows back up. Shipping a
    signal that only applies to images scanned from now on would leave existing
    banks permanently half-measured."""
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': detailed(),
                                               'b.jpg': detailed(seed=11)})
    _scan(client, bank_id)
    with app.app_context():
        for row in BankImage.query.filter_by(bank_id=bank_id).all():
            row.origin = None            # what a pre-provenance build left behind
            row.detail_ratio = None
            row.jpeg_quality = None
        db.session.commit()
    _scan(client, bank_id, rescan=False)
    with app.app_context():
        rows = BankImage.query.filter_by(bank_id=bank_id).all()
        assert all(r.origin is not None for r in rows)
        assert all(r.detail_ratio is not None for r in rows)
    # And a second plain scan now has nothing left to do — the top-up is a
    # one-shot repair, not a permanent re-scan of the whole bank.
    with app.app_context():
        from app.services.image_bank_service import _scan_pool
        assert _scan_pool(bank_id, False).count() == 0


def test_flat_images_do_not_claim_a_tiny_effective_resolution(client, tmp_path, app):
    """A blank frame abstains. It is already caught by 'uniform'; claiming it is
    also a 4x enlargement would be an invented verdict."""
    bank_id, _src = _mkbank(client, tmp_path, {'blank.jpg': flat(256)})
    _scan(client, bank_id)
    with app.app_context():
        row = _rows(bank_id)['blank.jpg']
        assert row.detail_ratio is None
        assert row.origin == 'unknown'   # still measured for origin
