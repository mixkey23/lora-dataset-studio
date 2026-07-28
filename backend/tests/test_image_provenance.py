"""Contract of the pure-PIL provenance signals (effective resolution, origin,
letterbox, JPEG quality).

The origin cases are the important ones: the three states must stay three. A
build that quietly collapses 'unknown' into "not AI" would be wrong about the
overwhelming majority of real files (measured: 3000/3000 images of a real chat
export carried no metadata at all), so `test_stripped_file_is_unknown` is a
regression guard on the whole feature's honesty, not a formality.
"""
import io

import pytest
from PIL import Image, ImageChops, PngImagePlugin

from app.services import image_provenance as prov


# --- fixtures ---------------------------------------------------------------
_FALLOFF = 1.5


def _detailed(w=256, h=256, seed=7):
    """A deterministic image with a PHOTOGRAPH-LIKE spectrum, built without numpy.

    Multi-octave value noise: random fields at 4, 8, 16 ... up to the full size,
    each smoothly upsampled and summed with a geometrically decaying weight. That
    approximates the falloff a real photo has, which is what matters here — the
    estimator's whole job is to spot where a natural falloff turns into a cliff,
    and flat white noise (a spectrum no camera ever produces) would exercise none
    of that. The TOP octave must be present at native size, or the fixture is
    itself an enlargement and every reading is meaningless.

    Sanity check on the fixture: at 1/2, 1/4 and 1/8 it reads ~0.79 / 0.56 / 0.53,
    within a few hundredths of what 900 real bank photos give for the same
    enlargements (~0.79 / 0.54 / 0.47). The numbers quoted in image_provenance's
    docstring come from those real images, not from here."""
    state = seed
    acc = Image.new('L', (w, h), 0)
    weight = 1.0
    total = 0.0
    layers = []
    n = 4
    while n <= min(w, h):
        layer = Image.new('L', (n, n))
        px = layer.load()
        for y in range(n):
            for x in range(n):
                state = (state * 1103515245 + 12345) & 0x7FFFFFFF
                px[x, y] = (state >> 16) & 0xFF
        layers.append((layer.resize((w, h), Image.BICUBIC), weight))
        total += weight
        weight /= _FALLOFF
        n *= 2
    for layer, wgt in layers:
        acc = ImageChops.add(acc, layer.point(lambda v, k=wgt / total: int(v * k)))
    return acc.convert('RGB')


def _enlarged(im, factor):
    w, h = im.size
    return (im.resize((w // factor, h // factor), Image.LANCZOS)
              .resize((w, h), Image.BICUBIC))


def _reopen(im, fmt='JPEG', **kw):
    buf = io.BytesIO()
    im.convert('RGB').save(buf, fmt, **kw)
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out


# --- effective resolution ---------------------------------------------------
def test_enlarged_image_scores_below_its_native_original():
    """The whole point: an image blown up from a quarter of its size must read
    LOWER than the same image at native resolution. The absolute value is a rank,
    not a measurement (see the module docstring), so the test asserts the
    ordering and a generous margin — never an exact figure."""
    native = _detailed()
    up = _enlarged(native, 4)
    r_native = prov.detail_ratio(native)
    r_up = prov.detail_ratio(up)
    assert r_native is not None and r_up is not None
    assert r_up < r_native
    assert r_up < 0.72          # below the shipped bank.detail_min default
    assert r_native > 0.9


def test_scores_are_monotone_in_the_enlargement_factor():
    native = _detailed()
    got = [prov.detail_ratio(_enlarged(native, k)) for k in (2, 4, 8)]
    assert all(v is not None for v in got)
    # Bigger enlargement never reads as MORE real detail.
    assert got[0] >= got[1] >= got[2]


def test_flat_image_has_no_opinion():
    """A blank frame carries no detail at any scale — it must abstain, not claim
    a tiny effective resolution (that would flag every solid-colour placeholder)."""
    assert prov.detail_ratio(Image.new('RGB', (900, 900), (128, 128, 128))) is None


def test_tiny_image_does_not_crash():
    assert prov.detail_ratio(Image.new('RGB', (3, 3), 'white')) is None


# --- origin -----------------------------------------------------------------
def test_comfyui_png_prompt_chunk_is_ai():
    info = PngImagePlugin.PngInfo()
    info.add_text('prompt', '{"3": {"class_type": "KSampler"}}')
    im = _reopen(Image.new('RGB', (64, 64), 'blue'), 'PNG', pnginfo=info)
    assert prov.origin(im) == ('ai', 'png-prompt')


def test_a1111_parameters_chunk_is_ai():
    info = PngImagePlugin.PngInfo()
    info.add_text('parameters', 'a photo\nSteps: 20, Sampler: Euler a')
    im = _reopen(Image.new('RGB', (64, 64), 'blue'), 'PNG', pnginfo=info)
    assert prov.origin(im) == ('ai', 'png-parameters')


def test_camera_exif_is_camera():
    exif = Image.Exif()
    exif[271] = 'ACME'          # Make
    exif[272] = 'Model One'     # Model
    im = _reopen(Image.new('RGB', (64, 64), 'green'), 'JPEG', exif=exif.tobytes())
    assert prov.origin(im) == ('camera', 'exif-camera')


def test_xmp_trained_algorithmic_media_is_ai():
    im = _reopen(Image.new('RGB', (64, 64), 'blue'))
    im.info['XML:com.adobe.xmp'] = (
        '<Iptc4xmpExt:digitalSourceType>'
        'http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia'
        '</Iptc4xmpExt:digitalSourceType>')
    assert prov.origin(im) == ('ai', 'xmp-ai-source')


def test_stripped_file_is_unknown_not_camera():
    """A file with no metadata is 'unknown'. It is NOT evidence of a photograph,
    and it is NOT evidence against AI — this is the trap the feature exists to
    avoid, and the state must stay its own third answer."""
    im = _reopen(Image.new('RGB', (64, 64), 'blue'))
    assert prov.origin(im) == ('unknown', None)


def test_ai_evidence_beats_camera_evidence():
    """A generator that copied an EXIF Make into its output is still a generator;
    nothing a camera does produces a workflow chunk."""
    info = PngImagePlugin.PngInfo()
    info.add_text('workflow', '{"nodes": []}')
    im = _reopen(Image.new('RGB', (64, 64), 'blue'), 'PNG', pnginfo=info)
    im.info['Make'] = 'ACME'
    assert prov.origin(im)[0] == 'ai'


def test_plain_editor_software_is_not_ai():
    """Retouching a photograph does not make it generated."""
    im = _reopen(Image.new('RGB', (64, 64), 'blue'))
    im.info['Software'] = 'Adobe Photoshop 26.0'
    assert prov.origin(im)[0] == 'unknown'


def test_evidence_token_never_carries_the_prompt():
    info = PngImagePlugin.PngInfo()
    secret = 'a very long private prompt about a named person'
    info.add_text('prompt', secret)
    im = _reopen(Image.new('RGB', (64, 64), 'blue'), 'PNG', pnginfo=info)
    _state, evidence = prov.origin(im)
    assert secret not in (evidence or '')
    assert len(evidence) <= 24          # fits the column, and is a token not data


# --- letterbox --------------------------------------------------------------
def test_letterbox_bars_are_measured():
    im = Image.new('RGB', (400, 400), 'black')
    im.paste(Image.new('RGB', (400, 200), 'white'), (0, 100))
    got = prov.bars_ratio(im)
    assert got == pytest.approx(0.5, abs=0.08)


def test_pillarbox_bars_are_measured():
    im = Image.new('RGB', (400, 400), 'black')
    im.paste(Image.new('RGB', (200, 400), 'white'), (100, 0))
    assert prov.bars_ratio(im) == pytest.approx(0.5, abs=0.08)


def test_full_frame_image_has_no_bars():
    assert prov.bars_ratio(Image.new('RGB', (400, 400), 'white')) == 0.0


# --- jpeg quality -----------------------------------------------------------
@pytest.mark.parametrize('q', [60, 75, 90])
def test_jpeg_quality_is_recovered_from_the_tables(q):
    im = _reopen(_detailed(128, 128), 'JPEG', quality=q)
    assert prov.jpeg_quality(im) == pytest.approx(q, abs=6)


def test_non_jpeg_has_no_quality():
    assert prov.jpeg_quality(_reopen(Image.new('RGB', (32, 32)), 'PNG')) is None


# --- the bundle -------------------------------------------------------------
def test_provenance_metrics_always_returns_every_key():
    got = prov.provenance_metrics(_reopen(_detailed(256, 256)))
    assert set(got) == {'detail_ratio', 'bars_ratio', 'jpeg_quality',
                        'origin', 'origin_evidence'}
    assert got['origin'] in prov.ORIGINS


def test_provenance_metrics_survives_a_broken_image():
    """One exotic file must never kill a 36 000-image pass."""
    class Exploding:
        size = (100, 100)
        info = {}

        def convert(self, _mode):
            raise OSError('broken')

    got = prov.provenance_metrics(Exploding())
    assert got['detail_ratio'] is None and got['origin'] == 'unknown'
