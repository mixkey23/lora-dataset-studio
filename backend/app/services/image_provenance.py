"""Signals about what an image really IS, as opposed to what its header claims —
the 🗃️ bank's answer to "this file says 2048 px, but is there 2048 px of picture
in it?" and "was this made by a camera or by a generator?".

Everything here is **pure Pillow**: no numpy, no cv2, no model. The bank's
quality pass runs on every install, extras or not (see ``image_quality``), and a
signal that only works for the subset of users who installed the ML extras is
half a feature. It also has to survive tens of thousands of images, so every
measurement below is bounded and was timed on a real 36 000-image bank.

Four signals, each raw — thresholds live in config ('bank' section) and are
applied at read time, so retuning one re-sorts the bank without a rescan.

**detail_ratio** — the effective resolution, 0..1 of the stored size.
    An image enlarged from 512 to 2048 carries no picture above the frequency it
    had at 512: re-deriving it from a 1/4-size copy loses almost nothing. So the
    measure is literally that question, asked at a ladder of scales: shrink the
    image to r, blow it back up, and see how much of it comes back. The largest r
    that still reconstructs the tile is its effective resolution. A ratio of 1.0
    means the pixels are all earning their keep; 0.5 means half the stored width
    is interpolation.

    ⚠️ This is a SCORE, never a verdict, and it is NOT `blur_score`. Sharpness
    (Laplacian variance) answers "is this crisp?" and moves with contrast and
    content; detail_ratio answers "at what SIZE does the picture stop?" and is a
    scale. But it cannot tell an enlargement from a soft photograph: motion blur,
    a background thrown out of focus and an aggressive denoiser all genuinely
    remove the same high frequencies. Which is fine for the job it is hired for —
    a LoRA learns just as little from either — as long as the UI says "real
    detail", not "upscaled".

    Measured limits, honestly (synthetic ground truth: 40 real bank photos shrunk
    and blown back up, then re-encoded at JPEG q88; medians):

        true 1/1   -> 1.00     true 1/3 -> 0.65     true 1/6 -> 0.49
        true 1/1.5 -> 1.00     true 1/4 -> 0.55     true 1/8 -> 0.50
        true 1/2   -> 0.80

    So the reading is monotone but COMPRESSED and biased high, and it SATURATES
    around 0.5: beyond ~4x every enlargement looks the same. It separates 3x+
    cleanly, catches only half of an exact 2x, and sees a 1.5x not at all.
    Callers must treat the number as a rank, not as a measurement of the original
    file's size (frontend bankProvenance.js carries the calibration that converts
    it back into an honest pixel figure).

    Two blind spots worth knowing:
      * NEAREST-neighbour enlargement is invisible. Blocky pixels ARE real high
        frequencies, so a nearest 8x upscale reads 1.0. It is also the one kind
        that is obvious to the eye, which is some consolation.
      * A crisp overlay on an otherwise enlarged image (a hard-edged watermark,
        burnt-in subtitles, UI chrome) is native-resolution content, and the
        max-over-tiles rule will rightly say "there is real detail here". On a
        photograph the overlay is a small part of a tile and changes nothing
        measurable, but on very clean synthetic content — where the enlarged
        tiles carry so little detail that they abstain — the overlay can end up
        being the ONLY tile that votes.

**origin** — 'ai' | 'camera' | 'unknown', and NEVER only two.
    A ComfyUI PNG carries its whole workflow in a `prompt` text chunk; A1111
    writes `parameters`; the IPTC/C2PA standard marks generated pixels with
    `trainedAlgorithmicMedia` in XMP. A camera writes Make/Model/exposure in
    EXIF. Both are certain when present — and USUALLY ABSENT: scrapers, chat apps
    and social networks strip metadata on sight. Measured on a real 36 000-image
    Telegram bank: 3000/3000 images carried nothing at all. So silence means
    'unknown' and must never be read as "not AI"; that inversion would be wrong
    on the overwhelming majority of files and is the central trap of this
    feature.

    Only a short evidence TOKEN is kept ('png-prompt', 'exif-camera', ...) — never
    the prompt itself, which is user content and often long.

**bars_ratio** — how much of the frame is a flat black letterbox/pillarbox.
    A phone screenshot of a video, a 16:9 still padded into a square. Cheap, and
    those bars poison a crop-based training pipeline.

**jpeg_quality** — the quality the last JPEG save used, recovered from the
    quantization tables. Free (the tables are already parsed by the decoder). It
    is a FACT about the file, not a flag: on the reference bank it is cleanly
    bimodal (~72 and ~87, the two profiles a chat app re-encodes with), and any
    threshold that caught the low mode would catch half the bank.
"""
from PIL import Image, ImageChops, ImageStat

try:                                    # Pillow >= 6; guarded so an ancient one
    from PIL import ExifTags            # degrades to "no camera evidence"
    _TAG_ID = {v: k for k, v in ExifTags.TAGS.items()}
except Exception:                       # pragma: no cover - very old Pillow
    _TAG_ID = {}

# --- effective resolution ---------------------------------------------------
# Native-resolution tiles: cropping (never resizing) is what keeps the real
# frequencies intact — a global downscale to a fixed working size would itself
# band-limit the image and erase the very evidence we are looking for.
_TILE = 512
# ...then ONE box-halving of each tile. JPEG/WEBP grain is high-frequency noise
# that no scale can reconstruct, so it drags every rung of the ladder up towards
# 1.0 and hides real enlargements (an 8x enlargement read as "full" before this
# step). Halving averages the grain away, costs 4x less, and only gives up the
# top octave — which we do not need: the point is to catch 2x+ enlargements.
_PRE = 2
_LADDER = (0.85, 0.7, 0.55, 0.42, 0.3, 0.2)
# Fraction of the tile's total detail that may go missing and still count as
# "reconstructed". Calibrated on synthetic ground truth (see module docstring).
_EPS = 0.22
# A tile with less detail than this says nothing about resolution (flat sky,
# black frame). Its vote is dropped rather than counted as "no detail = tiny".
# Set well above the codec noise floor on purpose: when a tile has almost nothing
# left, the residual curve is FLAT (every rung reconstructs the noise equally
# badly) and the reading comes out a confident 1.0 — the one wrong answer that
# matters, since it certifies an enlargement as full-resolution. Abstaining is
# the honest reply. Measured: raising this from 0.8 to 2.0 changes nothing at all
# on 400 real bank images (identical none/flag counts) nor on ground truth at
# 1/2, 1/4 and 1/8, so the caution is free.
_MIN_DETAIL = 2.0
# Five probe tiles, not three. The tiles vote by MAXIMUM (see detail_ratio), so
# every extra tile is another chance to find genuine detail and NOT cry wolf:
# measured on 400 real bank images, the share reading below 0.72 drops from 3.0%
# (3 tiles) to 1.75% (5 tiles), and the 10th percentile of images known to be
# full-resolution rises from 0.78 to 0.84. Costs ~6 ms more per image.
_TILE_VOTES = 5

# --- letterbox --------------------------------------------------------------
_BARS_PROBE = 256
_BARS_DARK = 22                          # 0..255; above this a row is not "black"

# --- origin -----------------------------------------------------------------
# PNG/WEBP text chunks that only a generator writes. ComfyUI: prompt + workflow.
# A1111/Forge/Fooocus: parameters. InvokeAI: invokeai_metadata / sd-metadata.
_AI_TEXT_KEYS = ('prompt', 'workflow', 'parameters', 'sd-metadata',
                 'invokeai_metadata', 'Dream')
# Substrings of an EXIF/PNG `Software` value that name a generator. Lowercased
# compare. Deliberately NOT including plain editors (Photoshop, GIMP): retouching
# a photograph does not make it generated.
_AI_SOFTWARE = ('comfyui', 'stable diffusion', 'stable-diffusion', 'automatic1111',
                'sd-webui', 'invokeai', 'novelai', 'midjourney', 'dall-e', 'dalle',
                'fooocus', 'adobe firefly', 'imagen', 'flux.1', 'grok-imagine')
# The IPTC "digital source type" the C2PA/XMP standard uses to mark generated
# pixels. This is the one metadata marker the commercial generators agreed on.
_AI_XMP = ('trainedalgorithmicmedia', 'compositewithtrainedalgorithmicmedia')
# EXIF tags only an actual capture device writes.
_CAM_TAGS = ('Make', 'Model', 'ExposureTime', 'FNumber', 'ISOSpeedRatings',
             'FocalLength', 'LensModel')
_EXIF_IFD = 0x8769                       # ExifOffset — where exposure data lives

ORIGINS = ('ai', 'camera', 'unknown')


def _rms(im) -> float:
    """RMS of a difference image, straight off its histogram (no numpy)."""
    hist = im.histogram()[:256]
    n = sum(hist)
    if not n:
        return 0.0
    return (sum(v * v * c for v, c in enumerate(hist)) / n) ** 0.5


def _tiles(g, tile=_TILE, votes=_TILE_VOTES):
    """Up to `votes` native-resolution tiles, richest first.

    A 3x3 probe grid ranked by standard deviation: an enlargement is only
    provable where there is something to enlarge, so the busiest tiles vote and
    the empty ones do not. Off-centre positions are included because a portrait's
    detail (hair, fabric, eyes) is rarely dead centre."""
    w, h = g.size
    if w <= tile and h <= tile:
        return [g]
    tw, th = min(tile, w), min(tile, h)
    cand = []
    for fy in (0.5, 0.15, 0.85):
        for fx in (0.5, 0.2, 0.8):
            x = int(round((w - tw) * fx))
            y = int(round((h - th) * fy))
            c = g.crop((x, y, x + tw, y + th))
            cand.append((ImageStat.Stat(c).stddev[0], c))
    cand.sort(key=lambda t: -t[0])
    return [c for _s, c in cand[:votes]]


def _tile_ratio(t):
    """Effective resolution of ONE tile, or None when the tile is too flat."""
    w, h = t.size
    if _PRE > 1:
        t = t.resize((max(8, w // _PRE), max(8, h // _PRE)), Image.BOX)
        w, h = t.size
    if w < 16 or h < 16:
        return None
    resid = []
    for r in _LADDER:
        sw, sh = max(1, round(w * r)), max(1, round(h * r))
        back = t.resize((sw, sh), Image.LANCZOS).resize((w, h), Image.BICUBIC)
        resid.append(_rms(ImageChops.difference(t, back)))
    total = resid[-1]                     # coarsest rung = the most detail lost
    if total < _MIN_DETAIL:
        return None
    norm = [v / total for v in resid]
    if norm[0] >= _EPS:
        return 1.0                        # detail all the way to the top rung
    # Walk down until reconstruction starts failing, and interpolate the crossing
    # so the score is continuous rather than snapped to a rung.
    prev_r, prev_v = 1.0, 1.0
    for r, v in zip(_LADDER, norm):
        if v >= _EPS:
            f = (_EPS - prev_v) / (v - prev_v) if v != prev_v else 0.0
            return max(0.05, min(1.0, prev_r + f * (r - prev_r)))
        prev_r, prev_v = r, v
    return _LADDER[-1]


def detail_ratio(im):
    """Effective resolution as a fraction of the stored size, or None when the
    image carries too little detail anywhere to have an opinion.

    The tiles vote by MAXIMUM, not by average: real detail found in ONE place
    proves the resolution is real, while a blurred background proves nothing."""
    try:
        g = im.convert('L')
    except (OSError, ValueError):
        return None
    best = None
    for t in _tiles(g):
        r = _tile_ratio(t)
        if r is None:
            continue
        best = r if best is None else max(best, r)
    return None if best is None else round(best, 4)


def bars_ratio(im):
    """Fraction of the frame taken by flat black bars (top+bottom, left+right —
    whichever is larger), or None if it cannot be measured.

    Scanned on a small box-reduced copy: bars are a macro feature, and reducing
    first both bounds the cost and stops a single stray bright pixel (a codec
    artefact in the padding) from ending the scan one row early."""
    try:
        g = im.convert('L')
        g.thumbnail((_BARS_PROBE, _BARS_PROBE), Image.BOX)
    except (OSError, ValueError):
        return None
    w, h = g.size
    if w < 8 or h < 8:
        return None
    px = g.load()
    xs = range(0, w, max(1, w // 48))
    ys = range(0, h, max(1, h // 48))

    def row_dark(y):
        return max(px[x, y] for x in xs) <= _BARS_DARK

    def col_dark(x):
        return max(px[x, y] for y in ys) <= _BARS_DARK

    def run(n, dark):
        i = 0
        while i < n // 2 and dark(i):
            i += 1
        return i

    top = run(h, row_dark)
    bottom = run(h, lambda i: row_dark(h - 1 - i))
    left = run(w, col_dark)
    right = run(w, lambda i: col_dark(w - 1 - i))
    return round(max((top + bottom) / h, (left + right) / w), 4)


# libjpeg's baseline luminance table. A JPEG stores the SCALED table, so the
# scale factor — and from it the quality the encoder was asked for — divides back
# out. Chroma is ignored: subsampling makes it a much noisier estimate.
_STD_LUMA = (16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
             14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
             18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
             49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99)


def jpeg_quality(im):
    """The quality of the last JPEG save, 1..100, or None for a non-JPEG (or a
    JPEG with a custom table we refuse to guess at).

    Entries pinned at 1 or 255 have hit the clamp and carry no scale information,
    so they are dropped; the MEDIAN of the rest resists a table that was tweaked
    in a few cells."""
    tables = getattr(im, 'quantization', None)
    if not tables or 0 not in tables:
        return None
    t = list(tables[0])[:64]
    if len(t) < 64:
        return None
    scales = [v * 100.0 / s for v, s in zip(t, _STD_LUMA) if 1 < v < 255]
    if not scales:
        return 100.0                      # table of 1s = quality 100 (or lossless-ish)
    s = sorted(scales)[len(scales) // 2]
    q = (200.0 - s) / 2.0 if s < 100.0 else 5000.0 / s
    return round(max(1.0, min(100.0, q)), 1)


def _exif_camera_tags(im):
    """{tag: value} for the capture tags present, main IFD and Exif sub-IFD."""
    if not _TAG_ID:
        return {}
    try:
        exif = im.getexif()
    except Exception:
        return {}
    if not exif:
        return {}
    out = {}
    sources = [exif]
    try:
        sub = exif.get_ifd(_EXIF_IFD)
        if sub:
            sources.append(sub)
    except Exception:
        pass
    for src in sources:
        for tag in _CAM_TAGS:
            tid = _TAG_ID.get(tag)
            if tid is None:
                continue
            try:
                val = src.get(tid)
            except Exception:
                continue
            if val not in (None, '', b''):
                out[tag] = val
    return out


def _as_text(val):
    if isinstance(val, bytes):
        return val.decode('utf-8', 'replace')
    return val if isinstance(val, str) else ''


def origin(im):
    """(origin, evidence) — one of 'ai' / 'camera' / 'unknown', plus a SHORT
    token naming what was found (never the metadata's content).

    AI evidence wins over camera evidence: a generator that also copied an EXIF
    Make into its output is still a generator, whereas nothing a camera writes
    can produce a workflow chunk. Absence of everything is 'unknown' — see the
    module docstring for why that must never collapse into "not AI"."""
    info = {}
    try:
        info = dict(im.info or {})
    except Exception:
        pass
    for key in _AI_TEXT_KEYS:
        val = info.get(key)
        if isinstance(val, (str, bytes)) and val:
            return 'ai', f'png-{key.lower()}'
    xmp = _as_text(info.get('XML:com.adobe.xmp') or info.get('xmp') or '').lower()
    if any(marker in xmp for marker in _AI_XMP):
        return 'ai', 'xmp-ai-source'
    software = _as_text(info.get('Software'))
    cam = _exif_camera_tags(im)
    if not software and _TAG_ID:
        try:
            software = _as_text(im.getexif().get(_TAG_ID.get('Software')))
        except Exception:
            software = ''
    low = software.lower()
    if any(name in low for name in _AI_SOFTWARE):
        return 'ai', 'software-tag'
    if cam:
        return 'camera', 'exif-camera'
    return 'unknown', None


def provenance_metrics(im) -> dict:
    """Every signal of this module for one decoded image, as stored on the row.

    Never raises: a signal that cannot be measured comes back None and the image
    keeps every other verdict it earned. A 36 000-image pass must not die on one
    exotic file."""
    out = {'detail_ratio': None, 'bars_ratio': None,
           'jpeg_quality': None, 'origin': 'unknown', 'origin_evidence': None}
    try:
        out['detail_ratio'] = detail_ratio(im)
    except Exception:
        pass
    try:
        out['bars_ratio'] = bars_ratio(im)
    except Exception:
        pass
    try:
        out['jpeg_quality'] = jpeg_quality(im)
    except Exception:
        pass
    try:
        out['origin'], out['origin_evidence'] = origin(im)
    except Exception:
        pass
    return out
