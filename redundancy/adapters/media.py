"""Media adapters for the incremental-saving module (spec 008 Part 2).

Each dataset lists its items per split and loads one item as an integer array:
  audio  int64 (channels, samples), signed 16-bit PCM      raw bits = 16 per sample
  image  uint8 (H, W, 3), 8-bit RGB                         raw bits = 8 per channel value
The data files stay outside git (data/008/...); only names, sizes and checksums are recorded.
"""
import glob
import os

import numpy as np

ROOT = os.environ.get("MEDIA_ROOT", "data/008")


def _track_no(path):
    return int(os.path.splitext(os.path.basename(path))[0])


def items(dataset, split):
    if dataset == "kodak":
        fs = sorted(glob.glob(os.path.join(ROOT, "kodak", "kodim*.png")))
        no = {f: int(os.path.basename(f)[5:7]) for f in fs}
        if split == "dev":
            return [f for f in fs if no[f] <= 12]
        if split == "report":
            return [f for f in fs if no[f] >= 13]
        return fs
    if dataset == "sqam":
        fs = sorted(glob.glob(os.path.join(ROOT, "sqam", "flac", "*.flac")), key=_track_no)
        if split == "dev":
            return [f for f in fs if _track_no(f) % 2 == 1]
        if split == "report":
            return [f for f in fs if _track_no(f) % 2 == 0]
        return fs
    if dataset == "librispeech":
        sub = {"dev": "dev-clean", "report": "test-clean"}[split]
        return sorted(glob.glob(os.path.join(ROOT, "librispeech", "LibriSpeech", sub, "*", "*", "*.flac")))
    raise KeyError(dataset)


def kind(dataset):
    return "image" if dataset == "kodak" else "audio"


def load(dataset, path, max_seconds=None):
    """Return (array, meta)."""
    if kind(dataset) == "image":
        from PIL import Image
        im = Image.open(path)
        img = np.asarray(im.convert("RGB"))
        return img, {"shape": list(img.shape), "mode": im.mode}
    import soundfile as sf
    info = sf.info(path)
    assert info.subtype == "PCM_16", (path, info.subtype)
    frames = None if not max_seconds else int(max_seconds * info.samplerate)
    x, sr = sf.read(path, dtype="int16", always_2d=True, frames=frames if frames else -1)
    x = x.T.astype(np.int64)
    return x, {"samplerate": sr, "channels": int(x.shape[0]), "samples": int(x.shape[1]),
               "seconds_used": x.shape[1] / sr, "seconds_total": info.duration}


def raw_bits(dataset, arr):
    return float(arr.size * (8 if kind(dataset) == "image" else 16))


def codec_bits(dataset, arr, meta):
    """Size in bits of a real lossless codec on exactly the coded data: FLAC at its highest level (-8,
    libsndfile compression level 1.0) for audio; PNG at zlib level 9 with optimisation for images.
    Container headers are included."""
    import io
    if kind(dataset) == "image":
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG", compress_level=9, optimize=True)
        return {"PNG (level 9, optimize)": 8.0 * buf.tell()}
    import tempfile
    import soundfile as sf
    tmp = os.path.join(ROOT, "tmp")
    os.makedirs(tmp, exist_ok=True)
    fd, p = tempfile.mkstemp(suffix=".flac", dir=tmp)
    os.close(fd)
    try:
        # written to a file: libsndfile's FLAC writer does not finish the stream in an in-memory buffer
        sf.write(p, arr.T.astype(np.int16), meta["samplerate"], format="FLAC", subtype="PCM_16", compression_level=1.0)
        size = os.path.getsize(p)
        back, _ = sf.read(p, dtype="int16", always_2d=True)
        assert np.array_equal(back.T.astype(np.int64), arr), "FLAC round trip differs"
    finally:
        os.remove(p)
    return {"FLAC -8": 8.0 * size}
