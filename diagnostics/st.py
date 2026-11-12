"""Minimal safetensors reader written from the file format: 8-byte little-endian header length,
JSON header, then raw tensor bytes addressed by `data_offsets` relative to the end of the header."""
import glob
import json
import os
import struct

import numpy as np


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return n, header


def index_snapshot(snapshot):
    """name -> dict(file, header_len, dtype, shape, data_offsets) over all shards."""
    out = {}
    for path in sorted(glob.glob(os.path.join(snapshot, "model-*.safetensors"))):
        n, header = read_header(path)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            out[name] = {"file": os.path.basename(path), "path": path, "header_len": n,
                         "dtype": info["dtype"], "shape": info["shape"],
                         "data_offsets": info["data_offsets"]}
    return out


def read_bytes(entry):
    b, e = entry["data_offsets"]
    with open(entry["path"], "rb") as f:
        f.seek(8 + entry["header_len"] + b)
        buf = f.read(e - b)
    assert len(buf) == e - b
    return np.frombuffer(buf, dtype=np.uint8).reshape(entry["shape"])
