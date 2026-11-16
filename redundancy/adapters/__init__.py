"""Adapters turn some kind of data into `dataset.Entry` objects.

An adapter class takes its config section and provides `entries() -> [name]` and
`load(name) -> Entry`. `load` runs inside worker processes, so it must be self-contained."""


def get(cfg):
    kind = cfg["type"]
    if kind == "fp4":
        from .fp4 import FP4
        return FP4(cfg)
    if kind == "array_file":
        from .array_file import ArrayFile
        return ArrayFile(cfg)
    if kind == "none":
        return None
    raise ValueError(f"unknown adapter type {kind}")
