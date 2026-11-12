"""Lazy contiguous chunks, preserving each original target and scale ownership."""

from dataclasses import replace

from .types import Target

DEFAULT_CHUNK_BYTES = 1 << 20


def _chunk(target, index, ranges, shape, rows=None):
    parts = tuple(replace(c, offset=c.offset + ranges[c.id][0], length=ranges[c.id][1])
                  for c in target.components)
    descriptor = {"parent_target_id": target.id, "parent_logical_shape": list(target.logical_shape),
                  "index": index, "rows": rows,
                  "ranges": {c.id: {"offset": ranges[c.id][0], "length": ranges[c.id][1],
                                    "parent_bytes": c.length} for c in target.components}}
    structure = {**target.structure, "chunk": descriptor}
    inference = {**target.inference, "chunk_boundary": "offline_storage_only"}
    return Target(f"{target.id}#chunk{index}", tuple(shape), parts, structure, inference)


def iter_target_chunks(target, max_bytes=DEFAULT_CHUNK_BYTES):
    """Cap total raw component bytes per chunk, not the Python process's RSS.

    Quantized 2-D weights split by rows: 1-row groups for FP4/Engram, 32-row
    groups for ordinary FP8. Each companion scale byte belongs to exactly one
    chunk. Unscaled tensors split on native scalar boundaries. No data is read.
    """
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("chunk bytes must be a positive integer")
    if not target.components or any(c.length <= 0 for c in target.components):
        raise ValueError("Nonempty components are required for chunking")
    if target.original_bytes <= max_bytes:
        yield _chunk(target, 0, {c.id: (0, c.length) for c in target.components}, target.logical_shape)
        return
    block = target.structure.get("scale_block_shape")
    if block is not None:
        parts = {c.id: c for c in target.components}
        if set(parts) != {"weight", "scale"} or len(target.logical_shape) != 2 or block not in ([1, 32], [32, 32]):
            raise ValueError("Unsupported weight/scale chunk layout")
        weight, scale = parts["weight"], parts["scale"]
        rows, columns = target.logical_shape
        row_bytes = columns // 2 if weight.role == "packed_fp4" else columns
        scale_columns = (columns + 31) // 32
        group_rows = block[0]
        if (weight.length != rows * row_bytes or scale.storage_format != "F8_E8M0"
                or scale.length != ((rows + group_rows - 1) // group_rows) * scale_columns):
            raise ValueError("Declared component sizes do not match scale layout")
        minimum = min(rows, group_rows) * row_bytes + scale_columns
        if max_bytes < minimum:
            raise ValueError(f"chunk bytes too small for one scale-aligned row group; need {minimum}")
        groups = max(1, max_bytes // (group_rows * row_bytes + scale_columns))
        step = groups * group_rows
        for index, start in enumerate(range(0, rows, step)):
            end = min(start + step, rows)
            scale_start, scale_end = start // group_rows, (end + group_rows - 1) // group_rows
            ranges = {"weight": (start * row_bytes, (end - start) * row_bytes),
                      "scale": (scale_start * scale_columns, (scale_end - scale_start) * scale_columns)}
            yield _chunk(target, index, ranges, (end - start, columns), [start, end])
        return
    if len(target.components) != 1:
        raise ValueError("Companion scale layout is required before splitting a multi-component target")
    component = target.components[0]
    alignment = {"BF16": 2, "F32": 4}.get(component.storage_format, 1)
    if component.length % alignment or max_bytes < alignment:
        raise ValueError("chunk bytes/length cannot preserve native scalar boundaries")
    step = max_bytes // alignment * alignment
    for index, start in enumerate(range(0, component.length, step)):
        length = min(step, component.length - start)
        count = length * 2 if component.role == "packed_fp4" else length // alignment
        yield _chunk(target, index, {component.id: (start, length)}, (count,))
