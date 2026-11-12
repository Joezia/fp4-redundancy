"""Explicit native component formats and input checks."""

from ..types import Support

FORMATS = {"fp4_packed_low_first": 4, "fp4_packed_high_first": 4,
           "F8_E4M3": 8, "F8_E5M2": 8, "F8_E8M0": 8, "BF16": 16,
           "F32": 8}  # User-selected F32 baseline: original bytes, not 32-bit symbols.
FIELDS = {"F8_E4M3": (4, 3), "F8_E5M2": (5, 2), "BF16": (8, 7)}


def support(target, parameters):
    if parameters:
        return Support(False, "No runtime parameters; select an implemented method id")
    if not target.components or len({c.id for c in target.components}) != len(target.components):
        return Support(False, "Nonempty, unique components are required")
    for component in target.components:
        width = FORMATS.get(component.storage_format)
        if width is None:
            return Support(False, f"Unsupported explicit storage format: {component.storage_format}")
        if (component.role == "packed_fp4") != (width == 4):
            return Support(False, "Packed FP4 needs an explicit nibble order and packed_fp4 role")
        alignment = 4 if component.storage_format == "F32" else max(1, width // 8)
        if component.length <= 0 or component.length % alignment:
            return Support(False, "Component length does not match its storage width")
    return Support(True, "Each original component has its own independent model and stream")

