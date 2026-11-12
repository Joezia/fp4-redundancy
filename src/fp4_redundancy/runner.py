"""Bounded, byte-exact evaluation and matched zero-order comparisons."""

import hashlib
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from types import MappingProxyType

from .chunks import DEFAULT_CHUNK_BYTES, iter_target_chunks
from .types import ResourceLimitError


def evaluate_target(registry, method_id, target, parameters=None, *, max_input_bytes=DEFAULT_CHUNK_BYTES,
                    components=None, artifact_observer=None, baseline_observer=None, _baseline_cache=None):
    """One technique against one target; every component must round-trip exactly.

    This boundary reads one bounded unit. Use evaluate_target_chunks for large
    tensors. It never aggregates subset results as a full model.
    """
    parameters = {} if parameters is None else parameters
    result = {"schema_version": "1.0", "target_id": target.id, "method_id": method_id,
              "parameters": parameters, "structure": dict(target.structure),
              "inference": dict(target.inference), "status": "not_implemented",
              "reason": "No registered implementation", "original_bytes": target.original_bytes,
              "payload_bytes": None, "side_information_bytes": None, "total_bytes": None,
              "saving_bytes": None, "saving_fraction": None, "baseline_comparison": None, "bitwise_verified": False,
              "encode_seconds": None, "decode_seconds": None, "source_sha256": None}
    if method_id not in registry.ids():
        return result
    try:
        method = registry.create(method_id)
        result["method"] = asdict(method.spec)
        support = method.check_support(target, parameters)
        if not support.applicable:
            result.update(status="inapplicable", reason=support.reason)
            return result
        if target.original_bytes > max_input_bytes:
            result.update(status="inapplicable", reason="Input exceeds byte limit; use evaluate_target_chunks")
            return result
        source = {} if components is None else dict(components)
        if components is None:
            for component in target.components:
                with Path(component.path).open("rb") as stream:
                    stream.seek(component.offset)
                    source[component.id] = stream.read(component.length)
        if set(source) != {c.id for c in target.components}:
            raise ValueError("Source component set differs from target")
        for component in target.components:
            if not isinstance(source[component.id], bytes) or len(source[component.id]) != component.length:
                raise ValueError(f"Truncated/invalid component: {component.id}")
        result["source_sha256"] = {key: hashlib.sha256(data).hexdigest()
                                   for key, data in source.items()}
        before = perf_counter()
        artifact = method.encode(target, MappingProxyType(source), parameters)
        result["encode_seconds"] = perf_counter() - before
        # A new instance prevents an encoder's cached originals from passing verification.
        decoder = registry.create(method_id)
        before = perf_counter()
        decoded = decoder.decode(artifact)
        result["decode_seconds"] = perf_counter() - before
        if dict(decoded) != source:
            result.update(status="verification_failed", reason="Decoded components differ")
            return result
        if artifact_observer is not None:
            artifact_observer(artifact)
        result.update(status="verified", reason="All original components restored byte for byte",
                      payload_bytes=len(artifact.payload),
                      side_information_bytes=len(artifact.side_information),
                      total_bytes=artifact.total_bytes,
                      saving_bytes=target.original_bytes - artifact.total_bytes,
                      saving_fraction=1 - artifact.total_bytes / target.original_bytes,
                      bitwise_verified=True)
        cache = {} if _baseline_cache is None else _baseline_cache
        cache[method_id] = dict(result)
        baseline_id = method.spec.baseline_method_id
        if baseline_id:
            if baseline_id == method_id:
                raise ValueError("A method cannot be its own zero-order baseline")
            baseline = cache.get(baseline_id)
            baseline_parameters = {"unit_bits": parameters["unit_bits"]} if "unit_bits" in parameters else {}
            if (baseline is None or baseline["target_id"] != target.id or
                baseline["source_sha256"] != result["source_sha256"] or baseline["parameters"] != baseline_parameters):
                baseline = evaluate_target(registry, baseline_id, target, baseline_parameters, max_input_bytes=max_input_bytes,
                                           components=source, artifact_observer=baseline_observer,
                                           _baseline_cache=cache)
                cache[baseline_id] = baseline
            paired = baseline["status"] == "verified"
            result["baseline_comparison"] = {
                "method_id": baseline_id, "status": baseline["status"], "reason": baseline["reason"],
                "kind": method.spec.comparison_kind or "probability_model",
                "parameters": baseline_parameters,
                "implementation_version": baseline.get("method", {}).get("implementation_version"),
                "encode_seconds": baseline["encode_seconds"], "decode_seconds": baseline["decode_seconds"],
                "original_bytes": baseline["original_bytes"], "payload_bytes": baseline["payload_bytes"],
                "side_information_bytes": baseline["side_information_bytes"], "total_bytes": baseline["total_bytes"],
                "net_saving_bytes": baseline["total_bytes"] - artifact.total_bytes if paired else None,
                "net_saving_fraction": 1 - artifact.total_bytes / baseline["total_bytes"] if paired else None}
    except ResourceLimitError as exc:
        result.update(status="resource_limited", reason=str(exc), payload_bytes=None, side_information_bytes=None,
                      total_bytes=None, saving_bytes=None, saving_fraction=None, baseline_comparison=None, bitwise_verified=False)
    except Exception as exc:
        result.update(status="failed", reason=f"{type(exc).__name__}: {exc}",
                      payload_bytes=None, side_information_bytes=None, total_bytes=None,
                      saving_bytes=None, saving_fraction=None, baseline_comparison=None, bitwise_verified=False)
    return result


def evaluate_target_chunks(registry, method_id, target, parameters=None, *, chunk_bytes=DEFAULT_CHUNK_BYTES):
    """Yield one verified/failed record at a time; retain no prior chunk payloads."""
    for chunk in iter_target_chunks(target, chunk_bytes):
        result = evaluate_target(registry, method_id, chunk, parameters, max_input_bytes=chunk_bytes)
        result["chunk"] = chunk.structure["chunk"]
        yield result
