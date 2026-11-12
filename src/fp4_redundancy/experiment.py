"""Resumable, bounded CPU execution. Persist states/results, discard payloads."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import fcntl
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sqlite3
import time
import zlib

from .catalog import builtin_registry
from .checkpoint import load_checkpoint
from .chunks import DEFAULT_CHUNK_BYTES, iter_target_chunks
from .runner import evaluate_target


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def _now():
    return datetime.now(timezone.utc).isoformat()


def select_chunks(model, chunk_bytes, selection):
    """Resolve explicit target/chunk pairs against the real inventory; read no payloads."""
    if not isinstance(selection, dict):
        raise ValueError("Selection must be a JSON object")
    if (selection.get("model_id") != model.model_id or selection.get("revision") != model.revision
            or selection.get("chunk_bytes") != chunk_bytes):
        raise ValueError("Selection model/revision/chunk size differs from this run")
    entries = selection.get("chunks")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Selection needs explicit nonempty chunks")
    requested = {}
    for item in entries:
        if not isinstance(item, dict) or set(item) != {"target_id", "index"}:
            raise ValueError("Selected chunks need target_id and index")
        name, index = item["target_id"], item["index"]
        if not isinstance(name, str) or type(index) is not int or index < 0:
            raise ValueError("Invalid selected target or chunk index")
        indices = requested.setdefault(name, set())
        if index in indices:
            raise ValueError("Duplicate selected chunk")
        indices.add(index)
    if set(requested) - {t.id for t in model.targets}:
        raise ValueError("Selected target is absent from checkpoint")
    result = []
    for target in model.targets:
        if target.id not in requested:
            continue
        missing = requested[target.id].copy()
        for chunk in iter_target_chunks(target, chunk_bytes):
            index = chunk.structure["chunk"]["index"]
            if index in missing:
                result.append(chunk)
                missing.remove(index)
            if not missing:
                break
        if missing:
            raise ValueError(f"Selected chunk indices outside target: {target.id}: {sorted(missing)}")
    return tuple(result)


def _states(artifact):
    header = json.loads(artifact.side_information)
    offset = 0
    for part in header["components"]:
        for stream in part["streams"]:
            size = stream["bytes"]
            if header["method"].startswith("rans"):
                stream["initial_state_hex"] = artifact.payload[offset:offset + min(4, size)].hex()
            offset += size
    return {"side_information": header, "payload_sha256": hashlib.sha256(artifact.payload).hexdigest(),
            "side_information_sha256": hashlib.sha256(artifact.side_information).hexdigest()}


def _worker(chunk, methods, chunk_bytes, keep_states, method_parameters=None):
    """One read per component, then every requested complete method independently."""
    registry, source, outcomes, states, baseline_cache = builtin_registry(), {}, [], {}, {}

    def remember(artifact):
        if keep_states:
            states[json.loads(artifact.side_information)["method"]] = _states(artifact)
    try:
        for component in chunk.components:
            with Path(component.path).open("rb") as stream:
                stream.seek(component.offset)
                source[component.id] = stream.read(component.length)
    except OSError:
        # evaluate_target records the actual read error separately for each method.
        source = None
    for method_id in methods:
        observed = {}

        def observe(artifact):
            header = json.loads(artifact.side_information)
            direct = []
            for part in header["components"]:
                direct.append({"id": part["id"], "role": part["role"], "format": part["format"],
                               "original_bytes": part["bytes"],
                               "payload_bytes": sum(s["bytes"] for s in part["streams"]),
                               "direct_side_information_bytes": len(_json(part).encode())})
            observed["components"] = direct
            observed["shared_side_information_bytes"] = len(artifact.side_information) - sum(
                part["direct_side_information_bytes"] for part in direct)
            remember(artifact)

        result = evaluate_target(registry, method_id, chunk, (method_parameters or {}).get(method_id), max_input_bytes=chunk_bytes,
                                 components=source, artifact_observer=observe,
                                 baseline_observer=remember, _baseline_cache=baseline_cache)
        # Shared structure/specification is stored once in run/chunk records.
        for key in ("structure", "inference", "parameters", "method"):
            result.pop(key, None)
        result.update(observed)
        outcomes.append(result)
    record = {"chunk_id": chunk.id, "target_id": chunk.structure["chunk"]["parent_target_id"],
              "original_bytes": chunk.original_bytes, "structure": dict(chunk.structure),
              "inference": dict(chunk.inference), "results": outcomes, "finished_utc": _now()}
    state_blob = zlib.compress(_json(states).encode(), level=1) if keep_states else None
    return record, state_blob


def _add(summary, record):
    summary["completed_chunks"] += 1
    summary["processed_original_bytes"] += record["original_bytes"]
    size = record["original_bytes"]
    summary["chunk_original_bytes_min"] = min(summary["chunk_original_bytes_min"], size) if summary["chunk_original_bytes_min"] is not None else size
    summary["chunk_original_bytes_max"] = max(summary["chunk_original_bytes_max"], size) if summary["chunk_original_bytes_max"] is not None else size
    for result in record["results"]:
        method = summary["methods"].setdefault(result["method_id"], {
            "statuses": {}, "status_original_bytes": {}, "verified_original_bytes": 0, "payload_bytes": 0, "side_information_bytes": 0,
            "shared_side_information_bytes": 0, "by_domain_layer": {}, "by_component_format_layout": {}})
        status = result["status"]
        method["statuses"][status] = method["statuses"].get(status, 0) + 1
        method["status_original_bytes"][status] = method["status_original_bytes"].get(status, 0) + result["original_bytes"]
        if status != "verified":
            continue
        original = result["original_bytes"]
        payload, side = result["payload_bytes"], result["side_information_bytes"]
        method["verified_original_bytes"] += original
        method["payload_bytes"] += payload
        method["side_information_bytes"] += side
        method["shared_side_information_bytes"] += result["shared_side_information_bytes"]
        comparison = result.get("baseline_comparison")
        if comparison:
            paired = method.setdefault("baseline_comparison", {
                "method_id": comparison["method_id"], "kind": comparison["kind"], "statuses": {}, "paired_chunks": 0,
                "paired_original_bytes": 0, "baseline_payload_bytes": 0, "baseline_side_information_bytes": 0,
                "baseline_total_bytes": 0, "method_total_bytes": 0})
            baseline_status = comparison["status"]
            paired["statuses"][baseline_status] = paired["statuses"].get(baseline_status, 0) + 1
            if baseline_status == "verified":
                paired["paired_chunks"] += 1
                paired["paired_original_bytes"] += original
                paired["baseline_payload_bytes"] += comparison["payload_bytes"]
                paired["baseline_side_information_bytes"] += comparison["side_information_bytes"]
                paired["baseline_total_bytes"] += comparison["total_bytes"]
                paired["method_total_bytes"] += payload + side
        structure = record["structure"]
        key = f"{structure.get('domain', 'unknown')}:{structure.get('layer', 'unknown')}"
        group = method["by_domain_layer"].setdefault(key, {"original_bytes": 0, "total_bytes": 0})
        group["original_bytes"] += original
        group["total_bytes"] += payload + side
        for component in result["components"]:
            layout = structure.get("scale_block_shape")
            weight_format = result["components"][0]["format"]
            key = f"{component['role']}:{component['format']}:weight={weight_format}:scale_block={layout}"
            group = method["by_component_format_layout"].setdefault(key, {
                "original_bytes": 0, "payload_bytes": 0, "direct_side_information_bytes": 0})
            for field in group:
                group[field] += component[field]


def _progress(summary, *, status, inventory_bytes):
    result = {**summary, "status": status, "updated_utc": _now(),
              "inventory_original_bytes": inventory_bytes,
              "processed_fraction": summary["processed_original_bytes"] / inventory_bytes}
    if "selection_original_bytes" in summary:
        result["coverage"] = "subset"
        result["selection_processed_fraction"] = summary["processed_original_bytes"] / summary["selection_original_bytes"]
    result["methods"] = {}
    for key, values in summary["methods"].items():
        method = {**values}
        original = method["verified_original_bytes"]
        method["total_bytes"] = method["payload_bytes"] + method["side_information_bytes"]
        method["saving_bytes"] = original - method["total_bytes"] if original else None
        method["saving_fraction_on_verified_bytes"] = 1 - method["total_bytes"] / original if original else None
        method["verified_coverage_fraction"] = original / inventory_bytes
        paired = values.get("baseline_comparison")
        if paired:
            baseline_bytes = paired["baseline_total_bytes"]
            method["baseline_comparison"] = {**paired,
                "net_saving_bytes": baseline_bytes - paired["method_total_bytes"] if baseline_bytes else None,
                "net_saving_fraction": 1 - paired["method_total_bytes"] / baseline_bytes if baseline_bytes else None,
                "paired_coverage_fraction": paired["paired_original_bytes"] / inventory_bytes}
        method["by_component_format_layout"] = {
            group_id: {**group, "direct_total_bytes": group["payload_bytes"] + group["direct_side_information_bytes"],
                       "saving_fraction_excluding_shared_header": 1 - (
                           group["payload_bytes"] + group["direct_side_information_bytes"]) / group["original_bytes"]}
            for group_id, group in values["by_component_format_layout"].items()}
        result["methods"][key] = method
    return result


def write_size_report(output, summary):
    """Export recorded sizes only, without reading weights or rerunning codecs."""
    fields = ("method_id", "chunk_bytes_limit", "chunk_original_bytes_min", "chunk_original_bytes_max",
              "verified_original_bytes", "payload_bytes", "side_information_bytes", "total_bytes",
              "verified_coverage_fraction", "statuses", "saving_bytes", "saving_fraction_on_verified_bytes",
              "baseline_method_id", "comparison_kind", "paired_original_bytes", "baseline_total_bytes", "paired_method_total_bytes",
              "net_saving_bytes", "net_saving_fraction")
    path = Path(output) / "sizes.csv"
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method_id, method in sorted(summary["methods"].items()):
            row = {key: summary[key] for key in fields[1:4]}
            row.update({key: method[key] for key in ("verified_original_bytes", "payload_bytes", "side_information_bytes",
                       "total_bytes", "verified_coverage_fraction", "saving_fraction_on_verified_bytes")})
            row["saving_bytes"] = method["verified_original_bytes"] - method["total_bytes"] if method["verified_original_bytes"] else None
            paired = method.get("baseline_comparison")
            if paired:
                row.update(baseline_method_id=paired["method_id"], paired_original_bytes=paired["paired_original_bytes"],
                           comparison_kind=paired["kind"],
                           baseline_total_bytes=paired["baseline_total_bytes"] if paired["paired_chunks"] else None,
                           paired_method_total_bytes=paired["method_total_bytes"] if paired["paired_chunks"] else None,
                           net_saving_bytes=paired["net_saving_bytes"], net_saving_fraction=paired["net_saving_fraction"])
            if not method["verified_original_bytes"]:
                for key in ("payload_bytes", "side_information_bytes", "total_bytes"):
                    row[key] = None
            writer.writerow({**row, "method_id": method_id, "statuses": _json(method["statuses"])})
    temporary.replace(path)


def run_experiment(snapshot, output, *, workers=1, chunk_bytes=DEFAULT_CHUNK_BYTES, methods=None,
                   resume=False, keep_states=True, max_chunks=None, selection=None, method_parameters=None):
    """A real run only when called explicitly. max_chunks is an explicit subset selection."""
    if workers < 1 or chunk_bytes < 1 or (max_chunks is not None and max_chunks < 1):
        raise ValueError("workers, chunk bytes and max chunks must be positive")
    registry = builtin_registry()
    methods = tuple(registry.ids() if methods is None else methods)
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(registry.ids()):
        raise ValueError("Choose unique implemented method ids")
    if method_parameters is not None:
        if (not isinstance(method_parameters, dict) or set(method_parameters)-set(methods)
                or any(not isinstance(value, dict) for value in method_parameters.values())):
            raise ValueError("Method parameters must map selected method ids to parameter objects")
    model = load_checkpoint(snapshot)
    if selection is not None and max_chunks is not None:
        raise ValueError("Use explicit selection or max_chunks, not both")
    selected = select_chunks(model, chunk_bytes, selection) if selection is not None else None
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=resume)
    source_root = Path(__file__).parent
    code_hash = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py")):
        code_hash.update(str(path.relative_to(source_root)).encode())
        code_hash.update(path.read_bytes())
    sources = {}
    for shard in model.metadata["header_sha256"]:
        stat = (Path(snapshot) / shard).stat()
        sources[shard] = [stat.st_size, stat.st_mtime_ns, stat.st_ino]
    config = {"model_id": model.model_id, "revision": model.revision,
              "inventory": dict(model.metadata), "snapshot": str(Path(snapshot).absolute()),
              "chunk_bytes": chunk_bytes, "methods": list(methods), "keep_states": keep_states,
              "max_chunks": max_chunks, "implementation_sha256": code_hash.hexdigest(), "sources": sources,
              "method_specs": {mid: asdict(registry.create(mid).spec) for mid in methods},
              "backend": "native", "payload_persisted": False,
              "state_storage": "zlib-compressed JSON inside SQLite, logs only; measured artifact sizes unchanged"}
    if selection is not None:
        config["selection"] = selection
    if method_parameters is not None:
        config["method_parameters"] = method_parameters
    from .methods.native import build_metadata
    config["native_build"] = build_metadata()
    if any(registry.create(mid).spec.category == "B" for mid in methods):
        from .methods.predictive import build_metadata as predictive_build_metadata
        config["predictive_build"] = predictive_build_metadata()
    if "inter_frame_reference" in methods and "predictive_build" not in config:
        from .methods.predictive import build_metadata as predictive_build_metadata
        config["predictive_build"] = predictive_build_metadata()
    from .methods.shared_patterns import METHODS as c_methods
    if set(methods).intersection(c_methods):
        from .methods.shared_patterns import build_metadata as shared_pattern_build_metadata
        config["shared_pattern_build"] = shared_pattern_build_metadata()
    from .methods.math_structure import METHODS as d_methods
    if set(methods).intersection(d_methods):
        from .methods.math_structure import build_metadata as math_structure_build_metadata
        config["math_structure_build"] = math_structure_build_metadata()
    if "vq_codeword" in methods:
        from .methods.codeword_vq import build_metadata as codeword_vq_build_metadata
        config["codeword_vq_build"] = codeword_vq_build_metadata()
    if any("unit_bits" in p and p["unit_bits"] != 4 for p in (method_parameters or {}).values()):
        from .methods.grouped_units import build_metadata as grouped_build_metadata
        config["grouped_units_build"] = grouped_build_metadata()
        if "jpeg_ls_like_fp4" in methods:
            from .methods.grouped_units import jpeg_build_metadata
            config["grouped_jpeg_build"] = jpeg_build_metadata()
    run_file = output / "run.json"
    if resume:
        previous = json.loads(run_file.read_text())
        if previous["configuration"] != config:
            raise ValueError("Resume configuration/code/source files differ from the original run")
    else:
        _atomic_json(run_file, {"created_utc": _now(), "host": platform.node(),
                                "python": platform.python_version(), "configuration": config})
    run_lock = (output / "run.lock").open("a")
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        run_lock.close()
        raise ValueError("This run directory is already being processed") from None
    _atomic_json(output / "process.json", {"pid": os.getpid(), "workers": workers, "started_utc": _now()})
    database = sqlite3.connect(output / "results.sqlite3")
    database.execute("PRAGMA journal_mode=WAL")
    database.execute("PRAGMA synchronous=NORMAL")
    database.execute("""CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY, target_id TEXT NOT NULL, original_bytes INTEGER NOT NULL,
        result_json TEXT NOT NULL, state_zlib BLOB)""")
    completed = set()
    summary = {"completed_chunks": 0, "processed_original_bytes": 0, "methods": {},
               "chunk_bytes_limit": chunk_bytes, "chunk_original_bytes_min": None, "chunk_original_bytes_max": None}
    if selected is not None:
        summary.update(selection_chunks=len(selected), selection_original_bytes=sum(c.original_bytes for c in selected))
    for chunk_id, raw in database.execute("SELECT chunk_id,result_json FROM chunks"):
        completed.add(chunk_id)
        _add(summary, json.loads(raw))
    inventory_bytes = model.metadata["tensor_bytes"]
    progress_file = output / "progress.json"
    _atomic_json(progress_file, _progress(summary, status="running", inventory_bytes=inventory_bytes))

    def jobs():
        chunks = iter(selected) if selected is not None else (chunk for target in model.targets for chunk in iter_target_chunks(target, chunk_bytes))
        if max_chunks is not None:
            chunks = itertools.islice(chunks, max_chunks)
        for chunk in chunks:
            if chunk.id not in completed:
                yield chunk

    last_write = time.monotonic()
    stop_file = output / "STOP"
    if resume:
        stop_file.unlink(missing_ok=True)
    stop_requested = False
    try:
        # OpenMP may already be initialized by a preceding C3 encode. A fresh
        # interpreter avoids inheriting libgomp locks/threads across fork.
        context = "spawn" if "vq_codeword" in methods or set(methods).intersection(d_methods) or any(mid in c_methods and c_methods[mid][1] == "C3" for mid in methods) else "fork"
        # SQLite stays in the parent; bounded jobs never retain the whole model.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context(context)) as pool:
            pending, iterator, exhausted = set(), iter(jobs()), False
            while pending or not exhausted:
                if stop_file.exists():
                    exhausted = stop_requested = True
                while len(pending) < workers * 2 and not exhausted:
                    try:
                        chunk = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    pending.add(pool.submit(_worker, chunk, methods, chunk_bytes, keep_states, method_parameters))
                if not pending:
                    break
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    record, state = future.result()
                    database.execute("INSERT INTO chunks VALUES (?,?,?,?,?)", (
                        record["chunk_id"], record["target_id"], record["original_bytes"], _json(record), state))
                    database.commit()
                    _add(summary, record)
                if time.monotonic() - last_write >= 5:
                    _atomic_json(progress_file, _progress(summary, status="running", inventory_bytes=inventory_bytes))
                    last_write = time.monotonic()
        failures = any((set(m["statuses"]) | set(m.get("baseline_comparison", {}).get("statuses", {}))) &
                       {"failed", "verification_failed", "not_implemented", "resource_limited"}
                       for m in summary["methods"].values())
        expected_bytes = summary.get("selection_original_bytes", inventory_bytes)
        if not stop_requested and max_chunks is None and summary["processed_original_bytes"] != expected_bytes:
            raise RuntimeError("Processed byte coverage differs from the checkpoint inventory")
        status = "stopped" if stop_requested else "completed_with_failures" if failures else "completed_subset" if max_chunks or selected is not None else "completed"
        final = _progress(summary, status=status, inventory_bytes=inventory_bytes)
        _atomic_json(progress_file, final)
        _atomic_json(output / "summary.json", final)
        write_size_report(output, final)
        return final
    except BaseException as exc:
        progress = _progress(summary, status="interrupted", inventory_bytes=inventory_bytes)
        progress["error"] = f"{type(exc).__name__}: {exc}"
        _atomic_json(progress_file, progress)
        raise
    finally:
        database.close()
        run_lock.close()
