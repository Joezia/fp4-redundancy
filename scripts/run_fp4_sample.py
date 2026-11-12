"""Plan and run a bounded FP4 sample using the existing verified experiment runner."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import argparse
import csv
import json
import os
import sqlite3
import hashlib
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fp4_redundancy.catalog import builtin_registry, catalog_output
from fp4_redundancy.checkpoint import load_checkpoint
from fp4_redundancy.chunks import iter_target_chunks
from fp4_redundancy.experiment import _atomic_json, _now, select_chunks

EXPERIMENT_DIR = ROOT / "experiments/fp4-module"
PLAN = EXPERIMENT_DIR / "plan.json"


def make_plan(snapshot):
    model = load_checkpoint(snapshot)
    registry = builtin_registry()
    completed = set()
    for path in (ROOT / "runs").glob("*/summary.json"):
        data = json.loads(path.read_text())
        if data.get("status") == "completed" and data.get("processed_fraction") == 1:
            completed.update(data["methods"])
    methods = sorted(set(registry.ids()) - completed)
    targets = {t.id: t for t in model.targets}
    chosen, descriptions, target_names = [], [], []
    for expert in (0, 1):
        for role in ("w1", "w2", "w3"):
            name = f"layers.0.ffn.experts.{expert}.{role}.weight"
            target = targets[name]
            if target.components[0].role != "packed_fp4":
                raise ValueError("Selected matrix is not packed FP4")
            target_names.append(name)
            for chunk in iter_target_chunks(target, 1048576):
                chosen.append({"target_id": name, "index": chunk.structure["chunk"]["index"]})
                descriptions.append(asdict(chunk))
    plan = {"kind": "fp4_complete_adjacent_matrices", "created_utc": _now(), "model_id": model.model_id,
            "revision": model.revision, "snapshot": str(Path(snapshot).resolve()), "chunk_bytes": 1048576,
            "chunks": chosen, "complete_matrix_targets": target_names, "methods": methods, "control_methods": ["arithmetic"],
            "full_model_completed_methods": sorted(completed),
            "not_implemented": [m["id"] for m in catalog_output()["methods"] if not m["implemented"]],
            "coverage": "subset", "sampling": "all rows of six complete FP4 matrices: layer0 adjacent experts0/1, w1/w2/w3; independent coding",
            "runtime": {"parallel_methods": 4, "workers_per_method": 4, "inner_threads": 1},
            "parameter_policy": "existing fixed method variants; no size/rank search in this screening run",
            "output_subdirectory": "fp4_module", "payload_persisted": False}
    plan["method_parameters"] = {"vq_codeword": {"vector_length": 4, "centers": 4, "objective": "rate", "correction_model": "auto"}}
    selected = select_chunks(model, 1048576, plan)
    plan.update(sample_chunks=len(selected), sample_original_bytes=sum(t.original_bytes for t in selected),
                sample_weight_bytes=sum(c.length for t in selected for c in t.components if c.role == "packed_fp4"),
                sample_scale_bytes=sum(c.length for t in selected for c in t.components if c.role == "scale"),
                inventory_original_bytes=model.metadata["tensor_bytes"])
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    with PLAN.open("x") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
        f.write("\n")
    _atomic_json(EXPERIMENT_DIR / "samples.json", descriptions)
    (EXPERIMENT_DIR / "plan.md").write_text(f"""# 连续完整 FP4 专家矩阵抽样

固定模型 {model.model_id}，版本 `{model.revision}`。
样本为 layer0 的连续专家0/1，各完整 w1/w2/w3，共6张矩阵，不再汇总不同层/模块的碎片。
覆盖每张矩阵全部行，逐矩阵独立编码，不跨矩阵或专家拼接。受插件输入上限限制，内部按1 MiB、连续行和scale对齐独立分块。

- 待测：尚未完成全模型的 {len(methods)} 个可执行方法 id（保留不同VQ适配及复用别名），另跑 arithmetic 对照。
- 原始权重 {plan['sample_weight_bytes']} bytes；独立scale {plan['sample_scale_bytes']} bytes；合计 {plan['sample_original_bytes']} bytes。
- 共 {plan['sample_chunks']} 个连续块，完整覆盖6个原生目标；精确形状/范围见 samples.json。
- 当前固定方法配置；码字VQ用已确定的4维、最多4中心、码长目标及auto纠错，不在本轮重新挑选参数。
- 最多4个方法并行、每方法4进程、内部线程1；不启动全模型遍历。
- 逐字节验证、完整计费；数据流验证后丢弃，状态/结果位于 runs/<方法>/fp4_module/。
- 表格只提取FP4权重，排除scale数据与scale独立元数据，原目标共有头部全额计给FP4；不摊薄必要开销。

结果是同一局部完整专家组的抽样，不代表全模型；每个矩阵的结果在SQLite中独立保留。
""", encoding="utf-8")
    print(json.dumps({k: plan[k] for k in ("sample_chunks", "sample_original_bytes", "sample_weight_bytes", "sample_scale_bytes", "methods", "not_implemented")}, ensure_ascii=False, indent=2))


def report(plan):
    """FP4-only accounting view of already verified artifacts; no codec execution."""
    rows, matrices, source_hashes = [], [], {}
    for method in plan["methods"] + plan["control_methods"]:
        output = ROOT / "runs" / method / plan["output_subdirectory"]
        path = output / "summary.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        groups, hashes = {}, {}
        with sqlite3.connect(f"file:{output/'results.sqlite3'}?mode=ro", uri=True) as db:
            for raw, in db.execute("SELECT result_json FROM chunks"):
                record = json.loads(raw)
                result = record["results"][0]
                if result["status"] != "verified":
                    raise RuntimeError(f"Unverified FP4 sample: {method}: {result['reason']}")
                parts = [c for c in result["components"] if c["role"] == "packed_fp4"]
                if len(parts) != 1:
                    raise ValueError("This FP4 report requires exactly one FP4 component per target")
                part = parts[0]
                # Removing the non-FP4 JSON component entries also removes one
                # list separator each. The whole original shared header stays.
                metadata = (part["direct_side_information_bytes"] + result["shared_side_information_bytes"]
                            - (len(result["components"]) - 1))
                key = record["target_id"]
                group = groups.setdefault(key, {"method_id": method, "target_id": key, "chunks": 0,
                    "fp4_original_bytes": 0, "fp4_payload_bytes": 0, "fp4_metadata_bytes": 0})
                group["chunks"] += 1
                group["fp4_original_bytes"] += part["original_bytes"]
                group["fp4_payload_bytes"] += part["payload_bytes"]
                group["fp4_metadata_bytes"] += metadata
                hashes[record["chunk_id"]] = result["source_sha256"]
        if len(groups) != len(plan["complete_matrix_targets"]):
            raise RuntimeError(f"Incomplete matrix coverage: {method}")
        row = {"method_id": method, "scope": "six_complete_adjacent_matrices", "status": summary["status"],
               "matrices": len(groups), "chunks": summary["completed_chunks"], "chunk_bytes_limit": plan["chunk_bytes"]}
        for field in ("fp4_original_bytes", "fp4_payload_bytes", "fp4_metadata_bytes"):
            row[field] = sum(g[field] for g in groups.values())
        if row["fp4_original_bytes"] != plan["sample_weight_bytes"] or row["chunks"] != plan["sample_chunks"]:
            raise RuntimeError(f"Incomplete FP4 bytes: {method}")
        row["fp4_total_bytes"] = row["fp4_payload_bytes"] + row["fp4_metadata_bytes"]
        row["fp4_saving_fraction"] = 1 - row["fp4_total_bytes"] / row["fp4_original_bytes"]
        row["fp4_payload_saving_fraction"] = 1 - row["fp4_payload_bytes"] / row["fp4_original_bytes"]
        row["net_saving_fraction"] = None
        for g in groups.values():
            g["fp4_total_bytes"] = g["fp4_payload_bytes"] + g["fp4_metadata_bytes"]
            g["fp4_saving_fraction"] = 1 - g["fp4_total_bytes"] / g["fp4_original_bytes"]
            g["fp4_payload_saving_fraction"] = 1 - g["fp4_payload_bytes"] / g["fp4_original_bytes"]
            matrices.append(g)
        source_hashes[method] = hashes
        rows.append(row)
    if not rows:
        return
    reference = source_hashes[rows[0]["method_id"]]
    if any(hashes != reference for hashes in source_hashes.values()):
        raise RuntimeError("Source hashes differ between methods")
    baseline = next((r for r in rows if r["method_id"] == "arithmetic"), None)
    if baseline:
        for r in rows:
            if r["method_id"] != "arithmetic":
                r["net_saving_fraction"] = 1 - r["fp4_total_bytes"] / baseline["fp4_total_bytes"]
    for filename, data in (("results", rows), ("matrix-results", matrices)):
        temporary = EXPERIMENT_DIR / (filename+".tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
        temporary.replace(EXPERIMENT_DIR / (filename+".csv"))
    lines = ["# 完整专家矩阵 FP4 抽样结果", "",
             f"更新于 {_now()}。完成 {len(rows)}/{len(plan['methods'])+1} 个方法；范围见[方案](plan.md)。",
             "仅计FP4权重，不含scale数据及其独立元数据；FP4码表/参数/纠错和原目标共有头部完整计费。",
             "Payload节省率 = 1 − payload / 原始packed字节；总节省率包含全部元数据，相对算术净节省也保持总成本口径。",
             "Payload保留ANS重归一化输出和最终状态，以及各方法的前缀、残差、纠错、索引、系数/因子等实际编码流；独立模型/布局信息仍计入元数据。详见[分类说明](../../docs/开发说明.md#payload与元数据)。",
             "Payload节省率仅用于诊断；元数据可能包含字典或参考信息，不代表排除元数据即可获得实际压缩收益。",
             "每项覆盖同一层的相邻专家0/1、各完整w1/w2/w3，共6张矩阵；每矩阵独立分块编码，不跨矩阵拼接。", "",
             "| 方法 | Payload bytes | Payload节省率 | 元数据 bytes | 合计 bytes | 总节省率 | 相对算术净节省 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in sorted(rows, key=lambda r: r["fp4_total_bytes"]):
        net = "—" if row["net_saving_fraction"] is None else f"{row['net_saving_fraction']:.2%}"
        lines.append(f"| {row['method_id']} | {row['fp4_payload_bytes']} | {row['fp4_payload_saving_fraction']:.2%} | {row['fp4_metadata_bytes']} | {row['fp4_total_bytes']} | {row['fp4_saving_fraction']:.2%} | {net} |")
    lines += ["", "每张完整矩阵的独立结果见 [matrix-results.csv](matrix-results.csv)，完整开销见 [results.csv](results.csv)。",
              "原始带scale结果、编码状态与逐字节验证仍保留在 runs/<方法>/fp4_module/。这只是局部完整矩阵抽样，不代表全模型。"]
    temporary = EXPERIMENT_DIR / "report.tmp"
    temporary.write_text("\n".join(lines)+"\n", encoding="utf-8")
    temporary.replace(EXPERIMENT_DIR / "report.md")
    validation_path = EXPERIMENT_DIR / "validation.json"
    previous = json.loads(validation_path.read_text()) if validation_path.exists() else {}
    _atomic_json(validation_path, {**previous, "completed_methods": len(rows),
        "verified_method_chunk_pairs": sum(r["chunks"] for r in rows), "matrices_per_method": 6,
        "fp4_original_bytes_per_method": plan["sample_weight_bytes"], "same_source_hashes": True,
        "accounting": "FP4 payload + direct FP4 metadata + entire target shared header - removed JSON separators",
        "source_sha256_by_chunk": reference})


def run(plan):
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "FP4_C_THREADS": "1", "FP4_D_THREADS": "1", "FP4_VQ_THREADS": "1"}
    # Submit expensive variants first so that a single final task does not hold up the batch.
    priority = ["cp_lossless", "ctw", "ksvd_lossless", "aq_lossless", "repair", "sequitur", "opq_lossless", "neural_entropy"]
    methods = priority + [m for m in plan["methods"] + plan["control_methods"] if m not in priority]
    methods = [m for m in methods if m in plan["methods"] + plan["control_methods"]]
    execution = {}
    def one(method):
        output = ROOT / "runs" / method / plan["output_subdirectory"]
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "fp4_redundancy", "run", plan["snapshot"], "--output", str(output),
                   "--methods", method, "--chunk-bytes", str(plan["chunk_bytes"]), "--selection", str(PLAN),
                   "--workers", str(plan["runtime"]["workers_per_method"])]
        if method in plan.get("method_parameters", {}):
            command += ["--method-parameters", json.dumps({method: plan["method_parameters"][method]})]
        if (output / "run.json").exists():
            command.append("--resume")
        start = time.monotonic()
        print(json.dumps({"event": "start", "method": method, "utc": _now()}), flush=True)
        with (output.parent / "fp4_module-execution.log").open("a") as log:
            outcome = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        result = {"exit_code": outcome.returncode, "wall_seconds": time.monotonic()-start, "finished_utc": _now()}
        print(json.dumps({"event": "finished", "method": method, **result}), flush=True)
        return result
    with ThreadPoolExecutor(max_workers=plan["runtime"]["parallel_methods"]) as pool:
        futures = {pool.submit(one, method): method for method in methods}
        for future in as_completed(futures):
            execution[futures[future]] = future.result()
            _atomic_json(EXPERIMENT_DIR / "execution.json", execution)
            report(plan)
    report(plan)
    if any(item["exit_code"] for item in execution.values()):
        raise RuntimeError("Some sample runs failed; inspect per-method fp4_module-execution.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "report"))
    parser.add_argument("--snapshot", default=os.environ.get("SNAPSHOT"))
    args = parser.parse_args()
    if args.command == "plan":
        if not args.snapshot:
            parser.error("source .env or pass --snapshot")
        make_plan(args.snapshot)
    else:
        plan = json.loads(PLAN.read_text())
        (run if args.command == "run" else report)(plan)


if __name__ == "__main__":
    main()
