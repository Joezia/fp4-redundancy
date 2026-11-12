import argparse
import json
import sys

from .catalog import builtin_registry, catalog_output, iter_chunk_plan, plan
from .checkpoint import load_checkpoint
from .manifest import load_manifest, save_manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="FP4 lossless redundancy framework — scaffold")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("methods", help="List techniques and whether an implementation is registered")
    checkpoint = commands.add_parser("checkpoint", help="Inspect local DeepSeek checkpoint headers; no compression")
    checkpoint.add_argument("snapshot")
    checkpoint.add_argument("--output", help="Write an input manifest to a new file")
    run = commands.add_parser("run", help="Compress and verify local checkpoint chunks; save states/results only")
    run.add_argument("snapshot")
    run.add_argument("--output", required=True)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--chunk-bytes", type=int, default=1048576)
    run.add_argument("--methods", nargs="+", choices=builtin_registry().ids())
    run.add_argument("--resume", action="store_true")
    run.add_argument("--no-states", action="store_true", help="Save results and hashes only")
    run.add_argument("--max-chunks", type=int, help="Explicitly run only the first N chunks (subset, not full model)")
    run.add_argument("--selection", help="JSON plan with model_id/revision/chunk_bytes and explicit target_id/index chunks")
    run.add_argument("--method-parameters", help='JSON mapping of method ids to algorithm parameters')
    for name in ("validate", "plan"):
        command = commands.add_parser(name)
        command.add_argument("manifest")
        command.add_argument("--check-files", action="store_true",
                             help="Also check source file presence and byte ranges")
        if name == "plan":
            command.add_argument("--categories", nargs="+", choices=list("ABCDEFG"), default=["A", "B", "C", "D"])
            command.add_argument("--chunk-bytes", type=int, help="Stream a scale-aligned chunk plan as JSONL")
            command.add_argument("--target", help="Select one original target id (requires --chunk-bytes)")
    args = parser.parse_args(argv)
    try:
        if args.command == "methods":
            output = catalog_output()
        elif args.command == "run":
            from .experiment import run_experiment
            from pathlib import Path
            selection = json.loads(Path(args.selection).read_text()) if args.selection else None
            output = run_experiment(args.snapshot, args.output, workers=args.workers,
                                    chunk_bytes=args.chunk_bytes, methods=args.methods,
                                    resume=args.resume, keep_states=not args.no_states, max_chunks=args.max_chunks,
                                    selection=selection,
                                    method_parameters=json.loads(args.method_parameters) if args.method_parameters else None)
        elif args.command == "checkpoint":
            manifest = load_checkpoint(args.snapshot)
            if args.output:
                save_manifest(manifest, args.output)
            output = {"kind": "checkpoint_inventory", "measurements": False,
                      "model_id": manifest.model_id, "revision": manifest.revision,
                      "targets": len(manifest.targets), **manifest.metadata}
        else:
            manifest = load_manifest(args.manifest, check_files=args.check_files)
            if args.command == "plan":
                if args.target and args.chunk_bytes is None:
                    raise ValueError("--target requires --chunk-bytes")
                if args.chunk_bytes is not None:
                    if args.chunk_bytes <= 0:
                        raise ValueError("--chunk-bytes must be positive")
                    for record in iter_chunk_plan(manifest, args.categories, args.chunk_bytes, args.target):
                        print(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                    return 0
                output = plan(manifest, args.categories)
            else:
                output = {"kind": "manifest_validation", "valid": True,
                          "model_id": manifest.model_id, "targets": len(manifest.targets),
                          "coverage_claim": manifest.coverage,
                          "coverage_independently_verified": False}
            output["source_file_ranges_checked"] = args.check_files
            output["storage_format_parsed"] = False
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, TypeError, KeyError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
