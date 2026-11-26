"""Download the fixed evaluation inputs (model + datasets) at pinned revisions.

Everything lands under data/lossy_eval/ (git-ignored). Writes manifest.json with
the resolved revisions so later runs can check they use the same inputs.
"""
import json, sys, time
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1] / "data" / "lossy_eval"
MODELS = {"gpt-oss-20b": ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee",
                          ["config.json", "generation_config.json", "model.safetensors.index.json", "model-0000?-of-00002.safetensors",
                           "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "LICENSE", "README.md"])}
# repo, revision, allow_patterns
DATASETS = {
    "wikitext": ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3", ["wikitext-2-raw-v1/*"]),
    "c4": ("allenai/c4", "1588ec454efa1a09f29cd18ddd04fe05fc8653a2", ["en/c4-validation.00000-of-00008.json.gz"]),
    "mmlu": ("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", ["all/*"]),
    "arc": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", ["ARC-Challenge/*", "ARC-Easy/*"]),
    "hellaswag": ("Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76", ["*"]),
    "winogrande": ("allenai/winogrande", "01e74176c63542e6b0bcb004dcdea22d94fb67b5", ["winogrande_xl/*", "*.py", "*.md"]),
    "piqa": ("ybisk/piqa", "2e8ac2dffd59bac8c3c6714948f4c551a0848bb0", ["*"]),  # loading script only; data below
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", ["main/*"]),
}


def main():
    which = sys.argv[1:] or ["models", "datasets"]
    man_path = ROOT / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    jobs = []
    if "models" in which:
        jobs += [("model", k, *v) for k, v in MODELS.items()]
    if "datasets" in which:
        jobs += [("dataset", k, *v) for k, v in DATASETS.items()]
    for kind, name, repo, rev, pats in jobs:
        t = time.time()
        try:
            p = snapshot_download(repo, repo_type=kind, revision=rev, allow_patterns=pats,
                                  local_dir=ROOT / f"{kind}s" / name)
            files = sorted(str(f.relative_to(p)) for f in Path(p).rglob("*") if f.is_file() and ".cache" not in f.parts)
            man[name] = dict(kind=kind, repo=repo, revision=rev, path=str(Path(p).relative_to(ROOT)), files=files, status="ok")
            print(f"{name}: ok {len(files)} files {time.time()-t:.0f}s", flush=True)
        except Exception as e:
            man[name] = dict(kind=kind, repo=repo, revision=rev, status=f"error: {type(e).__name__}: {e}")
            print(f"{name}: ERROR {e}", flush=True)
        man_path.write_text(json.dumps(man, indent=1))
    if "datasets" in which:
        fetch_piqa(man)
        man_path.write_text(json.dumps(man, indent=1))


# the HF repo for PIQA only holds a loading script; its train/dev archive (URL taken from that script)
PIQA_URL = "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip"


def fetch_piqa(man):
    import hashlib, io, urllib.request, zipfile
    dst = ROOT / "datasets" / "piqa" / "data"
    dst.mkdir(parents=True, exist_ok=True)
    raw = urllib.request.urlopen(PIQA_URL, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for n in z.namelist():
            if n.endswith(("dev.jsonl", "dev-labels.lst")):
                (dst / Path(n).name).write_bytes(z.read(n))
    man["piqa_data"] = dict(kind="dataset", url=PIQA_URL, sha256=hashlib.sha256(raw).hexdigest(),
                            files=["data/dev.jsonl", "data/dev-labels.lst"], status="ok")
    print("piqa data: ok", flush=True)


if __name__ == "__main__":
    main()
