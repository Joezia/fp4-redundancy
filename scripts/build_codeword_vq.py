"""Build the complete codeword-VQ plugin, reusing pinned A1 arithmetic code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "native/codeword_vq"
output = ROOT / "build"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--check", action="store_true", help="Run synthetic native ASan/UBSan checks")
args = parser.parse_args()
output.mkdir(exist_ok=True)
temporary = output / "libfp4_codeword_vq.tmp.so"
command = ["/usr/bin/g++", "-std=c++17", "-O3", "-DNDEBUG", "-march=native", "-fPIC", "-shared", "-fopenmp",
           "-fvisibility=hidden", "-Wl,--no-undefined", str(source / "vq.cpp"), "-L"+str(output),
           "-lfp4_entropy", "-Wl,-rpath,$ORIGIN", "-o", str(temporary)]
environment = {**os.environ, "PATH": "/usr/local/bin:/usr/bin:/bin:"+os.environ.get("PATH", "")}
subprocess.run(command, check=True, env=environment)
library = output / "libfp4_codeword_vq.so"
temporary.replace(library)
h = hashlib.sha256()
for path in sorted(p for p in source.rglob("*") if p.is_file()):
    h.update(str(path.relative_to(source)).encode()); h.update(path.read_bytes())
dependencies = ["native/shared_patterns/common.hpp", "native/predictive/common.hpp",
                "native/third_party/YAECL-Yet-Another-Entropy-Coding-Library/yaecl.hpp"]
metadata = {"source_sha256": h.hexdigest(), "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "a1_library_sha256": hashlib.sha256((output/"libfp4_entropy.so").read_bytes()).hexdigest(),
            "dependencies_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in dependencies},
            "command": command, "compiler": subprocess.check_output(["/usr/bin/g++", "--version"], text=True).splitlines()[0]}
(output / "codeword-vq-build.json").write_text(json.dumps(metadata, indent=2)+"\n")
print(library, flush=True)
if args.check:
    checked = output / "check_codeword_vq"
    command = ["/usr/bin/g++", "-std=c++17", "-O1", "-g", "-DNDEBUG", "-fopenmp", "-fsanitize=address,undefined",
               "-fno-omit-frame-pointer", str(ROOT/"tests/native_codeword_vq.cpp"), "-L"+str(output), "-lfp4_entropy",
               "-Wl,-rpath,$ORIGIN", "-o", str(checked)]
    subprocess.run(command, check=True, env=environment)
    subprocess.run([str(checked)], check=True, env={**environment, "ASAN_OPTIONS":"detect_leaks=0",
                                                  "UBSAN_OPTIONS":"halt_on_error=1", "FP4_VQ_THREADS":"2"})
