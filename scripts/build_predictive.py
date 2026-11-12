"""Build the independent B library, preserving A2's loaded A library inode."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "native/predictive"
vendor = source / "third_party"
output = ROOT / "build"
output.mkdir(exist_ok=True)
objects = output / "predictive-objects"
objects.mkdir(exist_ok=True)
if not (output / "libfp4_entropy.so").is_file():
    raise SystemExit("Build A1 first: python3 scripts/build_native.py")
environment = {**os.environ, "PATH": "/usr/local/bin:/usr/bin:/bin:" + os.environ.get("PATH", "")}


def compile_c(path):
    obj = objects / (path.parent.name + "-" + path.stem + ".o")
    command = ["/usr/bin/gcc", "-std=c99", "-O3", "-DNDEBUG", "-fPIC", "-fvisibility=hidden",
               "-DZSTD_DISABLE_ASM", "-c", str(path), "-o", str(obj)]
    subprocess.run(command, check=True, env=environment)
    return obj, command


c_sources = sorted((vendor / "zstd").glob("*/*.c"))
with ThreadPoolExecutor(max_workers=4) as pool:
    compiled = list(pool.map(compile_c, c_sources))
temporary = output / "libfp4_predictive.tmp.so"
command = ["/usr/bin/g++", "-std=c++17", "-O3", "-DNDEBUG", "-march=native", "-fPIC", "-shared",
           "-fvisibility=hidden", "-DCHARLS_STATIC", "-Wl,--no-undefined",
           "-I" + str(vendor / "CharLS/include"), "-I" + str(vendor / "SZ3/include"),
           "-I" + str(ROOT / "native/third_party/SZ3/include"), str(source / "predictive.cpp"),
           *map(str, sorted((vendor / "CharLS/src").glob("*.cpp"))),
           *(str(obj) for obj, _ in compiled), "-L" + str(output), "-lfp4_entropy",
           "-Wl,-rpath,$ORIGIN", "-o", str(temporary)]
subprocess.run(command, check=True, env=environment)
library = output / "libfp4_predictive.so"
temporary.replace(library)  # Never truncate a library already mmap'ed by another process.
h = hashlib.sha256()
for path in sorted(p for p in source.rglob("*") if p.is_file()):
    h.update(str(path.relative_to(source)).encode())
    h.update(path.read_bytes())
metadata = {"source_sha256": h.hexdigest(),
            "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "a1_library_sha256": hashlib.sha256((output / "libfp4_entropy.so").read_bytes()).hexdigest(),
            "command": command, "c_commands": [cmd for _, cmd in compiled],
            "compiler": subprocess.check_output(["/usr/bin/g++", "--version"], text=True).splitlines()[0]}
(output / "predictive-build.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(library)
