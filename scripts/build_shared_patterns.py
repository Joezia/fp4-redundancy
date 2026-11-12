"""Offline C build; independent library, parallel native compilation, atomic install."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--check", action="store_true", help="Also run synthetic ASan/UBSan native checks")
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "native/shared_patterns"
vendor = source / "third_party"
output = ROOT / "build"
objects = output / "shared-pattern-objects"
objects.mkdir(parents=True, exist_ok=True)
environment = {**os.environ, "PATH": "/usr/local/bin:/usr/bin:/bin:" + os.environ.get("PATH", "")}
defines = ["-DSIZEOF_UNSIGNED_LONG=8", "-DSIZEOF_SIZE_T=8", "-DSIZEOF_UNSIGNED_INT=4",
           "-DSIZEOF_UNSIGNED_LONG_LONG=8", "-DXD3_USE_LARGEFILE64=1", "-DSECONDARY_DJW=0", "-DSECONDARY_FGK=0", "-DSECONDARY_LZMA=0"]


def compile_source(item):
    path, compiler = item
    obj = objects / (path.stem + ".o")
    command = [compiler, "-std=c11" if compiler.endswith("gcc") else "-std=c++17",
               "-O3", "-DNDEBUG", "-march=native", "-fPIC", "-fvisibility=hidden", "-fopenmp",
               *(["-include", "assert.h"] if compiler.endswith("gcc") else []), *defines, "-I" + str(vendor / "eigen"), "-c", str(path), "-o", str(obj)]
    subprocess.run(command, check=True, env=environment)
    return obj, command


with ThreadPoolExecutor(max_workers=3) as pool:
    compiled = list(pool.map(compile_source, [(vendor / "xdelta/xdelta3.c", "/usr/bin/gcc"),
                          (source / "sequitur.cpp", "/usr/bin/g++"),
                          (source / "shared_patterns.cpp", "/usr/bin/g++")]))
temporary = output / "libfp4_shared_patterns.tmp.so"
command = ["/usr/bin/g++", "-shared", "-fopenmp", "-Wl,--no-undefined", *(str(p) for p, _ in compiled),
           "-L" + str(output), "-lfp4_entropy", "-Wl,-rpath,$ORIGIN", "-o", str(temporary)]
subprocess.run(command, check=True, env=environment)
library = output / "libfp4_shared_patterns.so"
temporary.replace(library)
h = hashlib.sha256()
for path in sorted(p for p in source.rglob("*") if p.is_file()):
    h.update(str(path.relative_to(source)).encode()); h.update(path.read_bytes())
metadata = {"source_sha256": h.hexdigest(), "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "a1_library_sha256": hashlib.sha256((output / "libfp4_entropy.so").read_bytes()).hexdigest(),
            "common_header_sha256": hashlib.sha256((ROOT / "native/predictive/common.hpp").read_bytes()).hexdigest(),
            "command": command, "compile_commands": [cmd for _, cmd in compiled],
            "compiler": subprocess.check_output(["/usr/bin/g++", "--version"], text=True).splitlines()[0]}
(output / "shared-pattern-build.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(library)
if args.check:
    checked = output / "check_shared_patterns"
    check_command = ["/usr/bin/g++", "-std=c++17", "-O1", "-g", "-DNDEBUG", "-fopenmp",
                     "-fsanitize=address,undefined", "-fno-omit-frame-pointer", *defines,
                     "-I" + str(vendor / "eigen"), str(ROOT / "tests/native_shared_patterns.cpp"),
                     str(source / "sequitur.cpp"), str(objects / "xdelta3.o"),
                     "-L" + str(output), "-lfp4_entropy", "-Wl,-rpath,$ORIGIN", "-o", str(checked)]
    subprocess.run(check_command, check=True, env=environment)
    subprocess.run([str(checked)], check=True, env={**environment, "ASAN_OPTIONS": "detect_leaks=0",
                                                  "UBSAN_OPTIONS": "halt_on_error=1", "FP4_C_THREADS": "2"})
