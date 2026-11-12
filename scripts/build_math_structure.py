"""Build D C++ plugins independently, reusing existing A1 and Eigen sources."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--check", action="store_true", help="Run synthetic native ASan/UBSan mechanism checks")
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "native/math_structure"
eigen = ROOT / "native/shared_patterns/third_party/eigen"
output = ROOT / "build"
objects = output / "math-structure-objects"
objects.mkdir(parents=True, exist_ok=True)
environment = {**os.environ, "PATH": "/usr/local/bin:/usr/bin:/bin:" + os.environ.get("PATH", "")}
flags = ["-std=c++17", "-O3", "-DNDEBUG", "-march=native", "-fPIC", "-fvisibility=hidden", "-fopenmp", "-I" + str(eigen)]


def compile_source(path):
    obj = objects / (path.stem + ".o")
    command = ["/usr/bin/g++", *flags, "-c", str(path), "-o", str(obj)]
    subprocess.run(command, check=True, env=environment)
    return obj, command


with ThreadPoolExecutor(max_workers=3) as pool:
    compiled = list(pool.map(compile_source, sorted(source.glob("*.cpp"))))
temporary = output / "libfp4_math_structure.tmp.so"
command = ["/usr/bin/g++", "-shared", "-fopenmp", "-Wl,--no-undefined", *(str(p) for p, _ in compiled),
           "-L" + str(output), "-lfp4_entropy", "-Wl,-rpath,$ORIGIN", "-o", str(temporary)]
subprocess.run(command, check=True, env=environment)
library = output / "libfp4_math_structure.so"
temporary.replace(library)
h = hashlib.sha256()
for p in sorted(p for p in source.rglob("*") if p.is_file()):
    h.update(str(p.relative_to(source)).encode()); h.update(p.read_bytes())
dependencies = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [ROOT / "native/predictive/common.hpp", ROOT / "native/shared_patterns/common.hpp",
                          ROOT / "native/shared_patterns/quantization.hpp", *sorted(p for p in eigen.rglob("*") if p.is_file())]}
metadata = {"source_sha256": h.hexdigest(), "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "a1_library_sha256": hashlib.sha256((output / "libfp4_entropy.so").read_bytes()).hexdigest(),
            "dependencies_sha256": dependencies, "command": command, "compile_commands": [c for _, c in compiled],
            "compiler": subprocess.check_output(["/usr/bin/g++", "--version"], text=True).splitlines()[0]}
(output / "math-structure-build.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(library, flush=True)
if args.check:
    checks = objects / "checks"
    checks.mkdir(exist_ok=True)
    check_flags = ["-std=c++17", "-O1", "-g", "-DNDEBUG", "-fopenmp", "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-I" + str(eigen)]
    def compile_check(path):
        obj = checks / (path.stem + ".o")
        subprocess.run(["/usr/bin/g++", *check_flags, "-c", str(path), "-o", str(obj)], check=True, env=environment)
        return str(obj)
    paths = [ROOT / "tests/native_math_structure.cpp", *(source / name for name in ("exact.cpp", "transforms.cpp", "sparse.cpp", "factorization.cpp"))]
    with ThreadPoolExecutor(max_workers=2) as pool:
        check_objects = list(pool.map(compile_check, paths))
    checked = output / "check_math_structure"
    subprocess.run(["/usr/bin/g++", *check_flags, *check_objects, "-L" + str(output), "-lfp4_entropy",
                    "-Wl,-rpath,$ORIGIN", "-o", str(checked)], check=True, env=environment)
    subprocess.run([str(checked)], check=True, env={**environment, "ASAN_OPTIONS": "detect_leaks=0",
                                                  "UBSAN_OPTIONS": "halt_on_error=1", "FP4_D_THREADS": "2"})
