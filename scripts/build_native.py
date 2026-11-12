"""Build pinned, vendored C++ codecs. No network or model access."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "native"
vendor = source / "third_party"
output = ROOT / "build"
output.mkdir(exist_ok=True)
library = output / "libfp4_entropy.so"
temporary_library = output / "libfp4_entropy.tmp.so"
fann_object = output / "fann.o"
fann_command = ["/usr/bin/gcc", "-std=c99", "-O3", "-fPIC", "-fvisibility=hidden",
                "-I" + str(vendor / "FANN/src/include"), "-c", str(vendor / "FANN/src/floatfann.c"),
                "-o", str(fann_object)]
command = ["/usr/bin/g++", "-std=c++17", "-O3", "-DNDEBUG", "-march=native", "-fPIC", "-shared",
           "-fvisibility=hidden", "-Wl,--no-undefined", "-I" + str(vendor / "SZ3/include"),
           "-I" + str(vendor / "YAECL-Yet-Another-Entropy-Coding-Library"),
           "-I" + str(vendor / "ryg_rans"), str(source / "entropy.cpp"), str(fann_object), "-lm", "-lgomp", "-o", str(temporary_library)]
environment = {**os.environ, "PATH": "/usr/local/bin:/usr/bin:/bin:" + os.environ.get("PATH", "")}
subprocess.run(fann_command, check=True, env=environment)
subprocess.run(command, check=True, env=environment)
temporary_library.replace(library)
fingerprint = hashlib.sha256()
for path in sorted(p for p in source.rglob("*") if p.is_file() and not {"predictive", "shared_patterns", "math_structure", "codeword_vq", "grouped_units", "grouped_jpeg"}.intersection(p.relative_to(source).parts)):
    fingerprint.update(str(path.relative_to(source)).encode())
    fingerprint.update(path.read_bytes())
metadata = {"source_sha256": fingerprint.hexdigest(),
            "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "command": command,
            "fann_command": fann_command,
            "compiler": subprocess.check_output(["/usr/bin/g++", "--version"], text=True).splitlines()[0]}
(output / "native-build.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(library)
