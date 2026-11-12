"""Instrument the new wide integer paths; numerical plugins use Python tests."""
import json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1];m=json.loads((root/'build/grouped-units-build.json').read_text())
command=m['compile_commands'][0];flags=command[1:command.index('-c')];flags[flags.index('-O3')]='-O1'
source=root/'native/grouped_units';out=root/'build/check_grouped_units'
cmd=['/usr/bin/g++',*flags,'-fsanitize=address,undefined','-fno-omit-frame-pointer',str(root/'tests/native_grouped_units.cpp'),
     *[str(source/p) for p in ('kernels.cpp','entropy.cpp','legacy_match.cpp','sz.cpp')],
     *map(str,sorted((root/'build/predictive-objects').glob('*.o'))),'-L'+str(root/'build'),'-lfp4_entropy','-Wl,-rpath,$ORIGIN','-o',str(out)]
env={**os.environ,'PATH':'/usr/bin:/bin:'+os.environ.get('PATH','')}
subprocess.run(cmd,check=True,env=env)
subprocess.run([str(out)],check=True,env={**env,'ASAN_OPTIONS':'detect_leaks=0','UBSAN_OPTIONS':'halt_on_error=1'})
