"""Build private packed-word adaptations; old plugin sources/binaries stay intact."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,os,subprocess
from generate_grouped_numeric import generate
from generate_grouped_vq import generate as generate_vq
ROOT=Path(__file__).resolve().parents[1]
src=ROOT/"native/grouped_units";out=ROOT/"build";generated=out/"grouped-generated";generated.mkdir(parents=True,exist_ok=True)
predictors=(ROOT/"native/predictive/predictors.hpp").read_text()
predictors=predictors.replace('#include "common.hpp"','').replace('namespace predictive {','namespace wide_predictive {\nusing namespace grouped;')
predictors=predictors.replace('uint16_t value,int64_t prediction','uint64_t value,int64_t prediction')
predictors=predictors.replace('unsigned r=wrap(int64_t(value)-prediction,bits),range=1u<<bits;','uint64_t r=wrap(int64_t(value)-prediction,bits),range=uint64_t(1)<<bits;')
predictors=predictors.replace('int64_t xy=0,xx=0;','__int128 xy=0,xx=0;')
predictors=predictors.replace('xy+=x*y;xx+=x*x;','xy+=__int128(x)*y;xx+=__int128(x)*x;')
predictors=predictors.replace('std::clamp<int64_t>(xy*4096/xx,-4096,4096)','std::clamp<__int128>(xy*4096/xx,-4096,4096)')
(generated/"wide_predictors.hpp").write_text(predictors)
numeric_sources=generate(ROOT,generated)
generate_vq(ROOT,generated)
env={**os.environ,"PATH":"/usr/bin:/bin:"+os.environ.get('PATH','')}
flags=['-std=c++17','-O3','-DNDEBUG','-march=native','-fPIC','-fvisibility=hidden','-fopenmp','-I'+str(generated),
       '-I'+str(ROOT/'native/third_party/SZ3/include'),'-I'+str(ROOT/'native/predictive/third_party/SZ3/include'),'-I'+str(ROOT/'native/shared_patterns/third_party/eigen')]
flags+=['-DSIZEOF_UNSIGNED_LONG=8','-DSIZEOF_SIZE_T=8','-DSIZEOF_UNSIGNED_INT=4','-DSIZEOF_UNSIGNED_LONG_LONG=8','-DXD3_USE_LARGEFILE64=1','-DSECONDARY_DJW=0','-DSECONDARY_FGK=0','-DSECONDARY_LZMA=0']
def compile_one(path):
 obj=generated/(path.stem+'.o');cmd=['/usr/bin/g++',*flags,'-c',str(path),'-o',str(obj)]
 subprocess.run(cmd,check=True,env=env);return obj,cmd
with ThreadPoolExecutor(max_workers=3) as pool:compiled=list(pool.map(compile_one,sorted(src.glob('*.cpp'))+numeric_sources))
temporary=out/'libfp4_grouped_units.tmp.so'
zstd_objects=sorted((out/'predictive-objects').glob('*.o'))
if not zstd_objects:raise RuntimeError('Build the pinned predictive Zstd objects first')
cmd=['/usr/bin/g++','-shared','-fopenmp','-Wl,--no-undefined',*[str(p) for p,_ in compiled],*map(str,zstd_objects),'-L'+str(out),'-lfp4_entropy','-Wl,-rpath,$ORIGIN','-o',str(temporary)]
subprocess.run(cmd,check=True,env=env);library=out/'libfp4_grouped_units.so';temporary.replace(library)
h=hashlib.sha256()
for p in sorted(p for p in src.rglob('*') if p.is_file()):h.update(str(p.relative_to(src)).encode());h.update(p.read_bytes())
deps=['native/predictive/common.hpp','native/predictive/predictors.hpp','native/third_party/YAECL-Yet-Another-Entropy-Coding-Library/yaecl.hpp']
deps+=['scripts/generate_grouped_numeric.py','native/shared_patterns/common.hpp','native/shared_patterns/reference.hpp','native/shared_patterns/quantization.hpp']
deps+=['native/math_structure/'+n for n in ('common.hpp','transforms.cpp','sparse.cpp','factorization.cpp')]
deps+=['scripts/generate_grouped_vq.py','native/codeword_vq/vq.cpp']
deps+=['native/predictive/third_party/SZ3/include/SZ3/lossless/Lossless_zstd.hpp']
deps+=[str(p.relative_to(ROOT)) for p in zstd_objects]
metadata={'source_sha256':h.hexdigest(),'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),
 'dependencies_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in deps},
 'a1_library_sha256':hashlib.sha256((out/'libfp4_entropy.so').read_bytes()).hexdigest(),'command':cmd,'compile_commands':[c for _,c in compiled]}
(out/'grouped-units-build.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(library,flush=True)
