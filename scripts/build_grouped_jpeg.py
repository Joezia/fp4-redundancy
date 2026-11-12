"""Keep the pinned CharLS codec unchanged; grow its bounded output buffer."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'build';gen=out/'grouped-jpeg-generated';gen.mkdir(exist_ok=True)
source=(root/'native/predictive/dedicated.hpp').read_text();s=source[source.index('inline Stream jpeg_encode'):source.index('inline Stream sz_encode')]
s=s.replace('Bytes data(encoder.estimated_destination_size());encoder.destination(data);data.resize(encoder.encode(raw));', '''Bytes data;
        size_t capacity=encoder.estimated_destination_size();
        for(;;){try{charls::jpegls_encoder attempt;attempt.frame_info({uint32_t(w),uint32_t(h),int32_t(bits),1}).near_lossless(0);
            data.resize(capacity);attempt.destination(data);data.resize(attempt.encode(raw));break;
        }catch(const charls::jpegls_error& error){if(error.code()!=charls::jpegls_errc::destination_buffer_too_small)throw;
            if(capacity>=(64u<<20))throw std::runtime_error("RESOURCE_LIMIT: JPEG destination exceeds 64 MiB cap");capacity=std::min<size_t>(capacity*2,64u<<20);}}''')
(gen/'jpeg_buffer.hpp').write_text('#include "'+str(root/'native/predictive/common.hpp')+'"\n#include "charls/charls.h"\nnamespace jpeg_buffer {using namespace predictive;\n'+s+'}\n')
src=root/'native/grouped_jpeg';vendor=root/'native/predictive/third_party/CharLS'
lib=out/'libfp4_grouped_jpeg.so';tmp=out/'libfp4_grouped_jpeg.tmp.so'
cmd=['/usr/bin/g++','-std=c++17','-O3','-DNDEBUG','-march=native','-fPIC','-shared','-fvisibility=hidden','-DCHARLS_STATIC','-Wl,--no-undefined',
     '-I'+str(gen),'-I'+str(vendor/'include'),str(src/'bridge.cpp'),*map(str,sorted((vendor/'src').glob('*.cpp'))),'-o',str(tmp)]
subprocess.run(cmd,check=True,env={**os.environ,'PATH':'/usr/bin:/bin:'+os.environ.get('PATH','')});tmp.replace(lib)
paths=[src/'bridge.cpp',root/'scripts/build_grouped_jpeg.py',root/'native/predictive/common.hpp',root/'native/predictive/dedicated.hpp']
paths+=sorted(p for p in vendor.rglob('*') if p.is_file())
m={'dependencies_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'library_sha256':hashlib.sha256(lib.read_bytes()).hexdigest(),'command':cmd}
(out/'grouped-jpeg-build.json').write_text(json.dumps(m,indent=2)+'\n');print(lib)
