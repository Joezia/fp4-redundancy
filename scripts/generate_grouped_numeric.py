"""Private typed copies of existing complete numerical methods, no new algorithm.

Only the source domain, sparse-width representation and exact correction bridge
change. Q20 serialization, shapes, ranks and training loops stay in original files.
"""
import re

def generate(root, out):
    q=(root/'native/shared_patterns/quantization.hpp').read_text()
    q=q.replace('#include "common.hpp"','#include "'+str(root/'native/grouped_units/numeric_common.hpp')+'"')
    q=q.replace('namespace patterns {','namespace grouped_numeric {')
    start=q.index('struct Dyadic');end=q.index('inline int32_t fixed_parameter',start)
    q=q[:start]+'''inline int scale_exponent(const Words& s,unsigned){uint32_t m=0;for(auto v:s)m=std::max(m,v);return m?32-__builtin_clz(m):0;}
inline double normalized_value(uint32_t v,unsigned,int exponent){return std::ldexp(double(v),-exponent);}
'''+q[end:]
    start=q.index('struct Projection');end=q.index('inline Stream quantization_encode',start)
    q=q[:start]+'''struct Projection {
 unsigned bits;int exponent;Projection(unsigned b,int e):bits(b),exponent(e){width(b);}
 uint32_t word(int64_t x)const{if(x<=0)return 0;__int128 v=x;int shift=exponent-20;
 if(shift>=0)v*=__int128(1)<<shift;else{unsigned n=-shift;v=(v+(__int128(1)<<(n-1)))>>n;}
 return uint32_t(std::min<__int128>(v,grouped::mask(bits)));}
};
'''+q[end:]
    q=re.sub(r'double\(fixed_value\((.*?)\)\)/UNIT',r'normalized_value(\1)',q)
    q=q.replace('error=entropy_encode(corrections)','error=correction_encode(corrections,fmt)')
    q=q.replace('auto errors=entropy_decode({s.count,emodel,Bytes(s.data.begin()+n,s.data.end())})','auto errors=correction_decode({s.count,emodel,Bytes(s.data.begin()+n,s.data.end())},fmt)')
    q=re.sub(r'\bblob\(', 'nblob(', q)
    (out/'grouped_quantization.hpp').write_text(q)
    common=(root/'native/math_structure/common.hpp').read_text()
    common=common.replace('#include "../shared_patterns/quantization.hpp"','#include "grouped_quantization.hpp"')
    common=common.replace('namespace mathematical {','namespace grouped_math {').replace('using namespace patterns;','using namespace grouped_numeric;')
    common=re.sub(r'double\(fixed_value\((.*?)\)\)/UNIT',r'normalized_value(\1)',common)
    common=re.sub(r'\bblob\(', 'nblob(', common)
    common=common.replace('s.count*6+128','s.count*8+128')
    (out/'grouped_math.hpp').write_text(common)
    sources=[]
    for name in ('transforms','sparse','factorization'):
        s=(root/f'native/math_structure/{name}.cpp').read_text()
        s=s.replace('#include "common.hpp"','#include "grouped_math.hpp"').replace('namespace mathematical {','namespace grouped_math {')
        s=re.sub(r'double\(fixed_value\((.*?)\)\)/UNIT',r'normalized_value(\1)',s)
        p=out/f'grouped_{name}.cpp';p.write_text(s);sources.append(p)
    return sources
