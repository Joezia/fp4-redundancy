"""Sparse categorical medoid adaptation of the existing complete VQ plugin."""
def generate(root,out):
    s=(root/'native/codeword_vq/vq.cpp').read_text()
    s=s[:s.index('Bytes encode_frame(')]+'}\n'
    s=s[s.index('namespace codeword_vq {'):]
    s=s.replace('namespace codeword_vq {','namespace grouped_vq {').replace('using namespace patterns;', 'using namespace grouped_numeric;\n'+'''
struct SparseCost {std::unordered_map<uint32_t,double> values;double unseen=0;
 double operator[](uint32_t v)const{auto i=values.find(v);return i==values.end()?unseen:i->second;}};
inline Bytes packed(const Words& s,unsigned bits){Bytes out;for(auto v:s)grouped::word(out,v,bits);return out;}
inline Words unpacked(const Bytes& b,size_t n,unsigned bits){Reader r{b.data(),b.size()};Words out(n);for(auto& v:out)v=grouped::word(r,bits);r.end();return out;}
''')
    s=s.replace('std::vector<uint16_t>','std::vector<uint32_t>')
    s=s.replace('std::vector<double> index_cost,error_cost;','std::vector<double> index_cost;SparseCost error_cost;')
    start=s.index('void costs(');end=s.index('Fit train(',start)
    s=s[:start]+'''void costs(Fit& f,const std::vector<Vector>& x,const std::vector<unsigned>& ids,uint64_t alphabet){
 std::vector<unsigned> index(f.book.size(),1);std::unordered_map<uint32_t,unsigned> errors;
 for(unsigned i=0;i<x.size();++i){++index[ids[i]];for(unsigned j=0;j<x[i].size();++j)++errors[x[i][j]^f.book[ids[i]][j]];}
 double ni=x.size()+index.size(),ne=double(x.size()*x[0].size())+double(alphabet);
 f.index_cost.resize(index.size());for(unsigned k=0;k<index.size();++k)f.index_cost[k]=-std::log2(index[k]/ni);
 f.error_cost.values.clear();f.error_cost.unseen=std::log2(ne);
 for(auto [v,n]:errors)f.error_cost.values[v]=-std::log2((n+1)/ne);
}
'''+s[end:]
    s=s.replace('Fit train(const Words& s,unsigned alphabet','Fit train(const Words& s,uint64_t alphabet')
    start=s.index('            std::vector<unsigned> hist(');end=s.index('            unsigned best=groups[k][0]',start)
    s=s[:start]+'''            std::vector<std::map<uint32_t,unsigned>> hist(o.dim);
            for(auto i:groups[k])for(unsigned j=0;j<o.dim;++j)++hist[j][x[i][j]];
            std::vector<std::map<uint32_t,double>> table(o.dim);
            for(unsigned j=0;j<o.dim;++j)for(auto [v,n]:hist[j]){
                double cost=0;for(auto [a,count]:hist[j])cost+=count*(o.objective?f.error_cost[a^v]:double(a!=v));
                table[j][v]=cost;}
'''+s[end:]
    s=s.replace('table[j*alphabet+x[i][j]]','table[j][x[i][j]]')
    s=s.replace('train(s,1u<<bits,o)','train(s,uint64_t(1)<<bits,o)')
    s=s.replace('auto error=correction(errors,ids,o.dim,fit.book.size(),1u<<bits,conditioned);', '''Words nib;for(auto v:errors)for(unsigned j=0;j<bits;j+=4)nib.push_back((v>>j)&15);
    auto error=correction(nib,ids,o.dim*(bits/4),fit.book.size(),16,conditioned);''')
    s=s.replace('auto errors=decode_correction({stream.count,emodel,Bytes(stream.data.begin()+ni,stream.data.end())},ids,dim,k,1u<<bits,mode);', '''auto nib=decode_correction({stream.count*(bits/4),emodel,Bytes(stream.data.begin()+ni,stream.data.end())},ids,dim*(bits/4),k,16,mode);
    Words errors(stream.count);for(size_t i=0;i<nib.size();++i)errors[i/(bits/4)]|=nib[i]<<(4*(i%(bits/4)));''')
    # Actual outer grouped prefix is 40 bytes, not the old 20-byte wrapper.
    s=s.replace('s.model.size()+20','s.model.size()+40')
    import re
    s=re.sub(r'\bblob\(', 'nblob(',s)
    s='#include "'+str(root/'native/grouped_units/numeric_common.hpp')+'"\n#include "'+str(root/'native/third_party/YAECL-Yet-Another-Entropy-Coding-Library/yaecl.hpp')+'"\n#include <omp.h>\n'+s
    p=out/'grouped_vq.hpp';p.write_text(s)
