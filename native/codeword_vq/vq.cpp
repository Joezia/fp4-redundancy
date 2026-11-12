// Complete categorical VQ adaptation. Representatives are observed codeword
// vectors; no numeric FP4 interpretation, field splitting, or source permutation.
#include "../shared_patterns/common.hpp"
#include "../third_party/YAECL-Yet-Another-Entropy-Coding-Library/yaecl.hpp"
#include <cmath>
#include <omp.h>

namespace codeword_vq {
using namespace patterns;
constexpr unsigned CODEC=80, Q=1u<<24;
struct Options {unsigned dim=8, centers=16, mode=2, objective=1;};
Options options(unsigned flags) {
    Options o{flags&255,(flags>>8)&255,(flags>>16)&3,(flags>>18)&1};
    if((flags>>19)||!(o.dim==2||o.dim==4||o.dim==8)||
       !(o.centers==4||o.centers==16||o.centers==64)||o.mode>2)
        throw std::runtime_error("Invalid categorical VQ options");
    return o;
}
unsigned workers() {
    const char* s=std::getenv("FP4_VQ_THREADS");if(!s)return 1;
    char* end;long n=std::strtol(s,&end,10);
    if(*end||n<1||n>4096)throw std::runtime_error("FP4_VQ_THREADS must be in 1..4096");return n;
}
struct Distribution {
    std::vector<uint32_t> cdf;
    Bytes save() const {
        Bytes out;put<uint32_t>(out,24);unsigned used=0;
        for(size_t i=0;i+1<cdf.size();++i)used+=cdf[i+1]>cdf[i];
        put<uint32_t>(out,used);
        for(size_t i=0;i+1<cdf.size();++i)if(cdf[i+1]>cdf[i]){
            put<uint16_t>(out,i);put<uint32_t>(out,cdf[i+1]-cdf[i]);}
        return out;
    }
    static Distribution read(const Bytes& data,unsigned alphabet) {
        Reader r{data.data(),data.size()};Distribution d;d.cdf.assign(alphabet+1,0);
        if(r.get<uint32_t>()!=24)throw std::runtime_error("VQ CDF precision differs");
        auto count=r.get<uint32_t>();if(!count||count>alphabet)throw std::runtime_error("VQ alphabet count");
        unsigned previous=0;uint64_t sum=0;
        for(unsigned i=0;i<count;++i){auto v=r.get<uint16_t>();auto f=r.get<uint32_t>();
            if(v>=alphabet||(i&&v<=previous)||!f||sum+f>Q)throw std::runtime_error("Invalid VQ CDF");
            d.cdf[v+1]=f;sum+=f;previous=v;}
        if(sum!=Q)throw std::runtime_error("VQ CDF total differs");
        for(unsigned i=0;i<alphabet;++i)d.cdf[i+1]+=d.cdf[i];r.end();return d;
    }
    double bits(unsigned v) const {auto f=cdf[v+1]-cdf[v];return f?-std::log2(double(f)/Q):1000;}
};
Distribution distribution(const std::vector<uint32_t>& counts) {
    uint64_t n=std::accumulate(counts.begin(),counts.end(),uint64_t(0));
    if(!n)throw std::runtime_error("Empty VQ probability model");
    unsigned used=0;for(auto c:counts)used+=c!=0;
    std::vector<uint32_t> f(counts.size(),0);std::vector<std::pair<uint64_t,unsigned>> remainders;
    uint32_t sum=0;
    for(unsigned i=0;i<counts.size();++i)if(counts[i]){
        uint64_t scaled=uint64_t(counts[i])*(Q-used);f[i]=1+scaled/n;sum+=f[i];
        remainders.emplace_back(scaled%n,i);}
    std::sort(remainders.begin(),remainders.end(),[](auto a,auto b){return a.first!=b.first?a.first>b.first:a.second<b.second;});
    for(unsigned i=0;i<Q-sum;++i)++f[remainders[i].second];
    Distribution d;d.cdf.push_back(0);for(auto v:f)d.cdf.push_back(d.cdf.back()+v);return d;
}
using Vector=std::vector<uint16_t>;
struct Fit {std::vector<Vector> book;std::vector<double> index_cost,error_cost;unsigned nt;};
Vector vector_at(const Words& s,size_t i,unsigned dim) {
    Vector out(dim,0);for(unsigned j=0;j<dim&&i*dim+j<s.size();++j)out[j]=s[i*dim+j];return out;
}
unsigned nearest(const Vector& x,const Fit& f,bool rate) {
    unsigned best=0;double cost=INFINITY;
    for(unsigned k=0;k<f.book.size();++k){double value=rate?f.index_cost[k]:0;
        for(unsigned j=0;j<x.size();++j)value+=rate?f.error_cost[x[j]^f.book[k][j]]:double(x[j]!=f.book[k][j]);
        if(value<cost){cost=value;best=k;}}
    return best;
}
unsigned nearest_at(const Words& s,size_t start,unsigned dim,const Fit& f,bool rate) {
    unsigned best=0;double cost=INFINITY;
    for(unsigned k=0;k<f.book.size();++k){double value=rate?f.index_cost[k]:0;
        for(unsigned j=0;j<dim;++j){unsigned v=start+j<s.size()?s[start+j]:0;
            value+=rate?f.error_cost[v^f.book[k][j]]:double(v!=f.book[k][j]);}
        if(value<cost){cost=value;best=k;}}
    return best;
}
void costs(Fit& f,const std::vector<Vector>& x,const std::vector<unsigned>& ids,unsigned alphabet) {
    std::vector<unsigned> index(f.book.size(),1),errors(alphabet,1);
    for(unsigned i=0;i<x.size();++i){++index[ids[i]];for(unsigned j=0;j<x[i].size();++j)++errors[x[i][j]^f.book[ids[i]][j]];}
    double ni=x.size()+index.size(),ne=x.size()*x[0].size()+alphabet;
    f.index_cost.resize(index.size());f.error_cost.resize(alphabet);
    for(unsigned k=0;k<index.size();++k)f.index_cost[k]=-std::log2(index[k]/ni);
    for(unsigned r=0;r<alphabet;++r)f.error_cost[r]=-std::log2(errors[r]/ne);
}
Fit train(const Words& s,unsigned alphabet,const Options& o) {
    size_t nv=(s.size()+o.dim-1)/o.dim;unsigned nt=std::min<size_t>(nv,2048);
    // One deterministic jittered position per consecutive stratum avoids
    // aliasing a fixed stride against recurring vector patterns. Positions
    // remain increasing; neither source nor encoded vectors are reordered.
    std::vector<Vector> x;for(unsigned i=0;i<nt;++i){size_t lo=size_t(i)*nv/nt,hi=size_t(i+1)*nv/nt;
        uint64_t h=uint64_t(i)+0x9e3779b97f4a7c15ull;h=(h^(h>>30))*0xbf58476d1ce4e5b9ull;
        h=(h^(h>>27))*0x94d049bb133111ebull;h^=h>>31;
        x.push_back(vector_at(s,lo+h%(hi-lo),o.dim));}
    Fit f;f.nt=nt;f.book.push_back(x[0]);std::vector<unsigned> minimum(nt,UINT_MAX);
    for(unsigned k=1;k<o.centers;++k){unsigned farthest=0;
        for(unsigned i=0;i<nt;++i){unsigned distance=0;for(unsigned j=0;j<o.dim;++j)distance+=x[i][j]!=f.book.back()[j];
            minimum[i]=std::min(minimum[i],distance);if(minimum[i]>minimum[farthest])farthest=i;}
        if(!minimum[farthest])break;f.book.push_back(x[farthest]);}
    std::vector<unsigned> ids(nt);for(unsigned i=0;i<nt;++i)ids[i]=nearest(x[i],f,false);
    costs(f,x,ids,alphabet);
    for(unsigned iteration=0;iteration<5;++iteration){
        for(unsigned i=0;i<nt;++i)ids[i]=nearest(x[i],f,o.objective);
        // Medoid update: only an actually assigned, observed vector may become
        // a representative. Lookup tables evaluate the complete cluster cost.
        std::vector<std::vector<unsigned>> groups(f.book.size());
        for(unsigned i=0;i<nt;++i)groups[ids[i]].push_back(i);
        for(unsigned k=0;k<f.book.size();++k)if(!groups[k].empty()){
            std::vector<unsigned> hist(o.dim*alphabet,0);
            for(auto i:groups[k])for(unsigned j=0;j<o.dim;++j)++hist[j*alphabet+x[i][j]];
            std::vector<double> table(o.dim*alphabet,0);
            for(unsigned j=0;j<o.dim;++j)for(unsigned v=0;v<alphabet;++v)if(hist[j*alphabet+v]){
                double cost=0;for(unsigned a=0;a<alphabet;++a)if(hist[j*alphabet+a])
                    cost+=hist[j*alphabet+a]*(o.objective?f.error_cost[a^v]:double(a!=v));
                table[j*alphabet+v]=cost;}
            unsigned best=groups[k][0];double best_cost=INFINITY;
            for(auto i:groups[k]){double cost=0;for(unsigned j=0;j<o.dim;++j)cost+=table[j*alphabet+x[i][j]];
                if(cost<best_cost){best_cost=cost;best=i;}}
            f.book[k]=x[best];}
        for(unsigned i=0;i<nt;++i)ids[i]=nearest(x[i],f,o.objective);
        costs(f,x,ids,alphabet);
    }return f;
}
Stream correction(const Words& errors,const Words& ids,unsigned dim,unsigned k,unsigned alphabet,bool conditioned) {
    std::vector<uint32_t> histogram(alphabet,0);for(auto r:errors)++histogram[r];auto global=distribution(histogram);
    unsigned groups=conditioned?k*dim:0;
    std::vector<Distribution> local(groups);std::vector<bool> enabled(groups,false);
    if(conditioned){std::vector<std::vector<uint32_t>> hist(groups,std::vector<uint32_t>(alphabet,0));
        for(size_t i=0;i<errors.size();++i)++hist[ids[i/dim]*dim+i%dim][errors[i]];
        for(unsigned g=0;g<groups;++g)if(std::accumulate(hist[g].begin(),hist[g].end(),uint64_t(0))){
            auto candidate=distribution(hist[g]);double gain=0;
            for(unsigned v=0;v<alphabet;++v)if(hist[g][v])gain+=hist[g][v]*(global.bits(v)-candidate.bits(v));
            // Model costs include their actual binary length and the outer
            // base64 expansion. Final shared/conditioned selection is exact.
            if(gain>8.0*4/3*(candidate.save().size()+4)){local[g]=std::move(candidate);enabled[g]=true;}}}
    yaecl::ArithmeticCodingEncoder<uint64_t,uint32_t> coder(32);
    for(size_t i=0;i<errors.size();++i){unsigned g=conditioned?ids[i/dim]*dim+i%dim:0;
        const auto& d=conditioned&&enabled[g]?local[g]:global;unsigned r=errors[i];coder.encode_interval(d.cdf[r],d.cdf[r+1],24);}
    coder.flush();Stream out{errors.size(),{},coder.bit_stream.bytes()};
    put<uint32_t>(out.model,1);put<uint32_t>(out.model,alphabet);put<uint64_t>(out.model,coder.bit_stream.size());
    blob(out.model,global.save());put<uint32_t>(out.model,groups);
    for(unsigned g=0;g<groups;++g){put<uint8_t>(out.model,enabled[g]);if(enabled[g])blob(out.model,local[g].save());}
    return out;
}
Words decode_correction(const Stream& stream,const Words& ids,unsigned dim,unsigned k,unsigned alphabet,bool conditioned) {
    Reader r{stream.model.data(),stream.model.size()};
    if(r.get<uint32_t>()!=1||r.get<uint32_t>()!=alphabet)throw std::runtime_error("VQ correction format differs");
    auto bits=r.get<uint64_t>();if(bits<2||bits>INT_MAX||(bits+7)/8!=stream.data.size()||
       (bits%8&&(stream.data.back()&((1u<<(8-bits%8))-1))))throw std::runtime_error("VQ correction length/padding");
    auto global=Distribution::read(blob(r),alphabet);auto groups=r.get<uint32_t>();
    if(groups!=(conditioned?k*dim:0))throw std::runtime_error("VQ context dimensions differ");
    std::vector<Distribution> local(groups);std::vector<bool> enabled(groups,false);
    for(unsigned g=0;g<groups;++g){auto flag=r.get<uint8_t>();if(flag>1)throw std::runtime_error("VQ context flag");enabled[g]=flag;if(flag)local[g]=Distribution::read(blob(r),alphabet);}r.end();
    yaecl::BitStream input;input.assign(stream.data.data(),stream.data.size(),bits);
    yaecl::ArithmeticCodingDecoder<uint64_t,uint32_t> coder(32,input);Words out(stream.count);
    for(size_t i=0;i<out.size();++i){unsigned g=conditioned?ids[i/dim]*dim+i%dim:0;
        const auto& d=conditioned&&enabled[g]?local[g]:global;auto value=coder.cumulative(24);
        auto at=std::upper_bound(d.cdf.begin(),d.cdf.end(),value);if(at==d.cdf.end())throw std::runtime_error("VQ arithmetic symbol");
        unsigned v=at-d.cdf.begin()-1;coder.decode_interval(d.cdf[v],d.cdf[v+1],24);out[i]=v;}
    return out;
}
Stream assemble(const Words& s,unsigned bits,const Options& o,const Fit& fit,const Words& ids,const Words& errors,
                const Stream& index,bool conditioned) {
    auto error=correction(errors,ids,o.dim,fit.book.size(),1u<<bits,conditioned);Stream out{s.size(),{},index.data};
    out.data.insert(out.data.end(),error.data.begin(),error.data.end());
    put<uint32_t>(out.model,1);put<uint16_t>(out.model,o.dim);put<uint16_t>(out.model,o.centers);put<uint16_t>(out.model,fit.book.size());
    put<uint8_t>(out.model,conditioned);put<uint8_t>(out.model,o.objective);put<uint32_t>(out.model,fit.nt);put<uint32_t>(out.model,ids.size());
    Words flat;for(auto& v:fit.book)flat.insert(flat.end(),v.begin(),v.end());blob(out.model,packed(flat,bits));
    blob(out.model,index.model);put<uint64_t>(out.model,index.data.size());blob(out.model,error.model);return out;
}
size_t digits(size_t n){size_t d=1;while(n>=10){n/=10;++d;}return d;}
size_t artifact_cost(const Stream& s){
    // Exact variable part of the NativeA JSON component: table_b64, bytes and
    // payload_bits. All other component/shared header fields are identical.
    size_t model=s.model.size()+20;return s.data.size()+4*((model+2)/3)+digits(s.data.size())+digits(s.data.size()*8);
}
Stream encode(const Words& s,unsigned bits,const Options& o) {
    auto f=train(s,1u<<bits,o);size_t nv=(s.size()+o.dim-1)/o.dim;Words ids(nv),errors(s.size());unsigned threads=workers();
    #pragma omp parallel for num_threads(threads) if(threads>1 && nv>64)
    for(size_t i=0;i<nv;++i)ids[i]=nearest_at(s,i*o.dim,o.dim,f,o.objective);
    std::vector<unsigned> counts(f.book.size(),0),mapping(f.book.size());for(auto id:ids)++counts[id];std::vector<Vector> used;
    for(unsigned k=0;k<f.book.size();++k)if(counts[k]){mapping[k]=used.size();used.push_back(f.book[k]);}
    f.book=std::move(used);for(auto& id:ids)id=mapping[id];
    for(size_t i=0;i<s.size();++i)errors[i]=s[i]^f.book[ids[i/o.dim]][i%o.dim];auto index=entropy_encode(ids);
    if(o.mode<2)return assemble(s,bits,o,f,ids,errors,index,o.mode==1);
    auto shared=assemble(s,bits,o,f,ids,errors,index,false),conditioned=assemble(s,bits,o,f,ids,errors,index,true);
    return artifact_cost(conditioned)<artifact_cost(shared)?conditioned:shared;
}
Words decode(const Stream& stream,unsigned bits) {
    Reader r{stream.model.data(),stream.model.size()};if(r.get<uint32_t>()!=1)throw std::runtime_error("VQ version");
    auto dim=r.get<uint16_t>(),requested=r.get<uint16_t>(),k=r.get<uint16_t>();auto mode=r.get<uint8_t>(),objective=r.get<uint8_t>();
    auto nt=r.get<uint32_t>(),nv=r.get<uint32_t>();
    if(!(dim==2||dim==4||dim==8)||!(requested==4||requested==16||requested==64)||!k||k>requested||mode>1||objective>1||
       !nt||nt>2048||nt>nv||nv!=(stream.count+dim-1)/dim)throw std::runtime_error("VQ geometry differs");
    auto book=unpacked(blob(r),k*dim,bits);auto imodel=blob(r);auto ni=r.get<uint64_t>();auto emodel=blob(r);r.end();
    if(ni>stream.data.size())throw std::runtime_error("VQ index boundary differs");
    auto ids=entropy_decode({nv,imodel,Bytes(stream.data.begin(),stream.data.begin()+ni)});
    for(auto id:ids)if(id>=k)throw std::runtime_error("VQ index outside codebook");
    auto errors=decode_correction({stream.count,emodel,Bytes(stream.data.begin()+ni,stream.data.end())},ids,dim,k,1u<<bits,mode);
    for(size_t i=0;i<errors.size();++i)errors[i]^=book[ids[i/dim]*dim+i%dim];return errors;
}
Bytes encode_frame(unsigned codec,unsigned fmt,unsigned flags,const uint8_t* raw,size_t size) {
    if(codec!=CODEC||width(fmt)>8||size>(1u<<20))throw std::runtime_error("Codeword VQ supports <=1 MiB components of complete 4/8-bit symbols");
    auto s=unpack(fmt,raw,size);auto out=encode(s,width(fmt),options(flags));Bytes model;
    put<uint32_t>(model,1);put<uint32_t>(model,1u<<width(fmt));blob(model,out.model);put<uint64_t>(model,out.data.size()*8);
    Bytes frame;put<uint32_t>(frame,MAGIC);put<uint64_t>(frame,size);put<uint32_t>(frame,CODEC);put<uint32_t>(frame,fmt);put<uint32_t>(frame,0);put<uint32_t>(frame,1);
    put<uint64_t>(frame,s.size());put<uint64_t>(frame,model.size());put<uint64_t>(frame,out.data.size());frame.insert(frame.end(),model.begin(),model.end());frame.insert(frame.end(),out.data.begin(),out.data.end());return frame;
}
Bytes decode_frame(const uint8_t* frame,size_t size) {
    Reader r{frame,size};if(r.get<uint32_t>()!=MAGIC)throw std::runtime_error("Codeword VQ frame magic");auto raw=r.get<uint64_t>();auto codec=r.get<uint32_t>(),fmt=r.get<uint32_t>();
    if(codec!=CODEC||width(fmt)>8||raw>(1u<<20)||r.get<uint32_t>()||r.get<uint32_t>()!=1)throw std::runtime_error("Codeword VQ flags/format");
    auto n=r.get<uint64_t>();if(n!=count(fmt,raw))throw std::runtime_error("Codeword VQ count differs");auto nm=r.get<uint64_t>(),nd=r.get<uint64_t>();auto model=r.take(nm),data=r.take(nd);r.end();
    Reader m{model.data(),model.size()};if(m.get<uint32_t>()!=1||m.get<uint32_t>()!=(1u<<width(fmt)))throw std::runtime_error("Codeword VQ model");
    auto control=blob(m);if(m.get<uint64_t>()!=nd*8)throw std::runtime_error("Codeword VQ bytes differ");m.end();return pack(fmt,decode({n,control,data},width(fmt)),raw);
}
} // namespace codeword_vq
static thread_local std::string error;
static int result(const predictive::Bytes& b,uint8_t** out,size_t* n){*out=static_cast<uint8_t*>(std::malloc(b.size()));if(!*out)throw std::bad_alloc();std::memcpy(*out,b.data(),b.size());*n=b.size();return 0;}
#define API extern "C" __attribute__((visibility("default")))
API int fp4_vq_encode(unsigned method,unsigned fmt,unsigned flags,const uint8_t* raw,size_t size,uint8_t** out,size_t* n){try{return result(codeword_vq::encode_frame(method,fmt,flags,raw,size),out,n);}catch(const std::exception& e){error=e.what();return -1;}}
API int fp4_vq_decode(const uint8_t* data,size_t size,uint8_t** out,size_t* n){try{return result(codeword_vq::decode_frame(data,size),out,n);}catch(const std::exception& e){error=e.what();return -1;}}
API const char* fp4_vq_error(){return error.c_str();}
API void fp4_vq_free(void* p){std::free(p);}
