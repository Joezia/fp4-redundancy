// M-ary KT/context-tree weighting adaptation. Reference: cran/BCT and
// Willems/Shtarkov/Tjalkens; sparse count storage reused from Nayuki.
// Integer posterior updates avoid platform-dependent floating-point CDFs.
#pragma once
#include "third_party/Nayuki/PpmModel.hpp"
namespace ctw_codec {
constexpr uint32_t DEPTH=3, CAP=65536, W=32768, Q=1u<<24;
class Model {
    struct Node {
        nayuki_ppm::Frequencies counts;
        uint32_t alpha=W/2;
        std::unordered_map<uint32_t,std::unique_ptr<Node>> children;
        Node(uint32_t a,bool root):counts(a,root){}
    };
    uint32_t alphabet,nodes=1;
    std::unique_ptr<Node> root;
    std::array<uint32_t,DEPTH> history{};
    std::array<Node*,DEPTH+1> path{};
    uint32_t kt(Node *n,uint32_t s)const{
        return uint32_t(uint64_t(2*n->counts.before(s)+s)*Q/(2*n->counts.observations()+alphabet));
    }
    std::array<uint32_t,DEPTH+1> prefixes(uint32_t s)const{
        std::array<uint32_t,DEPTH+1> p{};
        p[DEPTH]=kt(path[DEPTH],s);
        for(int i=DEPTH-1;i>=0;--i)p[i]=(uint64_t(path[i]->alpha)*kt(path[i],s)+uint64_t(W-path[i]->alpha)*p[i+1])/W;
        return p;
    }
public:
    explicit Model(uint32_t a):alphabet(a),root(new Node(a,true)){}
    void prepare(){
        Node *n=root.get();unsigned existing=0;
        for(;existing<DEPTH;++existing){auto it=n->children.find(history[existing]);if(it==n->children.end())break;n=it->second.get();}
        if(nodes+DEPTH-existing>CAP){root.reset(new Node(alphabet,true));nodes=1;history.fill(0);}
        path[0]=root.get();
        for(unsigned i=0;i<DEPTH;++i){auto &child=path[i]->children[history[i]];if(!child){child.reset(new Node(alphabet,false));++nodes;}path[i+1]=child.get();}
    }
    uint32_t cdf(uint32_t s)const{return prefixes(s)[0];}
    void update(uint32_t s){
        auto low=prefixes(s),high=prefixes(s+1);
        for(unsigned i=0;i<DEPTH;++i){
            uint64_t pk=kt(path[i],s+1)-kt(path[i],s),pc=high[i+1]-low[i+1],a=path[i]->alpha;
            uint64_t denominator=a*pk+(W-a)*pc;
            if(!denominator)throw std::runtime_error("CTW zero probability");
            path[i]->alpha=uint32_t(std::clamp<uint64_t>((a*pk*W+denominator/2)/denominator,1,W-1));
        }
        for(auto *n:path)n->counts.increment(s);
        for(unsigned i=DEPTH-1;i>0;--i)history[i]=history[i-1];history[0]=s;
    }
};
inline Stream encode(const Symbols &symbols,uint32_t alphabet){
    Model model(alphabet);yaecl::ArithmeticCodingEncoder<uint64_t,uint32_t> coder(32);
    for(auto s:symbols){model.prepare();auto lo=model.cdf(s),hi=model.cdf(s+1);coder.encode_interval(lo,hi,24);model.update(s);}
    coder.flush();Stream out{symbols.size(),{},coder.bit_stream.bytes()};
    for(uint32_t v:{1u,alphabet,DEPTH,CAP,nayuki_ppm::RESCALE,15u,24u})put<uint32_t>(out.model,v);
    put<uint64_t>(out.model,coder.bit_stream.size());return out;
}
inline Symbols decode(const Stream &stream){
    Reader r{stream.model.data(),stream.model.size()};
    if(r.get<uint32_t>()!=1)throw std::runtime_error("CTW version");
    uint32_t a=r.get<uint32_t>();
    if(!a||a>65536||(a&(a-1)))throw std::runtime_error("CTW alphabet");
    for(uint32_t v:{DEPTH,CAP,nayuki_ppm::RESCALE,15u,24u})if(r.get<uint32_t>()!=v)throw std::runtime_error("CTW configuration");
    auto bits=r.get<uint64_t>();
    if(r.position!=r.size||bits<2||bits>INT32_MAX||(bits+7)/8!=stream.data.size())throw std::runtime_error("CTW bit count");
    if(bits%8&&(stream.data.back()&((1u<<(8-bits%8))-1)))throw std::runtime_error("CTW padding");
    yaecl::BitStream input;input.assign(stream.data.data(),stream.data.size(),int(bits));
    yaecl::ArithmeticCodingDecoder<uint64_t,uint32_t> coder(32,input);Model model(a);Symbols out;out.reserve(stream.count);
    for(uint64_t i=0;i<stream.count;++i){model.prepare();uint32_t scaled=coder.cumulative(24),lo=0,hi=a;
        while(hi-lo>1){auto mid=(lo+hi)/2;if(model.cdf(mid)>scaled)hi=mid;else lo=mid;}
        coder.decode_interval(model.cdf(lo),model.cdf(lo+1),24);out.push_back(uint16_t(lo));model.update(lo);
    }return out;
}
} // namespace ctw_codec
