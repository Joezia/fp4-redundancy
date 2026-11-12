// Private whole-codeword probability factorization. There is one arithmetic
// update per original symbol; no field/bit-plane streams or data permutation.
#pragma once
namespace a2_tree {
constexpr uint32_t Q=1u<<24;
inline unsigned width(uint32_t a){if(a<2||a>65536||(a&(a-1)))throw std::runtime_error("A2 alphabet");unsigned w=0;while((1u<<w)<a)++w;return w;}
struct Interval{uint32_t symbol,low,high;};
template<class Model> Interval word(Model &model,unsigned w,uint32_t input,bool decoding){
    uint32_t lo=0,hi=Q,prefix=1,symbol=0;
    model.begin_word();
    for(unsigned d=0;d<w;++d){
        uint32_t p1=model.predict(prefix,d),leaves=1u<<(w-d-1);
        if(p1>32768)throw std::runtime_error("A2 probability");
        uint32_t left=std::clamp<uint32_t>((uint64_t(hi-lo)*(32768-p1))>>15,leaves,hi-lo-leaves);
        uint32_t middle=lo+left;
        unsigned bit=decoding?input>=middle:(input>>(w-d-1))&1;
        model.observe(bit,prefix,d);symbol=(symbol<<1)|bit;prefix=(prefix<<1)|bit;
        if(bit)lo=middle;else hi=middle;
    }
    model.finish_word(symbol);return {symbol,lo,hi};
}
template<class Model> Stream encode(const Symbols &symbols,uint32_t a){
    Model model(symbols,a);yaecl::ArithmeticCodingEncoder<uint64_t,uint32_t> coder(32);unsigned w=width(a);
    for(auto s:symbols){auto i=word(model,w,s,false);coder.encode_interval(i.low,i.high,24);}
    coder.flush();Stream out{symbols.size(),{},coder.bit_stream.bytes()};
    put<uint32_t>(out.model,1);put<uint32_t>(out.model,a);model.save(out.model);put<uint64_t>(out.model,coder.bit_stream.size());return out;
}
template<class Model> Symbols decode(const Stream &stream){
    Reader r{stream.model.data(),stream.model.size()};if(r.get<uint32_t>()!=1)throw std::runtime_error("A2 model version");
    uint32_t a=r.get<uint32_t>();unsigned w=width(a);Model model(r,a);auto bits=r.get<uint64_t>();
    if(r.position!=r.size||bits<2||bits>INT32_MAX||(bits+7)/8!=stream.data.size())throw std::runtime_error("A2 bit count");
    if(bits%8&&(stream.data.back()&((1u<<(8-bits%8))-1)))throw std::runtime_error("A2 padding");
    yaecl::BitStream input;input.assign(stream.data.data(),stream.data.size(),int(bits));
    yaecl::ArithmeticCodingDecoder<uint64_t,uint32_t> coder(32,input);Symbols out;out.reserve(stream.count);
    for(uint64_t i=0;i<stream.count;++i){auto x=word(model,w,coder.cumulative(24),true);coder.decode_interval(x.low,x.high,24);out.push_back(uint16_t(x.symbol));}return out;
}
} // namespace a2_tree
