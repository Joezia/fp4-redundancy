// Private buffer bridge. Complete B plugins fix their own predictor and backend.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace predictive {
using Bytes = std::vector<uint8_t>;
using Words = std::vector<uint16_t>;
constexpr uint32_t MAGIC = 0x31414e46;
constexpr size_t MAX_RAW = 4u << 20; // uint16_t bridge also fits A1's 16 MiB cap for packed FP4.
template<class T> void put(Bytes& b, T v) {
    using U = std::make_unsigned_t<T>;
    for (unsigned i=0; i<sizeof(T); ++i) b.push_back(uint8_t(U(v) >> (8*i)));
}
struct Reader {
    const uint8_t* p; size_t n, pos=0;
    template<class T> T get() {
        using U = std::make_unsigned_t<T>;
        if (sizeof(T)>n-pos) throw std::runtime_error("Truncated B metadata");
        U v=0; for (unsigned i=0;i<sizeof(T);++i) v |= U(p[pos++]) << (8*i);
        // Signed fields are explicitly two's complement, without out-of-range casts.
        if constexpr (std::is_signed_v<T>) {
            if (v > U(std::numeric_limits<T>::max())) return T(-1 - T(U(~v)));
        }
        return T(v);
    }
    Bytes take(size_t s) {
        if (s>n-pos) throw std::runtime_error("Truncated B payload");
        Bytes b(p+pos,p+pos+s); pos+=s; return b;
    }
    void end() const { if(pos!=n) throw std::runtime_error("Trailing B bytes"); }
};
struct Stream { uint64_t count; Bytes model, data; };
inline void blob(Bytes& b,const Bytes& v) { put<uint32_t>(b,v.size()); b.insert(b.end(),v.begin(),v.end()); }
inline Bytes blob(Reader& r) { return r.take(r.get<uint32_t>()); }
inline unsigned width(unsigned fmt) {
    if(fmt>6) throw std::runtime_error("Unknown B format");
    return fmt<2?4:fmt==5?16:8;
}
inline size_t count(unsigned fmt,size_t size) {
    width(fmt);
    if(!size || size>MAX_RAW || (fmt==5 && size%2)) throw std::runtime_error("Invalid B component length");
    return fmt<2?size*2:fmt==5?size/2:size;
}
inline Words unpack(unsigned fmt,const uint8_t* raw,size_t size) {
    Words s(count(fmt,size));
    for(size_t i=0;i<s.size();++i) {
        if(fmt<2) s[i]=(raw[i/2] >> (((i%2) ^ fmt)*4)) & 15;
        else if(fmt==5) s[i]=raw[2*i] | (uint16_t(raw[2*i+1])<<8);
        else s[i]=raw[i];
    }
    return s;
}
inline Bytes pack(unsigned fmt,const Words& s,size_t size) {
    if(s.size()!=count(fmt,size)) throw std::runtime_error("B decoded symbol count differs");
    Bytes b(size,0); auto mask=(1u<<width(fmt))-1;
    for(size_t i=0;i<s.size();++i) {
        if(s[i]>mask) throw std::runtime_error("B codeword outside native alphabet");
        if(fmt<2) b[i/2] |= uint8_t(s[i] << (((i%2)^fmt)*4));
        else if(fmt==5) { b[2*i]=uint8_t(s[i]); b[2*i+1]=uint8_t(s[i]>>8); }
        else b[i]=uint8_t(s[i]);
    }
    return b;
}
// Exact A1 backend, including its integer CDF precision and finalization rules.
extern "C" int fp4_native_encode(unsigned,unsigned,unsigned,const uint8_t*,size_t,uint8_t**,size_t*);
extern "C" int fp4_native_decode(const uint8_t*,size_t,uint8_t**,size_t*);
extern "C" void fp4_native_free(void*);
extern "C" const char* fp4_native_error();
inline Stream entropy_encode(const Words& words,unsigned codec=1) {
    Bytes raw;raw.reserve(words.size()*2); for(auto v:words) put<uint16_t>(raw,v);
    uint8_t* output=nullptr;size_t n=0;
    if(fp4_native_encode(codec,5,0,raw.data(),raw.size(),&output,&n)) throw std::runtime_error(fp4_native_error());
    Bytes frame(output,output+n);fp4_native_free(output); Reader r{frame.data(),frame.size()};
    if(r.get<uint32_t>()!=MAGIC || r.get<uint64_t>()!=raw.size() || r.get<uint32_t>()!=codec ||
       r.get<uint32_t>()!=5 || r.get<uint32_t>() || r.get<uint32_t>()!=1) throw std::runtime_error("A1 bridge header mismatch");
    Stream s; s.count=r.get<uint64_t>(); auto nm=r.get<uint64_t>(),nd=r.get<uint64_t>();
    s.model=r.take(nm); s.data=r.take(nd); r.end(); return s;
}
inline Words entropy_decode(const Stream& s,unsigned codec=1) {
    if(!s.count || s.count>MAX_RAW*2) throw std::runtime_error("B residual count exceeds bound");
    // The A1 byte bridge is capped at 16 MiB. B splits residual streams below that limit.
    Bytes frame;put<uint32_t>(frame,MAGIC);put<uint64_t>(frame,s.count*2);
    put<uint32_t>(frame,codec);put<uint32_t>(frame,5);put<uint32_t>(frame,0);put<uint32_t>(frame,1);
    put<uint64_t>(frame,s.count);put<uint64_t>(frame,s.model.size());put<uint64_t>(frame,s.data.size());
    frame.insert(frame.end(),s.model.begin(),s.model.end());frame.insert(frame.end(),s.data.begin(),s.data.end());
    uint8_t* out=nullptr;size_t n=0;
    if(fp4_native_decode(frame.data(),frame.size(),&out,&n)) throw std::runtime_error(fp4_native_error());
    Words words(n/2);for(size_t i=0;i<words.size();++i)words[i]=out[2*i]|(uint16_t(out[2*i+1])<<8);
    fp4_native_free(out); if(words.size()!=s.count)throw std::runtime_error("A1 bridge decoded count differs"); return words;
}
// No overflow or implementation-defined signed right shift in prediction.
inline int64_t floor_shift(int64_t x,unsigned bits) {
    const int64_t d=int64_t(1)<<bits; return x>=0?x/d:-((-x+d-1)/d);
}
inline uint16_t wrap(int64_t x,unsigned bits) { return uint16_t(uint64_t(x)&((1u<<bits)-1)); }
}
