#pragma once
#include "../predictive/common.hpp"
#include "../third_party/YAECL-Yet-Another-Entropy-Coding-Library/yaecl.hpp"
#include <map>
#include <unordered_map>
#include <array>
#include <climits>
#include <numeric>
namespace grouped {
using Bytes=predictive::Bytes;
using Words=std::vector<uint64_t>;
using predictive::put;using predictive::Reader;
constexpr uint32_t MAGIC=0x31505747;
constexpr size_t LIMIT=8u<<20;
struct Stream{size_t count;Bytes model,data;};
inline void gblob(Bytes& b,const Bytes& v){put<uint64_t>(b,v.size());b.insert(b.end(),v.begin(),v.end());}
inline Bytes gblob(Reader& r){return r.take(r.get<uint64_t>());}
inline uint64_t mask(unsigned b){return b==64?UINT64_MAX:(uint64_t(1)<<b)-1;}
inline uint64_t wrap(int64_t x,unsigned b){return uint64_t(x)&mask(b);}
inline int64_t floor_shift(int64_t x,unsigned b){return x>=0?x/(int64_t(1)<<b):-1-((-1-x)/(int64_t(1)<<b));}
inline void word(Bytes& out,uint64_t v,unsigned b){for(unsigned i=0;i<(b+7)/8;++i)out.push_back(uint8_t(v>>(8*i)));}
inline uint64_t word(Reader& r,unsigned b){uint64_t v=0;for(unsigned i=0;i<(b+7)/8;++i)v|=uint64_t(r.get<uint8_t>())<<(8*i);return v;}
inline void var(Bytes& out,uint64_t v){do{out.push_back(uint8_t(v&127)|(v>=128?128:0));v>>=7;}while(v);}
inline uint64_t var(Reader& r){uint64_t v=0;for(unsigned i=0;i<10;++i){auto c=r.get<uint8_t>();if(i==9&&c>1)throw std::runtime_error("Grouped varint overflow");v|=uint64_t(c&127)<<(7*i);if(!(c&128))return v;}throw std::runtime_error("Grouped varint length");}
inline Words unpack(const uint8_t* p,size_t size,unsigned bits){if(!size||size>LIMIT||bits<8||bits>64||bits%8||size%(bits/8))throw std::runtime_error("Invalid grouped word buffer");
 Reader r{p,size};Words out(size/(bits/8));for(auto& v:out)v=word(r,bits);return out;}
inline Bytes pack(const Words& s,unsigned bits){Bytes out;out.reserve(s.size()*(bits/8));for(auto v:s){if(v>mask(bits))throw std::runtime_error("Grouped word outside width");word(out,v,bits);}return out;}
Stream entropy(const Words&,unsigned,unsigned=1);
Words entropy_decode(const Stream&,unsigned,unsigned=1);
Stream encode_method(unsigned,unsigned,unsigned,const Words&);
Words decode_method(unsigned,unsigned,unsigned,const Stream&);
Stream legacy_match(unsigned,unsigned,const Words&);
Words legacy_unmatch(unsigned,unsigned,const Stream&);
Stream numeric_encode(unsigned,unsigned,const Words&);
Words numeric_decode(unsigned,unsigned,const Stream&);
Stream categorical_encode(unsigned,unsigned,const Words&);
Words categorical_decode(unsigned,const Stream&);
Stream sz_store(const Words&,unsigned);
Words sz_load(Stream,unsigned);
inline Bytes frame(unsigned method,unsigned bits,unsigned cols,const Stream& s){Bytes b;put<uint32_t>(b,MAGIC);put<uint32_t>(b,method);put<uint32_t>(b,bits);put<uint32_t>(b,cols);
 put<uint64_t>(b,s.count);put<uint64_t>(b,s.model.size());put<uint64_t>(b,s.data.size());b.insert(b.end(),s.model.begin(),s.model.end());b.insert(b.end(),s.data.begin(),s.data.end());return b;}
} // namespace grouped
