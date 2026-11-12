#pragma once
#include "common.hpp"
#include <set>
namespace grouped_numeric {
using Bytes=grouped::Bytes;using Words=std::vector<uint32_t>;
using Stream=grouped::Stream;using grouped::Reader;using grouped::put;
enum {VQ=41,PQ=42,OPQ=43,RVQ=44,AQ=45};
inline unsigned width(unsigned b){if(b!=8&&b!=16&&b!=32)throw std::runtime_error("Numerical word width");return b;}
inline void nblob(Bytes& b,const Bytes& v){put<uint32_t>(b,v.size());b.insert(b.end(),v.begin(),v.end());}
inline Bytes nblob(Reader& r){return r.take(r.get<uint32_t>());}
inline Stream entropy_encode(const Words& s){predictive::Words v;v.reserve(s.size());for(auto x:s){if(x>65535)throw std::runtime_error("Numerical limb exceeds 16 bits");v.push_back(x);}auto t=predictive::entropy_encode(v);return {s.size(),t.model,t.data};}
inline Words entropy_decode(const Stream& s){auto v=predictive::entropy_decode({s.count,s.model,s.data});return Words(v.begin(),v.end());}
// Corrections retain exact word bits. The fixed nibble serialization is billed
// and decoded without any floating-point operation or original-input cache.
inline Stream correction_encode(const Words& errors,unsigned b){Words n;for(auto v:errors)for(unsigned j=0;j<b;j+=4)n.push_back((v>>j)&15);auto s=entropy_encode(n);s.count=errors.size();Bytes m;put<uint32_t>(m,b);nblob(m,s.model);s.model=std::move(m);return s;}
inline Words correction_decode(const Stream& s,unsigned b){Reader r{s.model.data(),s.model.size()};if(r.get<uint32_t>()!=b)throw std::runtime_error("Correction unit differs");auto m=nblob(r);r.end();auto n=entropy_decode({s.count*(b/4),m,s.data});Words out(s.count);for(size_t i=0;i<n.size();++i){if(n[i]>15)throw std::runtime_error("Correction nibble range");out[i/(b/4)]|=n[i]<<(4*(i%(b/4)));}return out;}
}
