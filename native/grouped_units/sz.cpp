#include "common.hpp"
#include "SZ3/lossless/Lossless_zstd.hpp"
namespace grouped {
Stream sz_store(const Words& residuals,unsigned b){auto s=entropy(residuals,b,0);SZ3::Lossless_zstd zstd(3);Bytes data(SZ3::Lossless_zstd::compress_bound(s.data.size())+8);
 auto size=zstd.compress(s.data.data(),s.data.size(),data.data(),data.size());data.resize(size);put<uint64_t>(s.model,s.data.size());s.data=std::move(data);return s;}
Words sz_load(Stream s,unsigned b){if(s.model.size()<8)throw std::runtime_error("SZ wide size missing");Reader r{s.model.data()+s.model.size()-8,8};auto n=r.get<uint64_t>();
 if(!n||n>s.count*4+32)throw std::runtime_error("SZ wide Huffman size");Bytes data(n);auto p=data.data();SZ3::Lossless_zstd zstd(3);
 if(zstd.decompress(s.data.data(),s.data.size(),p,n)!=n)throw std::runtime_error("SZ wide Zstd size");s.model.resize(s.model.size()-8);s.data=std::move(data);return entropy_decode(s,b,0);}
}
