#include "common.hpp"
#include "SZ3/encoder/HuffmanEncoder.hpp"
#include "../third_party/ryg_rans/rans_byte.h"
namespace grouped {
struct Dist{Words alphabet;std::vector<uint32_t> freq,cdf;std::unordered_map<uint64_t,uint32_t> rank;unsigned precision;};
Dist distribution(const Words& s,unsigned bits,unsigned codec){std::map<uint64_t,uint32_t> counts;for(auto v:s)++counts[v];Dist d;
 if(codec==2&&counts.size()>65536)throw std::runtime_error("RESOURCE_LIMIT: rANS original normalized CDF supports at most 65536 observed joint symbols");
 if(counts.size()>(1u<<24))throw std::runtime_error("RESOURCE_LIMIT: observed alphabet exceeds 24-bit arithmetic CDF capacity");
 d.precision=codec==2?12:24;while((uint64_t(1)<<d.precision)<counts.size())++d.precision;
 for(auto [v,n]:counts){d.rank[v]=d.alphabet.size();d.alphabet.push_back(v);}
 uint32_t total=1u<<d.precision,remaining=total-counts.size(),sum=0;std::vector<std::pair<uint64_t,unsigned>> remainders;
 for(auto [v,n]:counts){uint64_t scaled=uint64_t(n)*remaining;d.freq.push_back(1+scaled/s.size());sum+=d.freq.back();remainders.emplace_back(scaled%s.size(),remainders.size());}
 std::sort(remainders.begin(),remainders.end(),[](auto a,auto b){return a.first!=b.first?a.first>b.first:a.second<b.second;});
 for(unsigned i=0;i<total-sum;++i)++d.freq[remainders[i].second];d.cdf.push_back(0);for(auto f:d.freq)d.cdf.push_back(d.cdf.back()+f);return d;
}
Stream entropy(const Words& s,unsigned bits,unsigned codec){auto d=distribution(s,bits,codec);Stream out{s.size(),{}, {}};
 put<uint32_t>(out.model,1);put<uint32_t>(out.model,codec);put<uint32_t>(out.model,bits);put<uint32_t>(out.model,d.precision);put<uint32_t>(out.model,d.alphabet.size());
 for(size_t i=0;i<d.alphabet.size();++i){word(out.model,d.alphabet[i],bits);put<uint32_t>(out.model,d.freq[i]);}
 if(codec==0){std::vector<uint32_t> ranks;for(auto v:s)ranks.push_back(d.rank[v]);SZ3::HuffmanEncoder<uint32_t> encoder;encoder.preprocess_encode(ranks,0);
  Bytes model(encoder.size_est());auto m=model.data();encoder.save(m);model.resize(m-model.data());gblob(out.model,model);
  out.data.resize(s.size()*4+32);auto p=out.data.data();encoder.encode(ranks,p);out.data.resize(p-out.data.data());
 }else if(codec==1){yaecl::ArithmeticCodingEncoder<uint64_t,uint32_t> coder(32);for(auto v:s){auto rank=d.rank[v];coder.encode_interval(d.cdf[rank],d.cdf[rank+1],24);}coder.flush();put<uint64_t>(out.model,coder.bit_stream.size());out.data=coder.bit_stream.bytes();
 }else if(codec==2){std::vector<RansEncSymbol> entries(d.alphabet.size());for(size_t i=0;i<entries.size();++i)RansEncSymbolInit(&entries[i],d.cdf[i],d.freq[i],d.precision);
  Bytes data(s.size()*8+32);auto end=data.data()+data.size(),p=end;RansState state;RansEncInit(&state);for(auto it=s.rbegin();it!=s.rend();++it)RansEncPutSymbol(&state,&p,&entries[d.rank[*it]]);RansEncFlush(&state,&p);out.data=Bytes(p,end);
 }else throw std::runtime_error("Unsupported grouped entropy codec");return out;
}
Words entropy_decode(const Stream& stream,unsigned bits,unsigned codec){Reader r{stream.model.data(),stream.model.size()};
 if(r.get<uint32_t>()!=1||r.get<uint32_t>()!=codec||r.get<uint32_t>()!=bits)throw std::runtime_error("Grouped entropy version/codec/width");
 unsigned precision=r.get<uint32_t>(),n=r.get<uint32_t>();if(!n||n>stream.count||(codec==2?(precision<12||precision>16):precision!=24))throw std::runtime_error("Grouped alphabet/CDF bound");
 Words alphabet;std::vector<uint32_t> f,cdf{0};uint64_t sum=0;
 for(unsigned i=0;i<n;++i){auto v=word(r,bits);auto frequency=r.get<uint32_t>();if((i&&v<=alphabet.back())||!frequency||sum+frequency>(uint64_t(1)<<precision))throw std::runtime_error("Invalid grouped distribution");
  alphabet.push_back(v);f.push_back(frequency);sum+=frequency;cdf.push_back(sum);}
 if(sum!=(uint64_t(1)<<precision))throw std::runtime_error("Grouped frequency sum");Words out;out.reserve(stream.count);
 if(codec==0){auto model=gblob(r);SZ3::HuffmanEncoder<uint32_t> decoder;auto m=static_cast<const uint8_t*>(model.data());size_t remaining=model.size();decoder.load(m,remaining);if(remaining)throw std::runtime_error("Trailing grouped Huffman model");
  auto p=static_cast<const uint8_t*>(stream.data.data());remaining=stream.data.size();auto ranks=decoder.decode(p,stream.count,remaining);if(remaining)throw std::runtime_error("Trailing grouped Huffman payload");for(auto rank:ranks){if(rank>=n)throw std::runtime_error("Grouped Huffman rank");out.push_back(alphabet[rank]);}
 }else if(codec==1){auto bits_used=r.get<uint64_t>();if(bits_used<2||bits_used>INT_MAX||(bits_used+7)/8!=stream.data.size()||(bits_used%8&&(stream.data.back()&((1u<<(8-bits_used%8))-1))))throw std::runtime_error("Grouped AC length/padding");
  yaecl::BitStream input;input.assign(stream.data.data(),stream.data.size(),bits_used);yaecl::ArithmeticCodingDecoder<uint64_t,uint32_t> decoder(32,input);
  for(size_t i=0;i<stream.count;++i){auto value=decoder.cumulative(24);auto at=std::upper_bound(cdf.begin(),cdf.end(),value);unsigned rank=at-cdf.begin()-1;if(rank>=n)throw std::runtime_error("Grouped AC symbol");decoder.decode_interval(cdf[rank],cdf[rank+1],24);out.push_back(alphabet[rank]);}
 }else{Reader p{stream.data.data(),stream.data.size()};auto state=p.get<uint32_t>();if(state<RANS_BYTE_L||state>=uint64_t(RANS_BYTE_L)*256)throw std::runtime_error("Grouped rANS state");
  std::vector<uint32_t> inverse(1u<<precision);for(unsigned i=0;i<n;++i)std::fill(inverse.begin()+cdf[i],inverse.begin()+cdf[i+1],i);
  for(size_t j=0;j<stream.count;++j){auto i=inverse[RansDecGet(&state,precision)];out.push_back(alphabet[i]);RansDecAdvanceStep(&state,cdf[i],f[i],precision);while(state<RANS_BYTE_L)state=(state<<8)|p.get<uint8_t>();}
  p.end();if(state!=RANS_BYTE_L)throw std::runtime_error("Grouped rANS ending");}
 r.end();return out;
}
} // namespace grouped
