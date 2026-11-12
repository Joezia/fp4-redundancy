#include "common.hpp"
#include "../shared_patterns/reference.hpp"
namespace grouped {
Stream legacy_match(unsigned method,unsigned bits,const Words& s){
 if(bits>16)throw std::runtime_error("Integer matcher supports at most 16-bit terminals");predictive::Words input(s.begin(),s.end());
 if(method==38){if(bits!=8)throw std::runtime_error("CDC uses actual packed bytes");Stream out{s.size(),{}, {}};
  for(unsigned v:{2u,32u,128u,512u,255u,63u})put<uint32_t>(out.model,v);
  std::unordered_map<uint64_t,std::vector<unsigned>> hashes;std::vector<std::pair<size_t,size_t>> dictionary;
  for(size_t pos=0;pos<s.size();){size_t end=std::min(s.size(),pos+512),at=std::min(end,pos+32);uint64_t h=0;
   for(;at<end;++at){h=(h<<1)+patterns::GEARv2[s[at]];if(!(h&(at-pos<128?255:63))){++at;break;}}
   size_t len=at-pos;auto hash=patterns::fingerprint(input,pos,len);auto& candidates=hashes[hash];unsigned id=UINT_MAX;
   for(auto c:candidates){auto [p,n]=dictionary[c];if(n==len&&std::equal(s.begin()+pos,s.begin()+pos+len,s.begin()+p)){id=c;break;}}
   if(id==UINT_MAX){var(out.data,len);for(size_t j=pos;j<pos+len;++j)out.data.push_back(s[j]);candidates.push_back(dictionary.size());dictionary.emplace_back(pos,len);}
   else{var(out.data,0);var(out.data,id);}pos+=len;
  }return out;
 }
 bool ss=method==33;size_t window=(ss?4096u:65535u)*4/bits,max_length=ss?18:258;
 Stream out{s.size(),{}, {}};put<uint32_t>(out.model,1);put<uint32_t>(out.model,window);put<uint32_t>(out.model,3);put<uint32_t>(out.model,max_length);
 patterns::MatchFinder finder(input);size_t i=0;
 while(i<s.size()){size_t flags_at=out.data.size();unsigned flags=0;if(ss)out.data.push_back(0);
  for(unsigned token=0;token<(ss?8u:1u)&&i<s.size();++token){auto [distance,length]=finder.find(i,window,max_length);size_t used;
   if(ss){if(length){put<uint16_t>(out.data,uint16_t(((distance-1)<<4)|(length-3)));used=length;}
    else{flags|=1u<<token;var(out.data,s[i]);used=1;}}
   else{var(out.data,distance);var(out.data,length);used=length;var(out.data,i+length<s.size()?s[i+length]:65536);if(i+length<s.size())++used;}
   for(size_t j=0;j<used;++j)finder.insert(i+j);i+=used;
  }if(ss)out.data[flags_at]=flags;
 }return out;
}
Words legacy_unmatch(unsigned method,unsigned bits,const Stream& s){Reader r{s.model.data(),s.model.size()};
 if(method==38){for(unsigned v:{2u,32u,128u,512u,255u,63u})if(r.get<uint32_t>()!=v)throw std::runtime_error("Packed CDC configuration");r.end();
  auto values=patterns::dedup_decode({s.count,{},s.data},8,true);return Words(values.begin(),values.end());}
 bool ss=method==33;for(unsigned v:{1u,unsigned((ss?4096:65535)*4/bits),3u,ss?18u:258u})if(r.get<uint32_t>()!=v)throw std::runtime_error("Grouped LZ window configuration");r.end();
 auto values=patterns::lz_decode({s.count,{},s.data},ss);return Words(values.begin(),values.end());
}
} // namespace grouped
