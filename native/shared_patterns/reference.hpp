#pragma once
#include "common.hpp"
#include "third_party/gear.hpp"
extern "C" {
#include "third_party/xdelta/xdelta3.h"
}
// Upstream printf-format macros must not escape into Eigen or adapter code.
#undef Q
#undef Z
namespace patterns {
inline uint64_t fingerprint(const Words& s,size_t begin,size_t length){uint64_t h=1469598103934665603ull;
    for(size_t i=begin;i<begin+length;++i){h^=s[i];h*=1099511628211ull;}return h^length;}
inline uint64_t gear(uint16_t v){return GEARv2[v&255]^((GEARv2[v>>8]<<1)|(GEARv2[v>>8]>>63));}
inline size_t cut(const Words& s,size_t begin){size_t end=std::min(s.size(),begin+1024),at=std::min(end,begin+64);uint64_t h=0;
    for(;at<end;++at){h=(h<<1)+gear(s[at]);uint64_t mask=at-begin<256?511:127;if(!(h&mask))return at-begin+1;}return end-begin;}
inline Stream dedup_encode(const Words& s,unsigned bits,bool cdc,uint64_t hash_mask=UINT64_MAX){
    Stream out{s.size(),{}, {}};std::unordered_map<uint64_t,std::vector<unsigned>> hashes;
    std::vector<std::pair<size_t,size_t>> dictionary;
    for(size_t pos=0;pos<s.size();){auto len=cdc?cut(s,pos):std::min<size_t>(256,s.size()-pos);auto hash=fingerprint(s,pos,len)&hash_mask;
        auto& candidates=hashes[hash];unsigned id=UINT32_MAX;
        for(auto candidate:candidates){auto [start,n]=dictionary[candidate];
            if(n==len && std::equal(s.begin()+pos,s.begin()+pos+len,s.begin()+start)){id=candidate;break;}}
        if(id!=UINT32_MAX){var(out.data,0);var(out.data,id);}
        else{var(out.data,len);auto b=packed(Words(s.begin()+pos,s.begin()+pos+len),bits);out.data.insert(out.data.end(),b.begin(),b.end());
            candidates.push_back(dictionary.size());dictionary.push_back({pos,len});}
        pos+=len;
    }return out;
}
inline Words dedup_decode(const Stream& stream,unsigned bits,bool cdc){Reader r{stream.data.data(),stream.data.size()};Words out;
    std::vector<std::pair<size_t,size_t>> dictionary;
    while(out.size()<stream.count){auto n=var(r);if(n){if(n>stream.count-out.size() || n>(cdc?1024:256))throw std::runtime_error("Invalid dedup literal chunk");
            auto words=unpacked(r.take((size_t(n)*bits+7)/8),n,bits);dictionary.push_back({out.size(),n});out.insert(out.end(),words.begin(),words.end());}
        else{auto id=var(r);if(id>=dictionary.size())throw std::runtime_error("Dedup references unavailable chunk");auto [start,len]=dictionary[id];
            if(len>stream.count-out.size())throw std::runtime_error("Dedup expansion exceeds component");
            // Avoid invalidated iterators if appending reallocates the output vector.
            for(size_t i=0;i<len;++i){auto value=out[start+i];out.push_back(value);}}
    }r.end();return out;
}
inline std::vector<unsigned> suffix_array(const Words& ref){std::vector<unsigned> order(ref.size()),rank(ref.begin(),ref.end()),next(ref.size());
    std::iota(order.begin(),order.end(),0);
    for(size_t step=1;step<ref.size();step*=2){auto tail=[&](unsigned at)->int64_t{return at+step<ref.size()?int64_t(rank[at+step]):-1;};
        auto less=[&](unsigned a,unsigned b){return rank[a]!=rank[b]?rank[a]<rank[b]:tail(a)<tail(b);};std::sort(order.begin(),order.end(),less);
        next[order[0]]=0;for(size_t i=1;i<order.size();++i)next[order[i]]=next[order[i-1]]+unsigned(less(order[i-1],order[i]));rank=next;
        if(rank[order.back()]+1==ref.size())break;
    }return order;
}
inline Bytes rlz_parse(const Words& input,size_t prefix){Words ref(input.begin(),input.begin()+prefix);auto sa=suffix_array(ref);Bytes data;
    for(size_t i=prefix;i<input.size();){
        auto compare=[&](unsigned at){size_t j=0;while(at+j<ref.size()&&i+j<input.size()&&ref[at+j]==input[i+j])++j;
            if(at+j==ref.size())return i+j!=input.size();if(i+j==input.size())return false;return ref[at+j]<input[i+j];};
        size_t lo=0,hi=sa.size();while(lo<hi){size_t mid=(lo+hi)/2;if(compare(sa[mid]))lo=mid+1;else hi=mid;}
        size_t length=0,offset=0;for(size_t at:std::array<size_t,2>{lo,lo?lo-1:sa.size()})if(at<sa.size()){
            size_t j=0;while(sa[at]+j<ref.size()&&i+j<input.size()&&ref[sa[at]+j]==input[i+j])++j;
            if(j>length){length=j;offset=sa[at];}}
        if(length){var(data,offset);var(data,length);i+=length;}else{var(data,input[i++]);var(data,0);}
    }return data;
}
inline Bytes wire(const Words& s,unsigned bits){Bytes b;for(auto v:s){b.push_back(v);if(bits==16)b.push_back(v>>8);}return b;}
inline Bytes vcdiff_parse(const Words& s,size_t prefix,unsigned bits){auto ref=wire(Words(s.begin(),s.begin()+prefix),bits),target=wire(Words(s.begin()+prefix,s.end()),bits);
    Bytes output(target.size()*2+4096);usize_t n=0;
    auto status=xd3_encode_memory(target.data(),target.size(),ref.data(),ref.size(),output.data(),&n,output.size(),0);
    if(status)throw std::runtime_error("Xdelta encode failed: "+std::to_string(status));output.resize(n);return output;
}
inline Stream reference_encode(const Words& s,unsigned bits,bool vcdiff){std::set<size_t> candidates;
    if(s.size()>1)candidates.insert(std::min<size_t>(s.size()/2,16));else candidates.insert(0);
    for(size_t n:{size_t(32),size_t(128),size_t(512),size_t(2048)})if(n<s.size()&&n<=s.size()/2)candidates.insert(n);
    Stream best{s.size(),{}, {}};size_t cost=SIZE_MAX;
    for(auto prefix:candidates){auto data=vcdiff?vcdiff_parse(s,prefix,bits):rlz_parse(s,prefix);Bytes model;var(model,prefix);
        auto ref=packed(Words(s.begin(),s.begin()+prefix),bits);model.insert(model.end(),ref.begin(),ref.end());
        auto bytes=data.size()+4*((model.size()+2)/3);if(bytes<cost){cost=bytes;best.model=std::move(model);best.data=std::move(data);}}
    return best;
}
inline Words reference_decode(const Stream& stream,unsigned bits,bool vcdiff){Reader m{stream.model.data(),stream.model.size()};auto prefix=var(m);
    if(prefix>=stream.count&&prefix)throw std::runtime_error("Reference consumes entire component");auto ref=unpacked(m.take((size_t(prefix)*bits+7)/8),prefix,bits);m.end();Words out=ref;
    if(vcdiff){auto source=wire(ref,bits);Bytes raw((stream.count-prefix)*(bits==16?2:1));usize_t got=0;
        auto status=xd3_decode_memory(stream.data.data(),stream.data.size(),source.data(),source.size(),raw.data(),&got,raw.size(),0);
        if(status||got!=raw.size())throw std::runtime_error("Xdelta decode failed or size differs");
        for(size_t i=0;i<raw.size();i+=bits==16?2:1)out.push_back(raw[i]|(bits==16?uint16_t(raw[i+1])<<8:0));
    }else{Reader r{stream.data.data(),stream.data.size()};while(out.size()<stream.count){auto offset=var(r),length=var(r);
        if(length){if(offset>ref.size()||length>ref.size()-offset||length>stream.count-out.size())throw std::runtime_error("RLZ reference outside paid dictionary");
            out.insert(out.end(),ref.begin()+offset,ref.begin()+offset+length);}
        else{if(offset>=(1u<<bits))throw std::runtime_error("Invalid RLZ literal");out.push_back(offset);}}r.end();}
    return out;
}
}
