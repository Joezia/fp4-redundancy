#pragma once
#include "../predictive/common.hpp"
#include <array>
#include <climits>
#include <map>
#include <mutex>
#include <numeric>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
namespace patterns {
using namespace predictive;
enum Method {RLE=30,LZ77=31,LZ78=32,LZSS=33,LZW=34,REPAIR=35,SEQUITUR=36,
             DEDUP=37,CDC=38,VCDIFF=39,RLZ=40,VQ=41,PQ=42,OPQ=43,RVQ=44,AQ=45};
inline void var(Bytes& b,uint32_t v){do{b.push_back(uint8_t(v&127)|(v>=128?128:0));v>>=7;}while(v);}
inline uint32_t var(Reader& r){uint32_t v=0;for(unsigned i=0;i<5;++i){auto c=r.get<uint8_t>();
    if(i==4 && c>15)throw std::runtime_error("C integer overflow");v|=uint32_t(c&127)<<(7*i);if(!(c&128))return v;}
    throw std::runtime_error("Invalid C varint");}
struct BitWriter {Bytes data;size_t bits=0;
    void put(uint32_t v,unsigned n){for(unsigned i=n;i;--i){if(bits%8==0)data.push_back(0);data.back()|=uint8_t(((v>>(i-1))&1)<<(7-bits%8));++bits;}}
};
struct BitReader {const Bytes& data;size_t bits,pos=0;
    uint32_t get(unsigned n){if(n>bits-pos)throw std::runtime_error("C bitstream truncated");uint32_t v=0;while(n--){v=(v<<1)|((data[pos/8]>>(7-pos%8))&1);++pos;}return v;}
    void end(){if(pos!=bits || (bits+7)/8!=data.size() || (bits%8 && (data.back()&((1u<<(8-bits%8))-1))))throw std::runtime_error("C trailing bits/padding");}
};
inline Bytes packed(const Words& s,unsigned bits){BitWriter w;for(auto v:s)w.put(v,bits);return w.data;}
inline Words unpacked(const Bytes& b,size_t n,unsigned bits){BitReader r{b,n*bits};Words s(n);for(auto& v:s)v=uint16_t(r.get(bits));r.end();return s;}
inline uint64_t pair_key(uint32_t a,uint32_t b){return (uint64_t(a)<<32)|b;}
inline void copy(Words& out,size_t distance,size_t length,size_t n){
    if(!distance||distance>out.size()||length>n-out.size())throw std::runtime_error("Invalid C backward reference");
    while(length--)out.push_back(out[out.size()-distance]);
}
inline Stream rle_encode(const Words& s){Stream r{s.size(),{}, {}};
    for(size_t i=0;i<s.size();){size_t j=i+1;while(j<s.size()&&s[j]==s[i])++j;var(r.data,j-i);var(r.data,s[i]);i=j;}return r;}
inline Words rle_decode(const Stream& stream){Reader r{stream.data.data(),stream.data.size()};Words out;
    while(out.size()<stream.count){auto n=var(r),v=var(r);if(!n||v>65535||n>stream.count-out.size())throw std::runtime_error("Invalid RLE run");out.insert(out.end(),n,uint16_t(v));}r.end();return out;}
struct MatchFinder {
    const Words& s;std::unordered_map<uint64_t,int> head;std::vector<int> prev;
    explicit MatchFinder(const Words& v):s(v),prev(v.size(),-1){}
    uint64_t hash(size_t i)const{return (uint64_t(s[i])<<32)|(uint64_t(s[i+1])<<16)|s[i+2];}
    void insert(size_t i){if(i+2>=s.size())return;auto key=hash(i);auto it=head.find(key);if(it!=head.end())prev[i]=it->second;head[key]=i;}
    std::pair<size_t,size_t> find(size_t i,size_t window,size_t max_length)const{
        size_t length=0,distance=0;if(i+2>=s.size())return {0,0};auto it=head.find(hash(i));if(it==head.end())return {0,0};
        // Exhaustive hash chain within the selected window, with full symbol verification.
        for(int j=it->second;j>=0 && i-size_t(j)<=window;j=prev[j]){
            size_t n=0;while(n<max_length&&i+n<s.size()&&s[j+n]==s[i+n])++n;
            if(n>length){length=n;distance=i-size_t(j);if(n==max_length)break;}}
        return length>=3?std::make_pair(distance,length):std::make_pair(size_t(0),size_t(0));
    }
};
inline Stream lz_encode(const Words& s,bool ss){Stream r{s.size(),{}, {}};MatchFinder finder(s);
    size_t i=0;while(i<s.size()){
        size_t flags_at=r.data.size();unsigned flags=0;if(ss)r.data.push_back(0);
        for(unsigned token=0;token<(ss?8u:1u)&&i<s.size();++token){
            auto [distance,length]=finder.find(i,ss?4096:65535,ss?18:258);size_t used;
            if(ss){if(length){uint16_t code=uint16_t(((distance-1)<<4)|(length-3));put<uint16_t>(r.data,code);used=length;}
                else{flags|=1u<<token;var(r.data,s[i]);used=1;}}
            else{var(r.data,distance);var(r.data,length);used=length;
                var(r.data,i+length<s.size()?s[i+length]:65536);if(i+length<s.size())++used;}
            for(size_t j=0;j<used;++j)finder.insert(i+j);i+=used;
        }
        if(ss)r.data[flags_at]=flags;
    }return r;
}
inline Words lz_decode(const Stream& stream,bool ss){Reader r{stream.data.data(),stream.data.size()};Words out;
    while(out.size()<stream.count){unsigned flags=ss?r.get<uint8_t>():0;
        for(unsigned token=0;token<(ss?8u:1u)&&out.size()<stream.count;++token){
            if(ss){if(flags&(1u<<token)){auto v=var(r);if(v>65535)throw std::runtime_error("Invalid LZSS literal");out.push_back(v);}
                else{auto code=r.get<uint16_t>();copy(out,(code>>4)+1,(code&15)+3,stream.count);}}
            else{auto distance=var(r),length=var(r),v=var(r);if(length){if(length>258||distance>65535)throw std::runtime_error("Invalid LZ77 match");copy(out,distance,length,stream.count);}
                else if(distance)throw std::runtime_error("LZ77 empty match has distance");
                if(v<65536){if(out.size()==stream.count)throw std::runtime_error("LZ77 extra literal");out.push_back(v);}
                else if(v!=65536||out.size()!=stream.count)throw std::runtime_error("LZ77 invalid final token");}
        }
    }r.end();return out;
}
inline Stream lz78_encode(const Words& s){Stream r{s.size(),{}, {}};std::unordered_map<uint64_t,uint32_t> dict;uint32_t prefix=0,next=1;
    for(auto v:s){auto key=pair_key(prefix,v);auto it=dict.find(key);if(it!=dict.end())prefix=it->second;
        else{var(r.data,prefix);var(r.data,v);dict.emplace(key,next++);prefix=0;}}
    if(prefix){var(r.data,prefix);var(r.data,65536);}return r;
}
inline Words lz78_decode(const Stream& stream){Reader r{stream.data.data(),stream.data.size()};Words out;
    std::vector<std::pair<uint32_t,uint16_t>> dict(1);std::vector<uint16_t> reversed;
    while(out.size()<stream.count){auto p=var(r),v=var(r);if(p>=dict.size()||v>65536)throw std::runtime_error("Invalid LZ78 phrase");
        reversed.clear();for(auto q=p;q;q=dict[q].first){reversed.push_back(dict[q].second);if(reversed.size()>stream.count-out.size())throw std::runtime_error("LZ78 overrun");}
        if(v==65536 && reversed.size()!=stream.count-out.size())throw std::runtime_error("LZ78 final phrase mismatch");
        for(auto it=reversed.rbegin();it!=reversed.rend();++it)out.push_back(*it);
        if(v!=65536){if(out.size()==stream.count)throw std::runtime_error("LZ78 overrun");out.push_back(v);dict.push_back({p,uint16_t(v)});}
    }r.end();return out;
}
inline Stream lzw_encode(const Words& s,unsigned bits){uint32_t alphabet=1u<<bits,next=alphabet,limit=1u<<20;
    std::unordered_map<uint64_t,uint32_t> dict;std::vector<uint32_t> codes;uint32_t phrase=s[0];
    for(size_t i=1;i<s.size();++i){auto key=pair_key(phrase,s[i]);auto it=dict.find(key);
        if(it!=dict.end())phrase=it->second;else{codes.push_back(phrase);if(next<limit)dict.emplace(key,next++);phrase=s[i];}}
    codes.push_back(phrase);BitWriter writer;unsigned width=bits+1;next=alphabet;
    for(size_t i=0;i<codes.size();++i){writer.put(codes[i],width);if(i&&next<limit){++next;if(next==(1u<<width)&&width<20)++width;}}
    Stream result{s.size(),{},writer.data};put<uint32_t>(result.model,codes.size());put<uint64_t>(result.model,writer.bits);return result;
}
inline Words lzw_decode(const Stream& stream,unsigned bits){Reader m{stream.model.data(),stream.model.size()};auto nc=m.get<uint32_t>();auto nb=m.get<uint64_t>();m.end();
    if(!nc||nc>stream.count||nb>stream.data.size()*8)throw std::runtime_error("Invalid LZW code count");BitReader r{stream.data,size_t(nb)};
    uint32_t alphabet=1u<<bits,limit=1u<<20,next=alphabet;unsigned width=bits+1;
    std::vector<std::pair<uint32_t,uint16_t>> dict;Words out,reversed;uint32_t previous=0;
    auto expand=[&](uint32_t code){reversed.clear();while(code>=alphabet){if(code-alphabet>=dict.size())throw std::runtime_error("Unknown LZW dictionary code");auto p=dict[code-alphabet];reversed.push_back(p.second);code=p.first;
            if(reversed.size()>stream.count)throw std::runtime_error("LZW phrase exceeds component");}reversed.push_back(code);};
    for(uint32_t i=0;i<nc;++i){auto code=r.get(width);bool special=i&&code==next;
        if(code>next||(!i&&code>=alphabet))throw std::runtime_error("Invalid LZW code");expand(special?previous:code);auto first=reversed.back();
        size_t needed=reversed.size()+unsigned(special);if(needed>stream.count-out.size())throw std::runtime_error("LZW overrun");
        for(auto it=reversed.rbegin();it!=reversed.rend();++it)out.push_back(*it);if(special)out.push_back(first);
        if(i&&next<limit){dict.push_back({previous,first});++next;if(next==(1u<<width)&&width<20)++width;}previous=code;
    }r.end();if(out.size()!=stream.count)throw std::runtime_error("LZW decoded length differs");return out;
}
}
