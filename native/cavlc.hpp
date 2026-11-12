// Complete CAVLC-style integer-codeword adapter, based on OpenH264's
// coeff_token / trailing ones / adaptive levels / total_zeros / run_before.
// Original unsigned codewords are nonnegative levels, with no index remapping.
// No video transform or quantization. nC is the preceding 16-symbol block.
// Extended level-prefix syntax covers every uint16 codeword; native VLC backend.
#pragma once
#include "third_party/OpenH264/tables.hpp"
namespace cavlc_codec {
struct BitOut{
    Bytes data;uint64_t bits=0;
    void write(uint32_t value,unsigned n){
        if(n>32)throw std::runtime_error("CAVLC write width");
        while(n){unsigned used=bits%8;if(!used)data.push_back(0);unsigned take=std::min(n,8-used);
            data.back()|=uint8_t(((value>>(n-take))&((1u<<take)-1))<<(8-used-take));bits+=take;n-=take;}
    }
    void vlc(const uint8_t *entry){if(!entry[1])throw std::runtime_error("CAVLC invalid VLC");write(entry[0],entry[1]);}
};
struct BitIn{
    const Bytes &data;uint64_t bits,position=0;
    uint32_t read(unsigned n){if(n>32||n>bits-position)throw std::runtime_error("CAVLC truncated bits");uint32_t v=0;
        while(n){unsigned used=position%8,take=std::min(n,8-used);v=(v<<take)|((data[position/8]>>(8-used-take))&((1u<<take)-1));position+=take;n-=take;}return v;}
};
inline unsigned nc_index(unsigned n){return n<2?0:n<4?1:n<8?2:3;}
inline uint64_t level_base(unsigned prefix,unsigned suffix){
    return (uint64_t(std::min(15u,prefix))<<suffix)+(prefix>=15&&suffix==0?15:0)+(prefix>=16?(uint64_t(1)<<(prefix-3))-4096:0);
}
inline unsigned suffix_size(unsigned prefix,unsigned suffix){return prefix==14&&suffix==0?4:prefix>=15?prefix-3:suffix;}
inline void level_write(BitOut &out,int32_t level,unsigned &suffix,bool first){
    uint32_t magnitude=level<0?uint32_t(-int64_t(level)):uint32_t(level);
    uint64_t code=2*magnitude-2+(level<0);if(first)code-=2;
    unsigned prefix=0;uint64_t remainder=0;unsigned size=0;
    for(;;++prefix){if(prefix>31)throw std::runtime_error("CAVLC level overflow");size=suffix_size(prefix,suffix);auto base=level_base(prefix,suffix);
        if(code>=base&&code-base<(uint64_t(1)<<size)){remainder=code-base;break;}}
    out.write(0,prefix);out.write(1,1);out.write(uint32_t(remainder),size);
    if(suffix==0)suffix=1;if(magnitude>(3u<<(suffix-1))&&suffix<6)++suffix;
}
inline int32_t level_read(BitIn &in,unsigned &suffix,bool first){
    unsigned prefix=0;while(!in.read(1))if(++prefix>31)throw std::runtime_error("CAVLC invalid level prefix");
    unsigned size=suffix_size(prefix,suffix);uint64_t code=level_base(prefix,suffix)+in.read(size);if(first)code+=2;
    uint64_t magnitude=(code+2)/2;if(magnitude>65535)throw std::runtime_error("CAVLC level outside uint16 range");
    int32_t level=code&1?-int32_t(magnitude):int32_t(magnitude);
    if(suffix==0)suffix=1;if(magnitude>(3u<<(suffix-1))&&suffix<6)++suffix;return level;
}
inline unsigned scalar_vlc(BitIn &in,const uint8_t (*table)[2],unsigned values){
    uint32_t code=0;for(unsigned n=1;n<=16;++n){code=(code<<1)|in.read(1);for(unsigned i=0;i<values;++i)if(table[i][1]==n&&table[i][0]==code)return i;}
    throw std::runtime_error("CAVLC invalid scalar VLC");
}
inline std::pair<unsigned,unsigned> token(BitIn &in,unsigned nc){
    uint32_t code=0;for(unsigned n=1;n<=16;++n){code=(code<<1)|in.read(1);
        for(unsigned total=0;total<=16;++total)for(unsigned trailing=0;trailing<=std::min(3u,total);++trailing){auto *v=openh264::g_kuiVlcCoeffToken[nc][total][trailing];if(v[1]==n&&v[0]==code)return {total,trailing};}}
    throw std::runtime_error("CAVLC invalid coeff token");
}
inline Stream encode(const Symbols &symbols,uint32_t alphabet){
    BitOut out;out.data.reserve(symbols.size()*3+32);unsigned previous=0;
    for(size_t start=0;start<symbols.size();start+=16){
        std::vector<int32_t> levels;std::vector<unsigned> positions;
        for(int i=15;i>=0;--i)if(start+i<symbols.size()&&symbols[start+i]){auto u=symbols[start+i];levels.push_back(int32_t(u));positions.push_back(i);}
        unsigned total=levels.size(),trailing=0;while(trailing<std::min(3u,total)&&std::abs(levels[trailing])==1)++trailing;
        out.vlc(openh264::g_kuiVlcCoeffToken[nc_index(previous)][total][trailing]);previous=total;if(!total)continue;
        for(unsigned i=0;i<trailing;++i)out.write(levels[i]<0,1);
        unsigned suffix=total>10&&trailing<3?1:0;
        for(unsigned i=trailing;i<total;++i)level_write(out,levels[i],suffix,i==trailing&&trailing<3);
        unsigned zeros=positions[0]+1-total;
        if(total<16)out.vlc(openh264::g_kuiVlcTotalZeros[total][zeros]);
        for(unsigned i=0;i+1<total&&zeros;++i){unsigned run=positions[i]-positions[i+1]-1;out.vlc(openh264::g_kuiVlcRunBefore[std::min(zeros,7u)][run]);zeros-=run;}
    }
    Stream s{symbols.size(),{},std::move(out.data)};for(uint32_t v:{1u,alphabet,16u,1u,1u})put<uint32_t>(s.model,v);put<uint64_t>(s.model,out.bits);return s;
}
inline Symbols decode(const Stream &s){
    Reader r{s.model.data(),s.model.size()};if(r.get<uint32_t>()!=1)throw std::runtime_error("CAVLC version");uint32_t a=r.get<uint32_t>();
    if(!a||a>65536||(a&(a-1)))throw std::runtime_error("CAVLC alphabet");for(uint32_t v:{16u,1u,1u})if(r.get<uint32_t>()!=v)throw std::runtime_error("CAVLC configuration");
    auto bits=r.get<uint64_t>();if(r.position!=r.size||(bits+7)/8!=s.data.size()||!bits)throw std::runtime_error("CAVLC bit count");
    if(bits%8&&(s.data.back()&((1u<<(8-bits%8))-1)))throw std::runtime_error("CAVLC padding");
    BitIn in{s.data,bits};Symbols result;result.reserve(s.count);unsigned previous=0;
    for(uint64_t start=0;start<s.count;start+=16){
        auto [total,trailing]=token(in,nc_index(previous));previous=total;std::array<int32_t,16> block{};
        if(total){std::vector<int32_t> levels(total);for(unsigned i=0;i<trailing;++i)levels[i]=in.read(1)?-1:1;
            unsigned suffix=total>10&&trailing<3?1:0;for(unsigned i=trailing;i<total;++i)levels[i]=level_read(in,suffix,i==trailing&&trailing<3);
            unsigned zeros=total<16?scalar_vlc(in,openh264::g_kuiVlcTotalZeros[total],17-total):0;
            int position=int(total+zeros)-1;
            for(unsigned i=0;i<total;++i){if(position<0||position>=16)throw std::runtime_error("CAVLC run position");block[position]=levels[i];
                if(i+1<total){unsigned run=zeros?scalar_vlc(in,openh264::g_kuiVlcRunBefore[std::min(zeros,7u)],std::min(zeros,14u)+1):0;
                    if(run>zeros)throw std::runtime_error("CAVLC run length");zeros-=run;position-=int(run)+1;}}
            if(position!=int(zeros))throw std::runtime_error("CAVLC zero accounting");
        }
        for(unsigned i=0;i<16;++i){int32_t level=block[i];if(level<0)throw std::runtime_error("CAVLC negative codeword");uint32_t u=uint32_t(level);
            if(u>=a)throw std::runtime_error("CAVLC codeword range");if(start+i<s.count)result.push_back(uint16_t(u));else if(u)throw std::runtime_error("CAVLC tail padding");}
    }
    if(in.position!=bits)throw std::runtime_error("CAVLC trailing bits");return result;
}
} // namespace cavlc_codec
