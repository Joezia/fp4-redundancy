// FPC v1.1 adaptation; complete upstream notice follows in FPC-NOTICE.txt.
// MODIFIED 2026-11-09: buffer API, unsigned 4/8/16-bit words, width-aware hashes,
// bounded tables, safe scalar accesses; retain FCM/DFCM choice, XOR and byte-prefix coding.
#pragma once
#include "common.hpp"
#include "predictors.hpp"
#include "charls/charls.h"
#include "SZ3/lossless/Lossless_zstd.hpp"
namespace predictive {
struct FpcState {
    std::vector<uint16_t> fcm=std::vector<uint16_t>(4096),dfcm=std::vector<uint16_t>(4096);
    uint32_t hash=0,dhash=0;uint16_t last=0;unsigned bits;
    explicit FpcState(unsigned w):bits(w){}
    uint16_t first() const{return fcm[hash];}
    uint16_t second() const{return wrap(uint32_t(last)+dfcm[dhash],bits);}
    void update(uint16_t v) {
        auto stride=wrap(int64_t(v)-last,bits);fcm[hash]=v;dfcm[dhash]=stride;
        hash=((hash<<6)^(v>>(bits>16?bits-16:0)))&4095;
        dhash=((dhash<<2)^(stride>>(bits>24?bits-24:0)))&4095;last=v;
    }
};
inline Stream fpc_encode(const Words& s,unsigned bits) {
    FpcState state(bits);Stream out{s.size(),{},Bytes((s.size()+1)/2,0)};
    for(size_t i=0;i<s.size();++i) {
        auto a=uint16_t(s[i]^state.first()),b=uint16_t(s[i]^state.second());bool use=b<a;
        auto residue=use?b:a;unsigned bytes=residue>255?2:residue?1:0;
        auto prefix=unsigned(use)*8+bytes;
        out.data[i/2]|=uint8_t(prefix<<((i%2)?0:4));
        for(unsigned j=0;j<bytes;++j)out.data.push_back(uint8_t(residue>>(j*8)));
        state.update(s[i]);
    }
    return out;
}
inline Words fpc_decode(const Stream& s,unsigned bits) {
    size_t prefixes=(s.count+1)/2;Reader r{s.data.data(),s.data.size()};auto header=r.take(prefixes);
    if(s.count%2 && (header.back()&15))throw std::runtime_error("FPC nonzero final prefix padding");
    FpcState state(bits);Words result(s.count);
    for(size_t i=0;i<result.size();++i) {
        unsigned p=(header[i/2]>>((i%2)?0:4))&15,n=p&7;
        if(n>(bits+7)/8)throw std::runtime_error("FPC invalid residual prefix");
        unsigned v=0;for(unsigned j=0;j<n;++j)v|=unsigned(r.get<uint8_t>())<<(j*8);
        if(v>=(1u<<bits))throw std::runtime_error("FPC residual outside native word");
        result[i]=uint16_t(v^((p&8)?state.second():state.first()));state.update(result[i]);
    }
    r.end();return result;
}
// CharLS is used unchanged: context correction, run mode and adaptive Golomb coding.
// NEAR=0 is explicit, and checked again on decode. No arithmetic replacement.
inline Stream jpeg_encode(const Words& s,unsigned bits,size_t cols) {
    Stream out{s.size(),{}, {}};
    for(size_t base=0;base<s.size();) {
        size_t w=std::min<size_t>(cols,65535),h=1;
        if(cols<=65535 && base%cols==0)h=std::min<size_t>((s.size()-base)/cols,65535);
        if(!h)h=1;
        w=std::min(w,s.size()-base);size_t n=w*h;
        Bytes raw;raw.reserve(n*((bits+7)/8));
        for(size_t i=base;i<base+n;++i) {raw.push_back(uint8_t(s[i]));if(bits==16)raw.push_back(uint8_t(s[i]>>8));}
        charls::jpegls_encoder encoder;encoder.frame_info({uint32_t(w),uint32_t(h),int32_t(bits),1}).near_lossless(0);
        Bytes data(encoder.estimated_destination_size());encoder.destination(data);data.resize(encoder.encode(raw));
        blob(out.data,data);base+=n;
    }
    return out;
}
inline Words jpeg_decode(const Stream& s,unsigned bits,size_t cols) {
    Reader r{s.data.data(),s.data.size()};Words words;words.reserve(s.count);
    while(words.size()<s.count) {
        auto data=blob(r);charls::jpegls_decoder decoder;decoder.source(data).read_header();auto f=decoder.frame_info();
        size_t w=std::min<size_t>(cols,65535),h=1,base=words.size();
        if(cols<=65535 && base%cols==0)h=std::min<size_t>((s.count-base)/cols,65535);
        if(!h)h=1;w=std::min<size_t>(w,s.count-base);
        if(decoder.near_lossless()!=0 || f.bits_per_sample!=int(bits) || f.component_count!=1 || f.width!=w || f.height!=h ||
           decoder.destination_size()!=w*h*((bits+7)/8))throw std::runtime_error("Invalid lossless JPEG-LS frame layout");
        Bytes raw(decoder.destination_size());decoder.decode(raw);
        for(size_t i=0;i<raw.size();i+=(bits+7)/8)words.push_back(raw[i] | (bits==16?uint16_t(raw[i+1])<<8:0));
    }
    r.end();return words;
}
inline Stream sz_encode(const Words& residuals) {
    auto out=entropy_encode(residuals,0);SZ3::Lossless_zstd zstd(3);
    Bytes data(SZ3::Lossless_zstd::compress_bound(out.data.size())+8);
    data.resize(zstd.compress(out.data.data(),out.data.size(),data.data(),data.size()));
    // Huffman table stays separately measurable; preserve the original Huffman + Zstd engines.
    put<uint64_t>(out.model,out.data.size());out.data=std::move(data);return out;
}
inline Words sz_decode(Stream stream) {
    if(stream.model.size()<8)throw std::runtime_error("Missing SZ3 Huffman size");
    Reader tail{stream.model.data()+stream.model.size()-8,8};size_t n=tail.get<uint64_t>();
    if(n>stream.count*4+16 || !n)throw std::runtime_error("SZ3 decoded Huffman size outside bound");
    Bytes raw(n);uint8_t* ptr=raw.data();SZ3::Lossless_zstd zstd(3);
    auto got=zstd.decompress(stream.data.data(),stream.data.size(),ptr,raw.size());
    if(got!=raw.size())throw std::runtime_error("SZ3 lossless size mismatch");
    stream.model.resize(stream.model.size()-8);stream.data=std::move(raw);return entropy_decode(stream,0);
}
}
