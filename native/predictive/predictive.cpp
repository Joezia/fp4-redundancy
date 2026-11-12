#include "common.hpp"
#include "predictors.hpp"
#include "dedicated.hpp"
#include "ima.hpp"
namespace predictive {
inline Stream adpcm_encode(const Words& source,unsigned bits,Bytes& parameters) {
    Ima state;Words codes(source.size()),corrections(source.size());int center=1<<(bits-1);
    for(size_t i=0;i<source.size();++i) {
        codes[i]=state.encode(int(source[i])-center);
        corrections[i]=wrap(int64_t(source[i])-state.value-center,bits);
    }
    auto q=entropy_encode(codes),r=entropy_encode(corrections);
    blob(parameters,q.model);put<uint64_t>(parameters,q.data.size());
    q.data.insert(q.data.end(),r.data.begin(),r.data.end());return {source.size(),r.model,q.data};
}
inline Words adpcm_decode(Stream stream,unsigned bits,Reader& parameters) {
    auto model=blob(parameters);auto n=parameters.get<uint64_t>();parameters.end();
    if(n>stream.data.size())throw std::runtime_error("Truncated ADPCM decision stream");
    auto q=entropy_decode({stream.count,model,Bytes(stream.data.begin(),stream.data.begin()+n)});
    stream.data.erase(stream.data.begin(),stream.data.begin()+n);auto residuals=entropy_decode(stream);
    Ima state;int center=1<<(bits-1);Words words(stream.count);
    for(size_t i=0;i<words.size();++i) {
        if(q[i]>15)throw std::runtime_error("ADPCM decision outside 4-bit range");
        state.decode(q[i]);words[i]=wrap(int64_t(residuals[i])+state.value+center,bits);
    }
    return words;
}
Bytes encode(unsigned method,unsigned fmt,unsigned cols,const uint8_t* raw,size_t size) {
    if(method<DELTA || method>INTRA)throw std::runtime_error("Unknown B plugin");
    auto s=unpack(fmt,raw,size);unsigned bits=width(fmt);
    if(!cols || cols>s.size())throw std::runtime_error("Invalid B row width");
    Stream out;Bytes params;
    if(method==JPEGLS)out=jpeg_encode(s,bits,cols);
    else if(method==FPC)out=fpc_encode(s,bits);
    else if(method==ADPCM)out=adpcm_encode(s,bits,params);
    else {
        auto p=choose(method,s,bits,cols);params=p.encoded;Words residuals(s.size());
        for(size_t i=0;i<s.size();++i) {
            auto pred=predict(method,s,i,bits,cols,p);
            residuals[i]=method==XOR?uint16_t(s[i]^uint16_t(pred)):wrap(int64_t(s[i])-pred,bits);
        }
        out=method==SZ?sz_encode(residuals):entropy_encode(residuals);
    }
    Bytes model;put<uint32_t>(model,1);put<uint32_t>(model,1u<<bits);put<uint32_t>(model,cols);
    blob(model,params);blob(model,out.model);put<uint64_t>(model,out.data.size()*8);
    Bytes frame;put<uint32_t>(frame,MAGIC);put<uint64_t>(frame,size);put<uint32_t>(frame,method);
    put<uint32_t>(frame,fmt);put<uint32_t>(frame,0);put<uint32_t>(frame,1);
    put<uint64_t>(frame,s.size());put<uint64_t>(frame,model.size());put<uint64_t>(frame,out.data.size());
    frame.insert(frame.end(),model.begin(),model.end());frame.insert(frame.end(),out.data.begin(),out.data.end());return frame;
}
Bytes decode(const uint8_t* data,size_t size) {
    Reader r{data,size};if(r.get<uint32_t>()!=MAGIC)throw std::runtime_error("Invalid B frame magic");
    auto raw=r.get<uint64_t>();auto method=r.get<uint32_t>(),fmt=r.get<uint32_t>();
    if(method<DELTA||method>INTRA||r.get<uint32_t>()||r.get<uint32_t>()!=1)throw std::runtime_error("Invalid B method/flags");
    size_t n=count(fmt,raw);unsigned bits=width(fmt);Stream stream;
    stream.count=r.get<uint64_t>();auto nm=r.get<uint64_t>(),nd=r.get<uint64_t>();
    if(stream.count!=n)throw std::runtime_error("Invalid B word count");
    auto model=r.take(nm);stream.data=r.take(nd);r.end();Reader m{model.data(),model.size()};
    if(m.get<uint32_t>()!=1 || m.get<uint32_t>()!=(1u<<bits))throw std::runtime_error("Invalid B version/alphabet");
    auto cols=m.get<uint32_t>();if(!cols||cols>n)throw std::runtime_error("Invalid B row width");
    auto params=blob(m);stream.model=blob(m);if(m.get<uint64_t>()!=nd*8)throw std::runtime_error("Invalid B byte count");m.end();
    Words words;
    if(method==ADPCM){Reader p{params.data(),params.size()};words=adpcm_decode(stream,bits,p);}
    else if(method==JPEGLS||method==FPC) {
        if(!params.empty()||!stream.model.empty())throw std::runtime_error("Unexpected dedicated-codec metadata");
        words=method==JPEGLS?jpeg_decode(stream,bits,cols):fpc_decode(stream,bits);
    } else {
        auto p=read_parameters(method,params,n,cols);
        auto residuals=method==SZ?sz_decode(stream):entropy_decode(stream);words.resize(n);
        for(size_t i=0;i<n;++i) {
            if(residuals[i]>=(1u<<bits))throw std::runtime_error("Residual exceeds codeword width");
            auto pred=predict(method,words,i,bits,cols,p);
            words[i]=method==XOR?uint16_t(residuals[i]^uint16_t(pred)):wrap(int64_t(residuals[i])+pred,bits);
        }
    }
    return pack(fmt,words,raw);
}
}
static thread_local std::string error;
static int result(const predictive::Bytes& b,uint8_t** out,size_t* n) {
    *out=static_cast<uint8_t*>(std::malloc(b.size()));if(!*out)throw std::bad_alloc();
    std::memcpy(*out,b.data(),b.size());*n=b.size();return 0;
}
#define API extern "C" __attribute__((visibility("default")))
API int fp4_b_encode(unsigned method,unsigned fmt,unsigned cols,const uint8_t* raw,size_t size,uint8_t** out,size_t* n) {
    try{return result(predictive::encode(method,fmt,cols,raw,size),out,n);}catch(const std::exception& e){error=e.what();return -1;}
}
API int fp4_b_decode(const uint8_t* frame,size_t size,uint8_t** out,size_t* n) {
    try{return result(predictive::decode(frame,size),out,n);}catch(const std::exception& e){error=e.what();return -1;}
}
API void fp4_b_free(void* p){std::free(p);}
API const char* fp4_b_error(){return error.c_str();}
