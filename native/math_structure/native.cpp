#include "common.hpp"
namespace mathematical {
Stream approximation_encode(const Words& source,unsigned fmt,unsigned method){Approximation fitted;
    if(method<=PCA)fitted=fit_transform(source,fmt,method);else if(method<=KSVD)fitted=fit_sparse(source,fmt,method);else fitted=fit_factors(source,fmt,method);
    auto control=parameters(method,fitted);
    // Round-trip the actual serialized parameters and coefficient stream before
    // constructing corrections. This is the exact same reconstruction as decode.
    auto coefficients=encode_i32(fitted.data);auto state=read_parameters(method,control,source.size());state.data=decode_i32(coefficients);
    auto prediction=reconstruct(method,state,source.size());Projection projection(fmt,state.exponent);Words errors(source.size());
    for(size_t i=0;i<source.size();++i)errors[i]=source[i]^projection.word(prediction[i]);
    return join(source.size(),1u<<width(fmt),control,{coefficients,entropy_encode(errors)});
}
Words approximation_decode(const Stream& stream,unsigned fmt,unsigned method){auto parts=split(stream,1u<<width(fmt));if(parts.streams.size()!=2)throw std::runtime_error("D numerical method needs coefficients and exact correction");
    auto state=read_parameters(method,parts.control,stream.count);
    if(parts.streams[0].count!=state.data.size()*2||parts.streams[1].count!=stream.count)throw std::runtime_error("D coefficient/correction counts differ");
    state.data=decode_i32(parts.streams[0]);auto prediction=reconstruct(method,state,stream.count);auto errors=entropy_decode(parts.streams[1]);Projection projection(fmt,state.exponent);Words output(stream.count);
    for(size_t i=0;i<output.size();++i)output[i]=errors[i]^projection.word(prediction[i]);return output;
}
Bytes encode(unsigned method,unsigned fmt,const uint8_t* raw,size_t size){if(method<BWT_MTF||method>CIRCULANT||size>COMPONENT_LIMIT)throw std::runtime_error("Unknown D method or component exceeds 1 MiB");
    auto source=unpack(fmt,raw,size);auto stream=exact(method)?exact_encode(source,width(fmt),method):approximation_encode(source,fmt,method);
    Bytes frame;put<uint32_t>(frame,MAGIC);put<uint64_t>(frame,size);put<uint32_t>(frame,method);put<uint32_t>(frame,fmt);put<uint32_t>(frame,0);put<uint32_t>(frame,1);
    put<uint64_t>(frame,source.size());put<uint64_t>(frame,stream.model.size());put<uint64_t>(frame,stream.data.size());frame.insert(frame.end(),stream.model.begin(),stream.model.end());frame.insert(frame.end(),stream.data.begin(),stream.data.end());return frame;
}
Bytes decode(const uint8_t* frame,size_t size){Reader r{frame,size};if(r.get<uint32_t>()!=MAGIC)throw std::runtime_error("Invalid D frame magic");auto raw=r.get<uint64_t>();auto method=r.get<uint32_t>(),fmt=r.get<uint32_t>();
    if(method<BWT_MTF||method>CIRCULANT||raw>COMPONENT_LIMIT||r.get<uint32_t>()||r.get<uint32_t>()!=1)throw std::runtime_error("Invalid D method/size/flags");auto n=count(fmt,raw);
    if(r.get<uint64_t>()!=n)throw std::runtime_error("D symbol count differs");auto nm=r.get<uint64_t>(),nd=r.get<uint64_t>();Stream stream{n,r.take(nm),r.take(nd)};r.end();
    auto words=exact(method)?exact_decode(stream,width(fmt),method):approximation_decode(stream,fmt,method);return pack(fmt,words,raw);
}
}
static thread_local std::string error;
static int result(const predictive::Bytes& b,uint8_t** out,size_t* n){*out=static_cast<uint8_t*>(std::malloc(b.size()));if(!*out)throw std::bad_alloc();std::memcpy(*out,b.data(),b.size());*n=b.size();return 0;}
#define API extern "C" __attribute__((visibility("default")))
API int fp4_d_encode(unsigned method,unsigned fmt,unsigned reserved,const uint8_t* raw,size_t size,uint8_t** out,size_t* n){try{if(reserved)throw std::runtime_error("D plugin has no runtime layout override");return result(mathematical::encode(method,fmt,raw,size),out,n);}catch(const std::exception& e){error=e.what();return -1;}}
API int fp4_d_decode(const uint8_t* data,size_t size,uint8_t** out,size_t* n){try{return result(mathematical::decode(data,size),out,n);}catch(const std::exception& e){error=e.what();return -1;}}
API const char* fp4_d_error(){return error.c_str();}
API void fp4_d_free(void* p){std::free(p);}
