#include "common.hpp"
#include "grammar.hpp"
#include "reference.hpp"
#include "quantization.hpp"
namespace patterns {
Bytes encode(unsigned method,unsigned fmt,const uint8_t* raw,size_t size){
    if(method<RLE||method>AQ)throw std::runtime_error("Unknown C method");auto s=unpack(fmt,raw,size);unsigned bits=width(fmt);Stream out;
    switch(method){
        case RLE:out=rle_encode(s);break;
        case LZ77:out=lz_encode(s,false);break;
        case LZSS:out=lz_encode(s,true);break;
        case LZ78:out=lz78_encode(s);break;
        case LZW:out=lzw_encode(s,bits);break;
        case REPAIR:out=repair_encode(s,1u<<bits);break;
        case SEQUITUR:out=sequitur_encode(s,1u<<bits);break;
        case DEDUP:case CDC:out=dedup_encode(s,bits,method==CDC);break;
        case VCDIFF:case RLZ:out=reference_encode(s,bits,method==VCDIFF);break;
        default:out=quantization_encode(s,fmt,method);
    }
    Bytes model;put<uint32_t>(model,1);put<uint32_t>(model,1u<<bits);blob(model,out.model);put<uint64_t>(model,out.data.size()*8);
    Bytes frame;put<uint32_t>(frame,MAGIC);put<uint64_t>(frame,size);put<uint32_t>(frame,method);put<uint32_t>(frame,fmt);put<uint32_t>(frame,0);put<uint32_t>(frame,1);
    put<uint64_t>(frame,s.size());put<uint64_t>(frame,model.size());put<uint64_t>(frame,out.data.size());frame.insert(frame.end(),model.begin(),model.end());frame.insert(frame.end(),out.data.begin(),out.data.end());return frame;
}
Bytes decode(const uint8_t* frame,size_t size){Reader r{frame,size};if(r.get<uint32_t>()!=MAGIC)throw std::runtime_error("Invalid C frame magic");
    auto raw=r.get<uint64_t>();auto method=r.get<uint32_t>(),fmt=r.get<uint32_t>();if(method<RLE||method>AQ||r.get<uint32_t>()||r.get<uint32_t>()!=1)throw std::runtime_error("Invalid C method/flags");
    auto n=count(fmt,raw);unsigned bits=width(fmt);if(r.get<uint64_t>()!=n)throw std::runtime_error("Invalid C symbol count");auto nm=r.get<uint64_t>(),nd=r.get<uint64_t>();auto model=r.take(nm),data=r.take(nd);r.end();
    Reader m{model.data(),model.size()};if(m.get<uint32_t>()!=1||m.get<uint32_t>()!=(1u<<bits))throw std::runtime_error("Invalid C version/alphabet");
    auto params=blob(m);if(m.get<uint64_t>()!=nd*8)throw std::runtime_error("Invalid C payload byte count");m.end();Stream stream{n,params,data};Words words;
    if(method!=LZW && method<REPAIR && !params.empty())throw std::runtime_error("Unexpected C1 parameters");
    switch(method){
        case RLE:words=rle_decode(stream);break;
        case LZ77:words=lz_decode(stream,false);break;
        case LZSS:words=lz_decode(stream,true);break;
        case LZ78:words=lz78_decode(stream);break;
        case LZW:words=lzw_decode(stream,bits);break;
        case REPAIR:case SEQUITUR:words=grammar_decode(stream,1u<<bits);break;
        case DEDUP:case CDC:if(!params.empty())throw std::runtime_error("Unexpected dedup parameters");words=dedup_decode(stream,bits,method==CDC);break;
        case VCDIFF:case RLZ:words=reference_decode(stream,bits,method==VCDIFF);break;
        default:words=quantization_decode(stream,fmt,method);
    }
    return pack(fmt,words,raw);
}
}
static thread_local std::string error;
static int result(const predictive::Bytes& b,uint8_t** out,size_t* n){*out=static_cast<uint8_t*>(std::malloc(b.size()));if(!*out)throw std::bad_alloc();std::memcpy(*out,b.data(),b.size());*n=b.size();return 0;}
#define API extern "C" __attribute__((visibility("default")))
API int fp4_c_encode(unsigned method,unsigned fmt,unsigned reserved,const uint8_t* raw,size_t size,uint8_t** out,size_t* n){
    try{if(reserved)throw std::runtime_error("C plugins do not take a geometry override");return result(patterns::encode(method,fmt,raw,size),out,n);}catch(const std::exception& e){error=e.what();return -1;}}
API int fp4_c_decode(const uint8_t* data,size_t size,uint8_t** out,size_t* n){try{return result(patterns::decode(data,size),out,n);}catch(const std::exception& e){error=e.what();return -1;}}
API const char* fp4_c_error(){return error.c_str();}
API void fp4_c_free(void* p){std::free(p);}
