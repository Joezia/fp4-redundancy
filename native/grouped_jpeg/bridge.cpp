// MODIFIED 2026-11-10: reuse CharLS NEAR=0 and the existing B sample layout;
// only provide a bounded growing destination, not an estimated-size upper bound.
#include "jpeg_buffer.hpp"
namespace jpeg_buffer {
Bytes encode(unsigned method,unsigned fmt,unsigned cols,const uint8_t* p,size_t n){if(method!=14||fmt!=2&&fmt!=5||!cols)throw std::runtime_error("Grouped JPEG arguments");
 auto s=unpack(fmt,p,n);auto stream=jpeg_encode(s,width(fmt),cols);Bytes model;put<uint32_t>(model,1);put<uint32_t>(model,1u<<width(fmt));put<uint32_t>(model,cols);blob(model,{});blob(model,stream.model);put<uint64_t>(model,stream.data.size()*8);
 Bytes b;put<uint32_t>(b,MAGIC);put<uint64_t>(b,n);put<uint32_t>(b,14);put<uint32_t>(b,fmt);put<uint32_t>(b,0);put<uint32_t>(b,1);
 put<uint64_t>(b,s.size());put<uint64_t>(b,model.size());put<uint64_t>(b,stream.data.size());b.insert(b.end(),model.begin(),model.end());b.insert(b.end(),stream.data.begin(),stream.data.end());return b;}
Bytes decode(const uint8_t* p,size_t n){Reader r{p,n};if(r.get<uint32_t>()!=MAGIC)throw std::runtime_error("JPEG bridge magic");auto size=r.get<uint64_t>();if(r.get<uint32_t>()!=14)throw std::runtime_error("JPEG method");auto fmt=r.get<uint32_t>();if(fmt!=2&&fmt!=5||r.get<uint32_t>()||r.get<uint32_t>()!=1)throw std::runtime_error("JPEG format");
 auto count=r.get<uint64_t>(),nm=r.get<uint64_t>(),nd=r.get<uint64_t>();Stream s{count,r.take(nm),r.take(nd)};r.end();
 Reader m{s.model.data(),s.model.size()};if(m.get<uint32_t>()!=1||m.get<uint32_t>()!=(1u<<width(fmt)))throw std::runtime_error("JPEG model version");auto cols=m.get<uint32_t>();if(!cols||cols>count||!blob(m).empty()||!blob(m).empty()||m.get<uint64_t>()!=s.data.size()*8)throw std::runtime_error("JPEG model layout");m.end();
 return pack(fmt,jpeg_decode(s,width(fmt),cols),size);}
}
static thread_local std::string error;
static int result(const predictive::Bytes& b,uint8_t** out,size_t* n){*out=static_cast<uint8_t*>(std::malloc(b.size()));if(!*out)throw std::bad_alloc();std::memcpy(*out,b.data(),b.size());*n=b.size();return 0;}
#define API extern "C" __attribute__((visibility("default")))
API int fp4_jpeg_encode(unsigned m,unsigned f,unsigned c,const uint8_t* p,size_t n,uint8_t** out,size_t* size){try{return result(jpeg_buffer::encode(m,f,c,p,n),out,size);}catch(const std::exception& e){error=e.what();return -1;}}
API int fp4_jpeg_decode(const uint8_t* p,size_t n,uint8_t** out,size_t* size){try{return result(jpeg_buffer::decode(p,n),out,size);}catch(const std::exception& e){error=e.what();return -1;}}
API void fp4_jpeg_free(void* p){std::free(p);}
API const char* fp4_jpeg_error(){return error.c_str();}
