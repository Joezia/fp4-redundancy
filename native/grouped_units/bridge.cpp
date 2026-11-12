#include "common.hpp"
namespace grouped {
Bytes reshape(unsigned bits,unsigned high_first,unsigned row_bytes,const uint8_t* p,size_t size,bool inverse){
 if(!(bits==8||bits==16||bits==32||bits==64)||high_first>1||!row_bytes||!size||size>LIMIT)throw std::runtime_error("Grouped packing arguments");
 size_t width=bits/8,grouped_row=(row_bytes+width-1)/width*width,source_row=inverse?grouped_row:row_bytes;
 if(size%source_row)throw std::runtime_error("Grouped row byte count");size_t rows=size/source_row;Bytes out(rows*(inverse?row_bytes:grouped_row),0);
 for(size_t row=0;row<rows;++row)for(size_t col=0;col<row_bytes;++col){auto value=p[row*source_row+col];if(high_first)value=uint8_t((value>>4)|(value<<4));out[row*(inverse?row_bytes:grouped_row)+col]=value;}
 if(inverse)for(size_t row=0;row<rows;++row)for(size_t col=row_bytes;col<grouped_row;++col)if(p[row*grouped_row+col])throw std::runtime_error("Grouped nonzero row-tail padding");return out;}
Bytes encode(unsigned method,unsigned bits,unsigned cols,const uint8_t* raw,size_t size){auto s=unpack(raw,size,bits);if(!cols||s.size()%cols)throw std::runtime_error("Grouped rectangular row layout");return frame(method,bits,cols,encode_method(method,bits,cols,s));}
Bytes decode(const uint8_t* data,size_t size){Reader r{data,size};if(r.get<uint32_t>()!=MAGIC)throw std::runtime_error("Grouped frame magic");unsigned method=r.get<uint32_t>(),bits=r.get<uint32_t>(),cols=r.get<uint32_t>();auto count=r.get<uint64_t>(),nm=r.get<uint64_t>(),nd=r.get<uint64_t>();
 if(!(bits==8||bits==16||bits==32||bits==64)||!count||count>LIMIT/(bits/8)||!cols||count%cols)throw std::runtime_error("Grouped frame geometry");auto model=r.take(nm),payload=r.take(nd);r.end();return pack(decode_method(method,bits,cols,{size_t(count),model,payload}),bits);}
}
static thread_local std::string error;
static int result(const predictive::Bytes& b,uint8_t** out,size_t* n){*out=static_cast<uint8_t*>(std::malloc(b.size()));if(!*out)throw std::bad_alloc();std::memcpy(*out,b.data(),b.size());*n=b.size();return 0;}
#define API extern "C" __attribute__((visibility("default")))
API int fp4_units_encode(unsigned method,unsigned bits,unsigned cols,const uint8_t* raw,size_t size,uint8_t** out,size_t* n){try{return result(grouped::encode(method,bits,cols,raw,size),out,n);}catch(const std::exception& e){error=e.what();return -1;}}
API int fp4_units_vq_encode(unsigned bits,unsigned flags,unsigned cols,const uint8_t* raw,size_t size,uint8_t** out,size_t* n){try{if(bits!=16&&bits!=32)throw std::runtime_error("Sparse categorical width");auto s=grouped::unpack(raw,size,bits);if(!cols||s.size()%cols)throw std::runtime_error("Categorical geometry");return result(grouped::frame(80,bits,cols,grouped::categorical_encode(bits,flags,s)),out,n);}catch(const std::exception& e){error=e.what();return -1;}}
API int fp4_units_decode(const uint8_t* raw,size_t size,uint8_t** out,size_t* n){try{return result(grouped::decode(raw,size),out,n);}catch(const std::exception& e){error=e.what();return -1;}}
API int fp4_units_pack(unsigned bits,unsigned high_first,unsigned cols,const uint8_t* raw,size_t size,uint8_t** out,size_t* n){try{return result(grouped::reshape(bits,high_first,cols,raw,size,false),out,n);}catch(const std::exception& e){error=e.what();return -1;}}
API int fp4_units_unpack(unsigned bits,unsigned high_first,unsigned cols,const uint8_t* raw,size_t size,uint8_t** out,size_t* n){try{return result(grouped::reshape(bits,high_first,cols,raw,size,true),out,n);}catch(const std::exception& e){error=e.what();return -1;}}
API const char* fp4_units_error(){return error.c_str();}
API void fp4_units_free(void* p){std::free(p);}
