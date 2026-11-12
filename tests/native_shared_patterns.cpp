#include "../native/shared_patterns/reference.hpp"
#include "../native/shared_patterns/quantization.hpp"
#include "../native/shared_patterns/grammar.hpp"

#include <iostream>
int main(){using namespace patterns;
 Words a(256,1);a.insert(a.end(),256,2);a.insert(a.end(),256,1);
 // Force every chunk fingerprint to collide; unequal chunks must remain literal.
 auto encoded=dedup_encode(a,8,false,0);
 if(dedup_decode(encoded,8,false)!=a)throw std::runtime_error("Hash collision lost data");
 Reader r{encoded.data.data(),encoded.data.size()};
 if(var(r)!=256)throw std::runtime_error("Missing first literal");r.take(256);
 if(var(r)!=256)throw std::runtime_error("False duplicate on hash collision");r.take(256);
 if(var(r)!=0||var(r)!=0)throw std::runtime_error("Exact duplicate was not referenced");r.end();
 std::vector<Book> b(2,Book(2));b[0][1][0]=100;b[1][1][0]=-99;Vec x{};x[0]=1;
 auto code=aq_beam(x,b);if(code!=Code{1,1})throw std::runtime_error("AQ collapsed to greedy RVQ");
 size_t cases=0;
 for(unsigned fmt=0;fmt<7;++fmt)for(size_t size:{size_t(1),size_t(2),size_t(3),size_t(7),size_t(16),size_t(31),size_t(64),size_t(257)}){
  if(fmt==5 && size%2)continue;
  Bytes raw(size);for(size_t i=0;i<size;++i)raw[i]=uint8_t(i*127+i/7);
  auto s=unpack(fmt,raw.data(),raw.size());unsigned bits=width(fmt);
  for(unsigned method=RLE;method<=AQ;++method){Stream stream;Words restored;
   switch(method){
   case RLE:stream=rle_encode(s);restored=rle_decode(stream);break;
   case LZ77:case LZSS:stream=lz_encode(s,method==LZSS);restored=lz_decode(stream,method==LZSS);break;
   case LZ78:stream=lz78_encode(s);restored=lz78_decode(stream);break;
   case LZW:stream=lzw_encode(s,bits);restored=lzw_decode(stream,bits);break;
   case REPAIR:stream=repair_encode(s,1u<<bits);restored=grammar_decode(stream,1u<<bits);break;
   case SEQUITUR:stream=sequitur_encode(s,1u<<bits);restored=grammar_decode(stream,1u<<bits);break;
   case DEDUP:case CDC:stream=dedup_encode(s,bits,method==CDC);restored=dedup_decode(stream,bits,method==CDC);break;
   case VCDIFF:case RLZ:stream=reference_encode(s,bits,method==VCDIFF);restored=reference_decode(stream,bits,method==VCDIFF);break;
   default:stream=quantization_encode(s,fmt,method);restored=quantization_decode(stream,fmt,method);break;
   }
   if(pack(fmt,restored,raw.size())!=raw)throw std::runtime_error("Boundary roundtrip differs");++cases;
  }
 }
 std::cout<<cases<<" boundary cases plus forced hash collision and non-greedy AQ passed\n";
}
