#include "../native/grouped_units/common.hpp"
#include <iostream>
// Numerical pipelines have separate whole-plugin Python/new-process tests.
namespace grouped {
Stream numeric_encode(unsigned,unsigned,const Words&){throw std::runtime_error("not part of integer sanitizer checks");}
Words numeric_decode(unsigned,unsigned,const Stream&){throw std::runtime_error("not part of integer sanitizer checks");}
Stream categorical_encode(unsigned,unsigned,const Words&){throw std::runtime_error("not part of integer sanitizer checks");}
Words categorical_decode(unsigned,const Stream&){throw std::runtime_error("not part of integer sanitizer checks");}
}
int main(){using namespace grouped;size_t cases=0;
 for(unsigned bits:{8u,16u,32u,64u})for(unsigned n:{1u,2u,7u,32u,257u})for(unsigned kind=0;kind<4;++kind){Words words(n);
  for(unsigned i=0;i<n;++i)words[i]=kind==0?0:kind==1?mask(bits):kind==2?(i%2?mask(bits):0):((uint64_t(i)*0xfedcba9876543211ull)^(uint64_t(i)<<31))&mask(bits);
  for(unsigned method:{0u,1u,2u,3u,10u,11u,12u,15u,16u,17u,18u,19u,20u,21u,30u,31u,33u,38u,64u,66u,67u}){
   if(bits==64&&method!=1&&method!=10&&method!=11&&method!=16&&method!=30)continue;
   if((method==31||method==33)&&bits>16)continue;if(method==38&&bits!=8)continue;
   auto stream=encode_method(method,bits,1,words);auto recovered=decode_method(method,bits,1,stream);
   if(words!=recovered)throw std::runtime_error("Integer boundary mismatch");++cases;
  }
 }
 std::cout<<cases<<" integer/sparse-entropy/prefix/CDC/row predictor sanitizer round trips passed\n";
}
