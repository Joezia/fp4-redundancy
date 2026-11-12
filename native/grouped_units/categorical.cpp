#include "grouped_vq.hpp"
namespace grouped {
Stream categorical_encode(unsigned bits,unsigned flags,const Words& s){grouped_numeric::Words input(s.begin(),s.end());return grouped_vq::encode(input,bits,grouped_vq::options(flags));}
Words categorical_decode(unsigned bits,const Stream& s){auto out=grouped_vq::decode(s,bits);return Words(out.begin(),out.end());}
}
