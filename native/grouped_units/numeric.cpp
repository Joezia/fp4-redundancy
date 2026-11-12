#include "grouped_math.hpp"
namespace grouped {
Stream numeric_encode(unsigned method,unsigned b,const Words& input){
 grouped_numeric::Words source(input.begin(),input.end());
 if(method>=41&&method<=45)return grouped_numeric::quantization_encode(source,b,method);
 using namespace grouped_math;
 Approximation fitted;if(method<=PCA)fitted=fit_transform(source,b,method);else if(method<=KSVD)fitted=fit_sparse(source,b,method);else fitted=fit_factors(source,b,method);
 auto control=parameters(method,fitted);auto coefficients=encode_i32(fitted.data);auto state=read_parameters(method,control,source.size());state.data=decode_i32(coefficients);
 auto prediction=reconstruct(method,state,source.size());Projection projection(b,state.exponent);grouped_numeric::Words errors(source.size());
 for(size_t i=0;i<source.size();++i)errors[i]=source[i]^projection.word(prediction[i]);
 return join(source.size(),b==32?0:1u<<b,control,{coefficients,grouped_numeric::correction_encode(errors,b)});
}
Words numeric_decode(unsigned method,unsigned b,const Stream& stream){
 if(method>=41&&method<=45){auto s=grouped_numeric::quantization_decode(stream,b,method);return Words(s.begin(),s.end());}
 using namespace grouped_math;auto parts=split(stream,b==32?0:1u<<b);if(parts.streams.size()!=2)throw std::runtime_error("Numerical stream count");
 auto state=read_parameters(method,parts.control,stream.count);
 if(parts.streams[0].count!=state.data.size()*2||parts.streams[1].count!=stream.count)throw std::runtime_error("Numerical coefficient/correction count");
 state.data=decode_i32(parts.streams[0]);auto prediction=reconstruct(method,state,stream.count);auto errors=grouped_numeric::correction_decode(parts.streams[1],b);Projection projection(b,state.exponent);Words out(stream.count);
 for(size_t i=0;i<out.size();++i)out[i]=errors[i]^projection.word(prediction[i]);return out;
}
}
