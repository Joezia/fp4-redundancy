#pragma once
#include "../shared_patterns/quantization.hpp"
namespace mathematical {
using namespace patterns;
using I32 = std::vector<int32_t>;
using Mat = Eigen::MatrixXd;
enum Method {BWT_MTF=60,DCT=61,DST=62,MDCT=63,CDF53=64,PCA=65,HAAR=66,BITPLANE=67,
             SPARSE=68,KSVD=69,SVD=70,CP=71,TUCKER=72,TT=73,KRONECKER=74,CIRCULANT=75};
constexpr size_t COMPONENT_LIMIT=1u<<20;
constexpr double PI=3.141592653589793238462643383279502884;
inline bool exact(unsigned m){return m==BWT_MTF||m==CDF53||m==HAAR||m==BITPLANE;}
inline int workers(){const char* p=std::getenv("FP4_D_THREADS");if(!p)return 1;char* end=nullptr;long n=std::strtol(p,&end,10);
    if(*end||n<1||n>4096)throw std::runtime_error("FP4_D_THREADS must be in 1..4096");return int(n);}
inline int32_t checked32(int64_t x){if(x<INT32_MIN||x>INT32_MAX)throw std::runtime_error("Integer transform exceeds signed 32-bit coefficient range");return int32_t(x);}
inline int64_t round128(__int128 x,unsigned bits){__int128 half=__int128(1)<<(bits-1),den=__int128(1)<<bits;
    __int128 result=x>=0?(x+half)/den:-((-x+half)/den);
    if(result<INT64_MIN||result>INT64_MAX)throw std::runtime_error("Fixed-point reconstruction overflow");return int64_t(result);}
inline Stream encode_i32(const I32& values){Words words;words.reserve(values.size()*2);
    for(auto v:values){uint32_t u=(uint32_t(v)<<1)^(v<0?UINT32_MAX:0);words.push_back(u&65535);words.push_back(u>>16);}return entropy_encode(words);}
inline I32 decode_i32(const Stream& stream){if(stream.count%2)throw std::runtime_error("Invalid signed coefficient count");auto words=entropy_decode(stream);I32 out;out.reserve(words.size()/2);
    for(size_t i=0;i<words.size();i+=2){uint32_t u=words[i]|(uint32_t(words[i+1])<<16);out.push_back(u&1?-1-int32_t(u>>1):int32_t(u>>1));}return out;}
inline void append(I32& out,const Mat& matrix){for(Eigen::Index i=0;i<matrix.rows();++i)for(Eigen::Index j=0;j<matrix.cols();++j)out.push_back(fixed_parameter(matrix(i,j)));}
inline Mat matrix(const I32& q,size_t start,unsigned rows,unsigned cols){if(start+size_t(rows)*cols>q.size())throw std::runtime_error("Truncated mathematical parameters");Mat out(rows,cols);
    for(unsigned i=0;i<rows;++i)for(unsigned j=0;j<cols;++j)out(i,j)=double(q[start+i*cols+j])/UNIT;return out;}
inline Mat signal(const Words& s,unsigned fmt,int exponent,size_t start,unsigned rows,unsigned cols){Mat out=Mat::Zero(rows,cols);
    for(unsigned i=0;i<rows;++i)for(unsigned j=0;j<cols;++j){auto at=start+i*cols+j;if(at<s.size())out(i,j)=double(fixed_value(s[at],fmt,exponent))/UNIT;}return out;}
struct Approximation {int exponent=0,threads=1;I32 parameters,data;};
struct Geometry {std::vector<unsigned> shape,ranks;};
inline Geometry geometry(unsigned m){
    if(m==CP)return {{4,4,4},{2}};
    if(m==TUCKER)return {{4,4,4},{2,2,2}};
    if(m==TT)return {{4,4,4},{1,2,2,1}};
    if(m==SVD)return {{8,8},{2}};
    if(m==KRONECKER)return {{8,8,2,2,4,4},{1}};
    if(m==CIRCULANT)return {{8,8},{8}};
    if(m==SPARSE||m==KSVD)return {{8,16},{3}};
    if(m==MDCT)return {{16,8},{8}};
    return {{8},{8}};
}
inline Bytes parameters(unsigned method,const Approximation& a){Bytes b;put<uint32_t>(b,1);put<int16_t>(b,a.exponent);put<uint16_t>(b,a.threads);
    auto g=geometry(method);put<uint32_t>(b,g.shape.size());for(auto v:g.shape)put<uint32_t>(b,v);put<uint32_t>(b,g.ranks.size());for(auto v:g.ranks)put<uint32_t>(b,v);
    put<uint32_t>(b,a.parameters.size());for(auto v:a.parameters)put<int32_t>(b,v);put<uint64_t>(b,a.data.size());return b;}
inline Approximation read_parameters(unsigned method,const Bytes& bytes,size_t n){Reader r{bytes.data(),bytes.size()};Approximation a;
    if(r.get<uint32_t>()!=1)throw std::runtime_error("Unknown D numerical parameter version");a.exponent=r.get<int16_t>();a.threads=r.get<uint16_t>();
    if(a.exponent<-140||a.exponent>140||a.threads<1||a.threads>4096)throw std::runtime_error("Invalid D normalization/thread parameter");
    auto g=geometry(method);if(r.get<uint32_t>()!=g.shape.size())throw std::runtime_error("D shape rank differs");for(auto v:g.shape)if(r.get<uint32_t>()!=v)throw std::runtime_error("D shape differs");
    if(r.get<uint32_t>()!=g.ranks.size())throw std::runtime_error("D decomposition rank count differs");for(auto v:g.ranks)if(r.get<uint32_t>()!=v)throw std::runtime_error("D decomposition ranks differ");
    auto size=r.get<uint32_t>();if(size>512)throw std::runtime_error("Excess D shared parameters");
    for(unsigned i=0;i<size;++i){auto p=r.get<int32_t>();if(std::abs(int64_t(p))>(1<<28))throw std::runtime_error("D shared parameter outside Q20 bound");a.parameters.push_back(p);}
    auto count=r.get<uint64_t>();if(!count||count>n*3+64)throw std::runtime_error("D coefficient count exceeds bound");a.data.resize(count);r.end();return a;
}
struct Streams {Bytes control;std::vector<Stream> streams;};
inline Stream join(size_t count,unsigned alphabet,const Bytes& control,const std::vector<Stream>& streams){Stream result{count,{}, {}};
    put<uint32_t>(result.model,1);put<uint32_t>(result.model,alphabet);blob(result.model,control);put<uint32_t>(result.model,streams.size());
    for(const auto& s:streams){put<uint64_t>(result.model,s.count);blob(result.model,s.model);put<uint64_t>(result.model,s.data.size());result.data.insert(result.data.end(),s.data.begin(),s.data.end());}
    put<uint64_t>(result.model,result.data.size()*8);return result;
}
inline Streams split(const Stream& s,unsigned alphabet){Reader r{s.model.data(),s.model.size()},p{s.data.data(),s.data.size()};
    if(r.get<uint32_t>()!=1||r.get<uint32_t>()!=alphabet)throw std::runtime_error("Invalid D stream version/alphabet");Streams out;out.control=blob(r);auto ns=r.get<uint32_t>();
    if(!ns||ns>16)throw std::runtime_error("Invalid D stream count");for(unsigned i=0;i<ns;++i){Stream stream;stream.count=r.get<uint64_t>();if(!stream.count||stream.count>s.count*6+128)throw std::runtime_error("D inner stream count outside bound");
        stream.model=blob(r);stream.data=p.take(r.get<uint64_t>());out.streams.push_back(std::move(stream));}
    if(r.get<uint64_t>()!=s.data.size()*8)throw std::runtime_error("D payload byte count differs");r.end();p.end();return out;}
std::pair<Words,uint32_t> bwt(const Words&);
Words inverse_bwt(const Words&,uint32_t);
Words mtf(const Words&,unsigned,bool);
I32 lifting(const I32&,bool,bool);
Stream exact_encode(const Words&,unsigned,unsigned);
Words exact_decode(const Stream&,unsigned,unsigned);
Mat dct_basis();Mat dst_basis();Mat mdct_cosines();Mat mdct_window();
Approximation fit_transform(const Words&,unsigned,unsigned);
std::vector<int64_t> reconstruct_transform(unsigned,const Approximation&,size_t);
Mat fixed_dictionary();std::pair<std::vector<unsigned>,Eigen::VectorXd> omp(const Mat&,const Eigen::VectorXd&);
Mat train_ksvd(const Mat&);
Approximation fit_sparse(const Words&,unsigned,unsigned);
std::vector<int64_t> reconstruct_sparse(unsigned,const Approximation&,size_t);
I32 factor_tile(unsigned,const Mat&);
std::vector<int64_t> reconstruct_factor_tile(unsigned,const I32&);
Approximation fit_factors(const Words&,unsigned,unsigned);
std::vector<int64_t> reconstruct_factors(unsigned,const Approximation&,size_t);
inline std::vector<int64_t> reconstruct(unsigned method,const Approximation& a,size_t n){
    if(method<=PCA)return reconstruct_transform(method,a,n);
    if(method==SPARSE||method==KSVD)return reconstruct_sparse(method,a,n);
    return reconstruct_factors(method,a,n);
}
}
