#pragma once
#include "common.hpp"
#include <Eigen/Dense>
#include <omp.h>
namespace patterns {
using Vec=std::array<double,8>;
using Code=std::array<unsigned,2>;
using Book=std::vector<Vec>;
constexpr int64_t UNIT=1<<20;
inline int threads(){const char* p=std::getenv("FP4_C_THREADS");if(!p)return 1;char* end=nullptr;auto n=std::strtol(p,&end,10);
    if(*end||n<1||n>4096)throw std::runtime_error("FP4_C_THREADS must be in 1..4096");return int(n);}
struct Dyadic {int significand,power;};
inline Dyadic numeric(uint16_t value,unsigned fmt){
    if(fmt<2){static constexpr int e2m1[8]={0,1,2,3,4,6,8,12};return {(value&8)?-e2m1[value&7]:e2m1[value&7],-1};}
    if(fmt==4)return value==255?Dyadic{0,0}:Dyadic{1,int(value)-127};
    if(fmt==6)return {int(value),0}; // User-selected F32 raw byte representation.
    unsigned mbits=fmt==5?7:fmt==2?3:2,ebits=fmt==5?8:fmt==2?4:5,bias=(1u<<(ebits-1))-1;
    unsigned exponent=(value>>mbits)&((1u<<ebits)-1),mantissa=value&((1u<<mbits)-1);
    if((fmt==2&&exponent==15&&mantissa==7)||(fmt!=2&&exponent==((1u<<ebits)-1)))return {0,0};
    int significand=exponent?int((1u<<mbits)|mantissa):int(mantissa);if(value&(1u<<(ebits+mbits)))significand=-significand;
    return {significand,int(exponent?exponent:1)-int(bias)-int(mbits)};
}
inline int scale_exponent(const Words& s,unsigned fmt){int best=INT_MIN;
    for(auto v:s){auto d=numeric(v,fmt);unsigned x=unsigned(std::abs(d.significand));if(x){unsigned log=0;while(x>>1){x>>=1;++log;}best=std::max(best,d.power+int(log)+1);}}
    return best==INT_MIN?0:best;
}
inline int64_t fixed_value(uint16_t v,unsigned fmt,int exponent){auto d=numeric(v,fmt);if(!d.significand)return 0;
    unsigned a=unsigned(std::abs(d.significand));int shift=d.power+20-exponent;uint64_t result;
    constexpr uint64_t cap=uint64_t(1)<<48;
    if(shift>=0)result=shift>=48||a>(cap>>shift)?cap:uint64_t(a)<<shift;
    else {unsigned n=unsigned(-shift);result=n>=64?0:(uint64_t(a)+(uint64_t(1)<<(n-1)))>>n;}
    return d.significand<0?-int64_t(result):int64_t(result);
}
inline int32_t fixed_parameter(double v){if(!std::isfinite(v))throw std::runtime_error("Nonfinite trained quantizer parameter");
    return int32_t(std::clamp<double>(std::round(v*UNIT),-(1<<28),(1<<28)));}
inline int64_t rounded_shift(int64_t v){return v>=0?(v+UNIT/2)/UNIT:-((-v+UNIT/2)/UNIT);}
inline double squared(const Vec& a,const Vec& b,unsigned lo=0,unsigned hi=8){double e=0;for(unsigned j=lo;j<hi;++j){auto d=a[j]-b[j];e+=d*d;}return e;}
inline unsigned nearest(const Vec& x,const Book& c,unsigned lo=0,unsigned hi=8){unsigned best=0;double cost=INFINITY;
    for(unsigned k=0;k<c.size();++k){auto e=squared(x,c[k],lo,hi);if(e<cost){cost=e;best=k;}}return best;}
inline Book kmeans(const std::vector<Vec>& x,unsigned k,unsigned lo,unsigned hi,int workers){Book c(k);c[0]=x[0];std::vector<double> distances(x.size(),INFINITY);
    for(unsigned seed=1;seed<k;++seed){size_t farthest=0;
        for(size_t i=0;i<x.size();++i){distances[i]=std::min(distances[i],squared(x[i],c[seed-1],lo,hi));if(distances[i]>distances[farthest])farthest=i;}c[seed]=x[farthest];}
    std::vector<unsigned> assignments(x.size());
    for(unsigned iteration=0;iteration<8;++iteration){
        #pragma omp parallel for num_threads(workers) if(workers>1 && x.size()>64)
        for(size_t i=0;i<x.size();++i)assignments[i]=nearest(x[i],c,lo,hi);
        std::vector<Vec> sums(k);std::vector<unsigned> counts(k,0);
        // Fixed reduction order, independent of worker scheduling.
        for(size_t i=0;i<x.size();++i){++counts[assignments[i]];for(unsigned j=lo;j<hi;++j)sums[assignments[i]][j]+=x[i][j];}
        for(unsigned a=0;a<k;++a)if(counts[a])for(unsigned j=lo;j<hi;++j)c[a][j]=sums[a][j]/counts[a];
    }return c;
}
inline Code aq_beam(const Vec& x,const std::vector<Book>& books){
    struct State {Code code;unsigned used;Vec sum;double error;};std::vector<State> beam{{{0,0},0,{},squared(x,{})}};
    for(unsigned stage=0;stage<2;++stage){std::vector<State> options;
        for(auto& previous:beam)for(unsigned m=0;m<2;++m)if(!(previous.used&(1u<<m)))for(unsigned k=0;k<books[m].size();++k){
            auto next=previous;next.used|=1u<<m;next.code[m]=k;for(unsigned j=0;j<8;++j)next.sum[j]+=books[m][k][j];next.error=squared(x,next.sum);options.push_back(next);}
        std::sort(options.begin(),options.end(),[](const auto& a,const auto& b){return a.error!=b.error?a.error<b.error:a.used!=b.used?a.used<b.used:a.code<b.code;});
        beam.clear();std::set<std::pair<unsigned,Code>> seen;
        for(auto& candidate:options)if(seen.emplace(candidate.used,candidate.code).second){beam.push_back(candidate);if(beam.size()==16)break;}
    }return beam.front().code;
}
struct Quantizer {
    unsigned method,k,nt;int exponent,workers;std::vector<Book> books;
    std::array<int32_t,64> rotation{};
    bool product()const{return method==PQ||method==OPQ;}
    unsigned stages()const{return method==VQ?1:2;}
    Vec rotate(const Vec& x)const{Vec out{};for(unsigned j=0;j<8;++j)for(unsigned i=0;i<8;++i)out[j]+=x[i]*double(rotation[i*8+j])/UNIT;return out;}
    Code assign(const Vec& input)const{
        auto x=method==OPQ?rotate(input):input;Code c{};
        if(method==AQ)return aq_beam(x,books);
        for(unsigned m=0;m<stages();++m){unsigned lo=product()?m*4:0,hi=product()?(m+1)*4:8;c[m]=nearest(x,books[m],lo,hi);
            if(method==RVQ)for(unsigned j=0;j<8;++j)x[j]-=books[m][c[m]][j];}
        return c;
    }
    std::array<int64_t,8> reconstruction(const Code& code)const{
        std::array<int64_t,8> z{},out{};
        for(unsigned m=0;m<stages();++m)for(unsigned j=product()?m*4:0;j<(product()?(m+1)*4:8);++j)
            z[j]+=int64_t(books[m][code[m]][j]*UNIT); // Stored Q20 values are exactly representable as double.
        if(method!=OPQ)return z;
        for(unsigned j=0;j<8;++j){int64_t sum=0;for(unsigned i=0;i<8;++i)sum+=int64_t(rotation[j*8+i])*z[i];out[j]=rounded_shift(sum);}return out;
    }
};
inline Quantizer train_quantizer(const Words& s,unsigned fmt,unsigned method){Quantizer q; q.method=method;q.exponent=scale_exponent(s,fmt);q.workers=threads();
    q.nt=std::min<size_t>((s.size()+7)/8,512);q.k=std::min(16u,q.nt);std::vector<Vec> x(q.nt);
    for(size_t i=0;i<q.nt;++i)for(unsigned j=0;j<8;++j)if(i*8+j<s.size())x[i][j]=double(fixed_value(s[i*8+j],fmt,q.exponent))/UNIT;
    if(method==VQ)q.books={kmeans(x,q.k,0,8,q.workers)};
    else if(method==PQ)q.books={kmeans(x,q.k,0,4,q.workers),kmeans(x,q.k,4,8,q.workers)};
    else if(method==OPQ){Eigen::MatrixXd X(x.size(),8);for(size_t i=0;i<x.size();++i)for(unsigned j=0;j<8;++j)X(i,j)=x[i][j];
        Eigen::MatrixXd R=Eigen::MatrixXd::Identity(8,8);
        for(unsigned iteration=0;iteration<4;++iteration){auto Z=(X*R).eval();std::vector<Vec> rotated(x.size());
            for(size_t i=0;i<x.size();++i)for(unsigned j=0;j<8;++j)rotated[i][j]=Z(i,j);
            q.books={kmeans(rotated,q.k,0,4,q.workers),kmeans(rotated,q.k,4,8,q.workers)};Eigen::MatrixXd Y(x.size(),8);
            for(size_t i=0;i<x.size();++i)for(unsigned m=0;m<2;++m){auto k=nearest(rotated[i],q.books[m],m*4,(m+1)*4);for(unsigned j=m*4;j<(m+1)*4;++j)Y(i,j)=q.books[m][k][j];}
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(X.transpose()*Y,Eigen::ComputeFullU|Eigen::ComputeFullV);R=svd.matrixU()*svd.matrixV().transpose();
        }
        for(unsigned i=0;i<64;++i)q.rotation[i]=fixed_parameter(R(i/8,i%8));std::vector<Vec> rotated(x.size());
        for(size_t i=0;i<x.size();++i)rotated[i]=q.rotate(x[i]);q.books={kmeans(rotated,q.k,0,4,q.workers),kmeans(rotated,q.k,4,8,q.workers)};
    }else{
        auto first=kmeans(x,q.k,0,8,q.workers);std::vector<Vec> residual=x;
        for(size_t i=0;i<x.size();++i){auto k=nearest(x[i],first);for(unsigned j=0;j<8;++j)residual[i][j]-=first[k][j];}
        q.books={first,kmeans(residual,q.k,0,8,q.workers)};
        if(method==AQ)for(unsigned iteration=0;iteration<4;++iteration){
            Eigen::MatrixXd A=Eigen::MatrixXd::Zero(x.size(),2*q.k),X(x.size(),8);
            for(size_t i=0;i<x.size();++i){auto code=aq_beam(x[i],q.books);A(i,code[0])=1;A(i,q.k+code[1])=1;for(unsigned j=0;j<8;++j)X(i,j)=x[i][j];}
            // Original AQ alternation: beam assignment, then joint codebook least squares.
            // Additive books have a translation gauge: the assignment matrix is
            // necessarily rank deficient. Use a minimum-norm pseudoinverse,
            // rather than unstable arbitrary coefficients that cancel in float.
            Eigen::JacobiSVD<Eigen::MatrixXd> least_squares(A,Eigen::ComputeThinU|Eigen::ComputeThinV);
            least_squares.setThreshold(1e-10);
            Eigen::MatrixXd C=least_squares.solve(X);
            for(unsigned m=0;m<2;++m)for(unsigned k=0;k<q.k;++k)for(unsigned j=0;j<8;++j)q.books[m][k][j]=C(m*q.k+k,j);
        }
    }
    for(unsigned m=0;m<q.stages();++m)for(auto& c:q.books[m])for(unsigned j=q.product()?m*4:0;j<(q.product()?(m+1)*4:8);++j)c[j]=double(fixed_parameter(c[j]))/UNIT;
    return q;
}
struct Projection {
    std::vector<std::pair<int64_t,uint16_t>> table;
    Projection(unsigned fmt,int exponent){table.reserve(1u<<width(fmt));for(unsigned i=0;i<(1u<<width(fmt));++i)table.emplace_back(fixed_value(i,fmt,exponent),i);
        std::sort(table.begin(),table.end());}
    uint16_t word(int64_t x)const{auto at=std::lower_bound(table.begin(),table.end(),std::make_pair(x,uint16_t(0)));uint16_t code=0;int64_t error=INT64_MAX;
        auto consider=[&](int64_t value){auto p=std::lower_bound(table.begin(),table.end(),std::make_pair(value,uint16_t(0)));auto e=std::abs(value-x);
            if(e<error || (e==error&&p->second<code)){error=e;code=p->second;}};
        if(at!=table.end())consider(at->first);if(at!=table.begin())consider(std::prev(at)->first);return code;}
};
inline Stream quantization_encode(const Words& s,unsigned fmt,unsigned method){auto q=train_quantizer(s,fmt,method);Projection projection(fmt,q.exponent);
    size_t nv=(s.size()+7)/8;Words indices(nv*q.stages()),corrections(s.size());
    #pragma omp parallel for num_threads(q.workers) if(q.workers>1 && nv>64)
    for(size_t i=0;i<nv;++i){Vec x{};for(unsigned j=0;j<8;++j)if(i*8+j<s.size())x[j]=double(fixed_value(s[i*8+j],fmt,q.exponent))/UNIT;
        auto code=q.assign(x);for(unsigned m=0;m<q.stages();++m)indices[i*q.stages()+m]=code[m];auto pred=q.reconstruction(code);
        for(unsigned j=0;j<8;++j)if(i*8+j<s.size())corrections[i*8+j]=s[i*8+j]^projection.word(pred[j]);
    }
    auto index=entropy_encode(indices),error=entropy_encode(corrections);Stream result{s.size(),{},index.data};result.data.insert(result.data.end(),error.data.begin(),error.data.end());
    put<uint32_t>(result.model,8);put<uint32_t>(result.model,q.k);put<uint32_t>(result.model,q.nt);put<int16_t>(result.model,q.exponent);put<uint16_t>(result.model,q.workers);
    if(method==OPQ)for(auto v:q.rotation)put<int32_t>(result.model,v);
    for(unsigned m=0;m<q.stages();++m)for(auto& center:q.books[m])for(unsigned j=q.product()?m*4:0;j<(q.product()?(m+1)*4:8);++j)put<int32_t>(result.model,int32_t(center[j]*UNIT));
    blob(result.model,index.model);blob(result.model,error.model);put<uint64_t>(result.model,index.data.size());return result;
}
inline Words quantization_decode(const Stream& s,unsigned fmt,unsigned method){Reader r{s.model.data(),s.model.size()};Quantizer q;q.method=method;
    if(r.get<uint32_t>()!=8)throw std::runtime_error("Invalid C3 vector dimension");q.k=r.get<uint32_t>();q.nt=r.get<uint32_t>();q.exponent=r.get<int16_t>();q.workers=r.get<uint16_t>();
    if(!q.k||q.k>16||!q.nt||q.nt>512||q.k>q.nt||q.exponent<-140||q.exponent>140||!q.workers||q.workers>4096)throw std::runtime_error("Invalid C3 model");
    if(method==OPQ)for(auto& v:q.rotation){v=r.get<int32_t>();if(std::abs(int64_t(v))>UNIT+2)throw std::runtime_error("Invalid OPQ matrix entry");}
    q.books.resize(q.stages(),Book(q.k));for(unsigned m=0;m<q.stages();++m)for(auto& center:q.books[m])for(unsigned j=q.product()?m*4:0;j<(q.product()?(m+1)*4:8);++j){
        auto p=r.get<int32_t>();if(std::abs(int64_t(p))>(1<<28))throw std::runtime_error("C3 codebook outside fixed-point bound");center[j]=double(p)/UNIT;}
    auto imodel=blob(r),emodel=blob(r);auto n=r.get<uint64_t>();r.end();if(n>s.data.size())throw std::runtime_error("Invalid C3 stream boundary");
    auto nv=(s.count+7)/8;auto indices=entropy_decode({nv*q.stages(),imodel,Bytes(s.data.begin(),s.data.begin()+n)});
    auto errors=entropy_decode({s.count,emodel,Bytes(s.data.begin()+n,s.data.end())});Projection projection(fmt,q.exponent);Words out(s.count);
    // No floating-point transform is used during reconstruction: all values here
    // are exactly represented Q20 integers, including the quantized OPQ inverse.
    for(size_t i=0;i<nv;++i){Code code{};for(unsigned m=0;m<q.stages();++m){code[m]=indices[i*q.stages()+m];if(code[m]>=q.k)throw std::runtime_error("C3 index outside codebook");}
        auto pred=q.reconstruction(code);for(unsigned j=0;j<8;++j)if(i*8+j<s.count)out[i*8+j]=errors[i*8+j]^projection.word(pred[j]);}
    return out;
}
}
