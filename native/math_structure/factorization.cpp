#include "common.hpp"
namespace mathematical {
static Mat unfold(const Mat& tile,unsigned mode){Mat out(4,16);
    for(unsigned i=0;i<4;++i)for(unsigned j=0;j<4;++j)for(unsigned k=0;k<4;++k){unsigned index=i*16+j*4+k;
        unsigned row=mode==0?i:mode==1?j:k,col=mode==0?j*4+k:mode==1?i*4+k:i*4+j;out(row,col)=tile(index/8,index%8);}return out;
}
static Mat left_vectors(const Mat& x,unsigned rank){Eigen::JacobiSVD<Mat> s(x,Eigen::ComputeThinU|Eigen::ComputeThinV);return s.matrixU().leftCols(rank);}
static std::array<Mat,3> cp_als(const Mat& tile){std::array<Mat,3> factors;
    for(unsigned m=0;m<3;++m)factors[m]=left_vectors(unfold(tile,m),2);
    for(unsigned sweep=0;sweep<20;++sweep)for(unsigned mode=0;mode<3;++mode){Mat design(16,2);unsigned other1=mode==0?1:0,other2=mode==2?1:2;
        for(unsigned i=0;i<4;++i)for(unsigned j=0;j<4;++j)for(unsigned r=0;r<2;++r)design(i*4+j,r)=factors[other1](i,r)*factors[other2](j,r);
        Eigen::JacobiSVD<Mat> solver(design,Eigen::ComputeThinU|Eigen::ComputeThinV);solver.setThreshold(1e-10);
        factors[mode]=solver.solve(unfold(tile,mode).transpose()).transpose();
        // Resolve scaling indeterminacy during ALS without changing the fit.
        for(unsigned r=0;r<2;++r){double norm=factors[mode].col(r).norm();if(norm>1e-12){factors[mode].col(r)/=norm;factors[(mode+1)%3].col(r)*=norm;}}
    }
    for(unsigned r=0;r<2;++r)for(unsigned m=1;m<3;++m){double norm=factors[m].col(r).norm();if(norm>1e-12){factors[m].col(r)/=norm;factors[0].col(r)*=norm;}}
    return factors;
}
static size_t factor_size(unsigned method){if(method==CP)return 24;if(method==KRONECKER)return 20;if(method==CIRCULANT)return 8;return 32;}
I32 factor_tile(unsigned method,const Mat& tile){I32 out;
    if(tile.squaredNorm()==0)return I32(factor_size(method),0);
    if(method==SVD){Eigen::JacobiSVD<Mat> s(tile,Eigen::ComputeThinU|Eigen::ComputeThinV);Mat a=s.matrixU().leftCols(2)*s.singularValues().head(2).cwiseSqrt().asDiagonal();
        Mat b=s.singularValues().head(2).cwiseSqrt().asDiagonal()*s.matrixV().leftCols(2).transpose();append(out,a);append(out,b);
    }else if(method==CP){auto f=cp_als(tile);for(auto& m:f)append(out,m);
    }else if(method==TUCKER){std::array<Mat,3> f;for(unsigned m=0;m<3;++m){f[m]=left_vectors(unfold(tile,m),2);append(out,f[m]);}
        for(unsigned a=0;a<2;++a)for(unsigned b=0;b<2;++b)for(unsigned c=0;c<2;++c){double value=0;
            for(unsigned i=0;i<4;++i)for(unsigned j=0;j<4;++j)for(unsigned k=0;k<4;++k){unsigned n=i*16+j*4+k;value+=tile(n/8,n%8)*f[0](i,a)*f[1](j,b)*f[2](k,c);}out.push_back(fixed_parameter(value));}
    }else if(method==TT){Mat first=unfold(tile,0);Eigen::JacobiSVD<Mat> s(first,Eigen::ComputeThinU|Eigen::ComputeThinV);append(out,s.matrixU().leftCols(2));
        Mat remaining=s.singularValues().head(2).asDiagonal()*s.matrixV().leftCols(2).transpose();Mat second(8,4);
        for(unsigned a=0;a<2;++a)for(unsigned j=0;j<4;++j)for(unsigned k=0;k<4;++k)second(a*4+j,k)=remaining(a,j*4+k);
        Eigen::JacobiSVD<Mat> next(second,Eigen::ComputeThinU|Eigen::ComputeThinV);append(out,next.matrixU().leftCols(2));
        append(out,next.singularValues().head(2).asDiagonal()*next.matrixV().leftCols(2).transpose());
    }else if(method==KRONECKER){Mat rearranged(4,16);
        for(unsigned a=0;a<2;++a)for(unsigned b=0;b<2;++b)for(unsigned i=0;i<4;++i)for(unsigned j=0;j<4;++j)rearranged(a*2+b,i*4+j)=tile(a*4+i,b*4+j);
        Eigen::JacobiSVD<Mat> s(rearranged,Eigen::ComputeThinU|Eigen::ComputeThinV);double scale=std::sqrt(s.singularValues()[0]);
        for(unsigned i=0;i<4;++i)out.push_back(fixed_parameter(s.matrixU()(i,0)*scale));for(unsigned j=0;j<16;++j)out.push_back(fixed_parameter(s.matrixV()(j,0)*scale));
    }else if(method==CIRCULANT){for(unsigned k=0;k<8;++k){double value=0;for(unsigned i=0;i<8;++i)value+=tile(i,(i+k)%8);out.push_back(fixed_parameter(value/8));}}
    else throw std::runtime_error("Unknown D factorization");
    if(out.size()!=factor_size(method))throw std::runtime_error("D factor serialization length differs");return out;
}
std::vector<int64_t> reconstruct_factor_tile(unsigned method,const I32& d){if(d.size()!=factor_size(method))throw std::runtime_error("D factor core dimensions differ");
    for(auto v:d)if(std::abs(int64_t(v))>(1<<28))throw std::runtime_error("D factor outside Q20 bound");std::vector<int64_t> out(64,0);
    if(method==SVD){for(unsigned i=0;i<8;++i)for(unsigned j=0;j<8;++j){__int128 sum=0;for(unsigned r=0;r<2;++r)sum+=__int128(d[i*2+r])*d[16+r*8+j];out[i*8+j]=round128(sum,20);}}
    else if(method==CP){for(unsigned i=0;i<4;++i)for(unsigned j=0;j<4;++j)for(unsigned k=0;k<4;++k){__int128 sum=0;
        for(unsigned r=0;r<2;++r)sum+=__int128(d[i*2+r])*d[8+j*2+r]*d[16+k*2+r];out[i*16+j*4+k]=round128(sum,40);}}
    else if(method==TUCKER){for(unsigned i=0;i<4;++i)for(unsigned j=0;j<4;++j)for(unsigned k=0;k<4;++k){__int128 sum=0;
        for(unsigned a=0;a<2;++a)for(unsigned b=0;b<2;++b)for(unsigned c=0;c<2;++c)sum+=__int128(d[i*2+a])*d[8+j*2+b]*d[16+k*2+c]*d[24+a*4+b*2+c];out[i*16+j*4+k]=round128(sum,60);}}
    else if(method==TT){for(unsigned i=0;i<4;++i)for(unsigned j=0;j<4;++j)for(unsigned k=0;k<4;++k){__int128 sum=0;
        for(unsigned a=0;a<2;++a)for(unsigned b=0;b<2;++b)sum+=__int128(d[i*2+a])*d[8+(a*4+j)*2+b]*d[24+b*4+k];out[i*16+j*4+k]=round128(sum,40);}}
    else if(method==KRONECKER){for(unsigned a=0;a<2;++a)for(unsigned b=0;b<2;++b)for(unsigned i=0;i<4;++i)for(unsigned j=0;j<4;++j)out[(a*4+i)*8+b*4+j]=round128(__int128(d[a*2+b])*d[4+i*4+j],20);}
    else if(method==CIRCULANT){for(unsigned i=0;i<8;++i)for(unsigned j=0;j<8;++j)out[i*8+j]=d[(j+8-i)%8];}
    else throw std::runtime_error("Unknown factor decoder");return out;
}
Approximation fit_factors(const Words& s,unsigned fmt,unsigned method){Approximation a;a.exponent=scale_exponent(s,fmt);a.threads=workers();size_t count=(s.size()+63)/64,step=factor_size(method);a.data.resize(count*step);std::vector<std::exception_ptr> errors(count);
    #pragma omp parallel for num_threads(a.threads) if(a.threads>1 && count>16)
    for(size_t i=0;i<count;++i){try{auto d=factor_tile(method,signal(s,fmt,a.exponent,i*64,8,8));std::copy(d.begin(),d.end(),a.data.begin()+i*step);}catch(...){errors[i]=std::current_exception();}}
    for(auto& error:errors)if(error)std::rethrow_exception(error);return a;
}
std::vector<int64_t> reconstruct_factors(unsigned method,const Approximation& a,size_t n){size_t count=(n+63)/64,step=factor_size(method);
    if(!a.parameters.empty()||a.data.size()!=count*step)throw std::runtime_error("D tensor tile/factor layout differs");std::vector<int64_t> out;out.reserve(count*64);
    for(size_t i=0;i<count;++i){auto tile=reconstruct_factor_tile(method,I32(a.data.begin()+i*step,a.data.begin()+(i+1)*step));out.insert(out.end(),tile.begin(),tile.end());}out.resize(n);return out;
}
}
