#include "common.hpp"
namespace mathematical {
Mat dct_basis(){Mat b(8,8);for(unsigned k=0;k<8;++k)for(unsigned i=0;i<8;++i)b(k,i)=(k?0.5:1/std::sqrt(8.0))*std::cos(PI*(i+.5)*k/8);return b;}
Mat dst_basis(){Mat b(8,8);for(unsigned k=0;k<8;++k)for(unsigned i=0;i<8;++i)b(k,i)=std::sqrt(2.0/9)*std::sin(PI*(i+1)*(k+1)/9);return b;}
Mat mdct_cosines(){Mat b(16,8);for(unsigned i=0;i<16;++i)for(unsigned k=0;k<8;++k)b(i,k)=std::cos(PI/8*(i+.5+4)*(k+.5));return b;}
Mat mdct_window(){Mat w(1,16);for(unsigned i=0;i<16;++i)w(0,i)=std::sin(PI/16*(i+.5));return w;}
Approximation fit_transform(const Words& s,unsigned fmt,unsigned method){Approximation a;a.exponent=scale_exponent(s,fmt);a.threads=workers();Mat basis,mean=Mat::Zero(1,8);
    if(method==MDCT){auto cosines=mdct_cosines(),window=mdct_window();append(a.parameters,window);append(a.parameters,cosines);
        size_t vectors=(s.size()+7)/8,frames=vectors+1;a.data.resize(frames*8);
        #pragma omp parallel for num_threads(a.threads) if(a.threads>1 && frames>64)
        for(size_t f=0;f<frames;++f)for(unsigned k=0;k<8;++k){double value=0;
            for(unsigned i=0;i<16;++i){int64_t at=int64_t(f*8+i)-8;if(at>=0&&size_t(at)<s.size())value+=double(fixed_value(s[at],fmt,a.exponent))/UNIT*window(0,i)*cosines(i,k);}
            a.data[f*8+k]=fixed_parameter(value);}
        return a;
    }
    if(method==DCT)basis=dct_basis();else if(method==DST)basis=dst_basis();else{
        size_t nt=std::min<size_t>((s.size()+7)/8,512);Mat x(nt,8);
        for(size_t i=0;i<nt;++i)x.row(i)=signal(s,fmt,a.exponent,i*8,1,8);
        mean=x.colwise().mean();Mat centered=x.rowwise()-mean.row(0);Mat covariance=centered.transpose()*centered/double(nt);
        Eigen::SelfAdjointEigenSolver<Mat> eigen(covariance);if(eigen.info()!=Eigen::Success)throw std::runtime_error("PCA eigensolver failed");
        basis.resize(8,8);for(unsigned i=0;i<8;++i)basis.row(i)=eigen.eigenvectors().col(7-i).transpose();append(a.parameters,mean);
    }
    append(a.parameters,basis);
    // Analysis uses the actual serialized basis/mean. Decoder uses the same
    // stored integers, not a second trigonometric/eigenvalue computation.
    auto stored=matrix(a.parameters,method==PCA?8:0,8,8);
    auto center=method==PCA?matrix(a.parameters,0,1,8):mean;
    size_t nv=(s.size()+7)/8;a.data.resize(nv*8);
    #pragma omp parallel for num_threads(a.threads) if(a.threads>1 && nv>64)
    for(size_t i=0;i<nv;++i){auto x=signal(s,fmt,a.exponent,i*8,1,8);Eigen::VectorXd coefficients=stored*(x-center).transpose();
        for(unsigned j=0;j<8;++j)a.data[i*8+j]=fixed_parameter(coefficients[j]);}
    return a;
}
std::vector<int64_t> reconstruct_transform(unsigned method,const Approximation& a,size_t n){std::vector<int64_t> out(n,0);size_t nv=(n+7)/8;
    if(method==MDCT){if(a.parameters.size()!=144||a.data.size()!=(nv+1)*8)throw std::runtime_error("MDCT window/frame layout differs");
        std::vector<__int128> accumulator((nv+2)*8,0);
        // Sine-windowed IMDCT: 2/N=1/4; 50% overlap. Keep the exact integer
        // overlap sum, then round once. Prefix/suffix priming frames are local.
        for(size_t f=0;f<nv+1;++f)for(unsigned i=0;i<16;++i){__int128 value=0;
            for(unsigned k=0;k<8;++k)value+=__int128(a.data[f*8+k])*a.parameters[16+i*8+k];accumulator[f*8+i]+=value*a.parameters[i];}
        for(size_t i=0;i<n;++i)out[i]=round128(accumulator[8+i],42);return out; // Q20^3 / (4*Q20^2).
    }
    if(a.parameters.size()!=(method==PCA?72:64)||a.data.size()!=nv*8)throw std::runtime_error("DCT/DST/PCA parameter layout differs");
    size_t base=method==PCA?8:0;for(size_t i=0;i<nv;++i)for(unsigned j=0;j<8&&i*8+j<n;++j){__int128 sum=0;
        for(unsigned k=0;k<8;++k)sum+=__int128(a.parameters[base+k*8+j])*a.data[i*8+k];out[i*8+j]=round128(sum,20)+(method==PCA?a.parameters[j]:0);}
    return out;
}
}
