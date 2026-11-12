#include "../native/math_structure/common.hpp"
#include <iostream>
using namespace mathematical;
static void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
int main(){
    Words banana{'B','A','N','A','N','A'};auto transformed=bwt(banana);
    require(transformed.first==Words{'N','N','B','A','A','A'}&&transformed.second==3,"BWT golden vector");
    require(mtf(transformed.first,256,false)==Words{78,0,67,67,0,0},"MTF golden vector");
    require(inverse_bwt(transformed.first,transformed.second)==banana,"BWT LF inverse");
    require(lifting({1,2,3,4},true,false)==I32{2,2,0,1},"CDF53 lifting golden vector");
    require(lifting({1,2,3,4},false,false)==I32{2,2,1,1},"Haar S-transform golden vector");
    require(floor_shift(-1,1)==-1&&floor_shift(-3,1)==-2,"Signed floor convention");
    bool overflow=false;try{lifting({INT32_MIN,INT32_MAX},false,false);}catch(const std::runtime_error&){overflow=true;}require(overflow,"Integer overflow must fail explicitly");
    for(auto basis:{dct_basis(),dst_basis()})require((basis*basis.transpose()-Mat::Identity(8,8)).norm()<1e-12,"DCT/DST orthonormal basis");
    auto window=mdct_window(),cosines=mdct_cosines();for(unsigned i=0;i<8;++i)require(std::abs(window(0,i)*window(0,i)+window(0,i+8)*window(0,i+8)-1)<1e-12,"Princen-Bradley sine window");
    // Test TDAC without ANY correction: first/last impulses, partial tails,
    // correct analysis, synthesis normalization, padding and overlap-add.
    for(size_t n:{size_t(1),size_t(7),size_t(8),size_t(9),size_t(17),size_t(39)}){
        std::vector<double> x(n);for(size_t i=0;i<n;++i)x[i]=std::sin(i*.37);x.front()+=1;x.back()-=.5;
        size_t nv=(n+7)/8;std::vector<double> ola((nv+2)*8,0);
        for(size_t frame=0;frame<nv+1;++frame){double coefficients[8]={};
            for(unsigned k=0;k<8;++k)for(unsigned i=0;i<16;++i){int64_t at=int64_t(frame*8+i)-8;if(at>=0&&size_t(at)<n)coefficients[k]+=x[at]*window(0,i)*cosines(i,k);}
            for(unsigned i=0;i<16;++i)for(unsigned k=0;k<8;++k)ola[frame*8+i]+=.25*coefficients[k]*cosines(i,k)*window(0,i);
        }
        for(size_t i=0;i<n;++i)require(std::abs(ola[8+i]-x[i])<1e-11,"MDCT TDAC/boundary reconstruction failed");
    }
    Mat identity=Mat::Zero(8,16);identity.leftCols(8)=Mat::Identity(8,8);Eigen::VectorXd x=Eigen::VectorXd::Zero(8);x[0]=1;x[3]=2;x[6]=-.5;
    auto sparse=mathematical::omp(identity,x);require(sparse.first.size()==3,"OMP support solving missing");Mat active(8,3);for(unsigned j=0;j<3;++j)active.col(j)=identity.col(sparse.first[j]);
    require((active*sparse.second-x).norm()<1e-12,"OMP orthogonal coefficient refit");
    Mat training(8,64);for(unsigned j=0;j<64;++j)for(unsigned i=0;i<8;++i)training(i,j)=std::sin(.17*(i+1)*(j+1))+.1*(i==j%8);
    auto learned=train_ksvd(training);require(learned.allFinite()&&(learned-fixed_dictionary()).norm()>.1,"K-SVD did not learn a dictionary");
    for(unsigned k=0;k<16;++k)require(std::abs(learned.col(k).norm()-1)<1e-9,"K-SVD atom normalization");
    Mat matrix8(8,8);for(unsigned i=0;i<8;++i)for(unsigned j=0;j<8;++j)matrix8(i,j)=double(int(i)-3)*(j+1)/128;
    Mat tensor(8,8);for(unsigned i=0;i<4;++i)for(unsigned j=0;j<4;++j)for(unsigned k=0;k<4;++k){auto at=i*16+j*4+k;tensor(at/8,at%8)=double(i+1)*(int(j)-1)*(k+2)/128;}
    for(unsigned method:{unsigned(SVD),unsigned(CP),unsigned(TUCKER),unsigned(TT),unsigned(KRONECKER),unsigned(CIRCULANT)}){
        Mat input=method==SVD?matrix8:tensor;
        if(method==KRONECKER)for(unsigned a=0;a<2;++a)for(unsigned b=0;b<2;++b)for(unsigned i=0;i<4;++i)for(unsigned j=0;j<4;++j)input(a*4+i,b*4+j)=double(int(a*2+b)-1)*(int(i*4+j)-7)/64;
        if(method==CIRCULANT)for(unsigned i=0;i<8;++i)for(unsigned j=0;j<8;++j)input(i,j)=double(int((j+8-i)%8)-3)/8;
        auto factors=factor_tile(method,input);auto predicted=reconstruct_factor_tile(method,factors);double error=0;
        for(unsigned i=0;i<64;++i)error=std::max(error,std::abs(double(predicted[i])/UNIT-input(i/8,i%8)));
        require(error<5e-5,"Factorization did not reconstruct known mathematical structure");
    }
    size_t cases=0;for(unsigned fmt=0;fmt<7;++fmt)for(size_t bytes:{size_t(1),size_t(2),size_t(3),size_t(7),size_t(16),size_t(31),size_t(64),size_t(257)}){
        if(fmt==5&&bytes%2)continue;Bytes raw(bytes);for(size_t i=0;i<bytes;++i)raw[i]=uint8_t(i*127+i/3);auto words=unpack(fmt,raw.data(),raw.size());
        for(unsigned method=BWT_MTF;method<=CIRCULANT;++method){Words restored;
            if(exact(method))restored=exact_decode(exact_encode(words,width(fmt),method),width(fmt),method);
            else{auto a=method<=PCA?fit_transform(words,fmt,method):method<=KSVD?fit_sparse(words,fmt,method):fit_factors(words,fmt,method);
                auto state=read_parameters(method,parameters(method,a),words.size());state.data=decode_i32(encode_i32(a.data));auto prediction=reconstruct(method,state,words.size());Projection projection(fmt,state.exponent);
                Words correction(words.size());for(size_t i=0;i<words.size();++i)correction[i]=words[i]^projection.word(prediction[i]);correction=entropy_decode(entropy_encode(correction));
                restored.resize(words.size());for(size_t i=0;i<words.size();++i)restored[i]=correction[i]^projection.word(prediction[i]);}
            require(pack(fmt,restored,raw.size())==raw,"D boundary byte restoration");++cases;
        }
    }
    std::cout<<cases<<" D boundary cases and transform/TDAC/OMP/K-SVD/factorization mechanism checks passed\n";
}
