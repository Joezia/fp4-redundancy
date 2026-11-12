#include "common.hpp"
namespace mathematical {
Mat fixed_dictionary(){Mat dictionary(8,16);dictionary.leftCols(8)=Mat::Identity(8,8);dictionary.rightCols(8)=dct_basis().transpose();return dictionary;}
std::pair<std::vector<unsigned>,Eigen::VectorXd> omp(const Mat& dictionary,const Eigen::VectorXd& x){std::vector<unsigned> selected;Eigen::VectorXd residual=x,coefficients;
    for(unsigned step=0;step<3;++step){double best=0;unsigned atom=16;
        for(unsigned k=0;k<16;++k)if(std::find(selected.begin(),selected.end(),k)==selected.end()){
            double norm=dictionary.col(k).norm(),score=norm>1e-12?std::abs(dictionary.col(k).dot(residual))/norm:0;
            if(score>best){best=score;atom=k;}}
        if(atom==16||best<1e-12)break;selected.push_back(atom);Mat active(8,selected.size());for(size_t i=0;i<selected.size();++i)active.col(i)=dictionary.col(selected[i]);
        Eigen::JacobiSVD<Mat> solver(active,Eigen::ComputeThinU|Eigen::ComputeThinV);solver.setThreshold(1e-10);coefficients=solver.solve(x);residual=x-active*coefficients;
        if(residual.squaredNorm()<1e-20)break;
    }
    return {selected,coefficients};
}
Mat train_ksvd(const Mat& x){Mat dictionary=fixed_dictionary();
    for(unsigned k=0;k<16&&k<unsigned(x.cols());++k){double norm=x.col(k).norm();if(norm>1e-12)dictionary.col(k)=x.col(k)/norm;}
    Mat codes=Mat::Zero(16,x.cols());
    for(unsigned iteration=0;iteration<4;++iteration){codes.setZero();for(Eigen::Index i=0;i<x.cols();++i){auto a=omp(dictionary,x.col(i));for(size_t j=0;j<a.first.size();++j)codes(a.first[j],i)=a.second[j];}
        for(unsigned atom=0;atom<16;++atom){std::vector<unsigned> used;for(Eigen::Index i=0;i<x.cols();++i)if(std::abs(codes(atom,i))>1e-12)used.push_back(i);
            if(used.empty()){Mat error=x-dictionary*codes;Eigen::Index best=0;double maximum=0;
                for(Eigen::Index i=0;i<x.cols();++i)if(error.col(i).squaredNorm()>maximum){maximum=error.col(i).squaredNorm();best=i;}
                if(maximum>1e-20)dictionary.col(atom)=error.col(best)/std::sqrt(maximum);continue;}
            Mat error(8,used.size());for(size_t i=0;i<used.size();++i){unsigned col=used[i];error.col(i)=x.col(col)-dictionary*codes.col(col)+dictionary.col(atom)*codes(atom,col);}
            Eigen::JacobiSVD<Mat> decomposition(error,Eigen::ComputeThinU|Eigen::ComputeThinV);
            dictionary.col(atom)=decomposition.matrixU().col(0);
            for(size_t i=0;i<used.size();++i)codes(atom,used[i])=decomposition.singularValues()[0]*decomposition.matrixV()(i,0);
        }
    }return dictionary;
}
Approximation fit_sparse(const Words& s,unsigned fmt,unsigned method){Approximation a;a.exponent=scale_exponent(s,fmt);a.threads=workers();size_t nv=(s.size()+7)/8,nt=std::min<size_t>(nv,512);Mat dict;
    if(method==SPARSE)dict=fixed_dictionary();else{Mat x(8,nt);for(size_t i=0;i<nt;++i)x.col(i)=signal(s,fmt,a.exponent,i*8,1,8).transpose();dict=train_ksvd(x);}
    append(a.parameters,dict);dict=matrix(a.parameters,0,8,16);
    // Independent OMP problems may run in parallel; records are flattened in
    // original vector order so that scheduling cannot reorder coefficients.
    std::vector<I32> records(nv);std::vector<std::exception_ptr> errors(nv);
    #pragma omp parallel for num_threads(a.threads) if(a.threads>1 && nv>64)
    for(size_t i=0;i<nv;++i){try{auto code=omp(dict,signal(s,fmt,a.exponent,i*8,1,8).transpose());auto& record=records[i];record.push_back(0);
        for(size_t j=0;j<code.first.size();++j){auto coefficient=fixed_parameter(code.second[j]);if(coefficient){record.push_back(code.first[j]);record.push_back(coefficient);++record[0];}}}catch(...){errors[i]=std::current_exception();}}
    for(auto& error:errors)if(error)std::rethrow_exception(error);
    for(auto& record:records)a.data.insert(a.data.end(),record.begin(),record.end());return a;
}
std::vector<int64_t> reconstruct_sparse(unsigned,const Approximation& a,size_t n){if(a.parameters.size()!=128)throw std::runtime_error("Sparse dictionary dimensions differ");std::vector<int64_t> out(n);size_t offset=0;
    for(size_t i=0;i<(n+7)/8;++i){if(offset==a.data.size())throw std::runtime_error("Truncated sparse vector");auto nnz=a.data[offset++];if(nnz<0||nnz>3||size_t(nnz)*2>a.data.size()-offset)throw std::runtime_error("Invalid sparse support count");
        std::array<__int128,8> sum{};unsigned seen=0;
        for(int j=0;j<nnz;++j){auto atom=a.data[offset++],coefficient=a.data[offset++];if(atom<0||atom>=16||(seen&(1u<<atom)))throw std::runtime_error("Invalid/duplicate sparse atom");seen|=1u<<atom;
            for(unsigned k=0;k<8;++k)sum[k]+=__int128(a.parameters[k*16+atom])*coefficient;}
        for(unsigned k=0;k<8&&i*8+k<n;++k)out[i*8+k]=round128(sum[k],20);
    }if(offset!=a.data.size())throw std::runtime_error("Trailing sparse coefficients");return out;
}
}
