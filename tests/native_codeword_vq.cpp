#include "../native/codeword_vq/vq.cpp"
#include <iostream>
int main(){using namespace codeword_vq;
    Words words;for(unsigned i=0;i<4096;++i)words.push_back((i%2)+1);
    auto a=correction(words,Words(words.size()/2,0),2,1,16,false);
    auto reference=entropy_encode(words);
    if(a.data!=reference.data)throw std::runtime_error("Shared correction differs from A1 arithmetic stream");
    auto counts=std::vector<uint32_t>(16,0);for(auto v:words)++counts[v];auto d=distribution(counts).save();
    if(Bytes(reference.model.begin(),reference.model.end()-8)!=d)throw std::runtime_error("CDF normalization differs from A1");
    Words ids(words.size()/2);for(size_t i=0;i<ids.size();++i)ids[i]=i%2;
    for(size_t i=0;i<words.size();++i)words[i]=1+ids[i/2]*2+i%2;
    auto c=correction(words,ids,2,2,16,true);
    if(c.data.size()>=a.data.size()||decode_correction(c,ids,2,2,16,true)!=words)
        throw std::runtime_error("Conditional correction did not exploit known center/position structure");
    Fit f;f.book={{0,0},{1,1}};f.index_cost={10,0};f.error_cost=std::vector<double>(16,1);f.error_cost[0]=0;
    if(nearest({0,0},f,false)!=0||nearest({0,0},f,true)!=1)
        throw std::runtime_error("Rate assignment ignored index cost");
    auto trained=train(words,16,{4,16,2,1});
    for(auto& v:trained.book){bool found=false;for(size_t i=0;i<(words.size()+3)/4;++i)found|=vector_at(words,i,4)==v;
        if(!found)throw std::runtime_error("Center is not an observed codeword vector");}
    size_t cases=0;
    for(unsigned fmt:{0,1,2,3,4,6})for(unsigned n:{1,2,3,7,16,31,64,257})for(unsigned dim:{2,4,8})for(unsigned centers:{4,16,64}){
        Bytes raw(n);for(unsigned i=0;i<n;++i)raw[i]=uint8_t(i*127+i/7);
        unsigned flags=dim|(centers<<8)|(2<<16)|(1<<18);
        auto frame=encode_frame(CODEC,fmt,flags,raw.data(),raw.size());
        if(decode_frame(frame.data(),frame.size())!=raw)throw std::runtime_error("Codeword VQ boundary bytes differ");++cases;
    }
    std::cout<<cases<<" boundary cases plus A1 bitstream, conditional model, medoid and rate-objective checks passed\n";
}
