// Bounded autoregressive neural entropy model: FANN trains a per-component
// 8-8-1 network; all 81 quantized weights are serialized. Encoding and decoding
// both use the same integer inference, never floating-point decoder CDFs.
#pragma once
#include <cmath>
#include <random>
#include "third_party/FANN/src/include/floatfann.h"
namespace neural_codec {
constexpr unsigned INPUTS=8,HIDDEN=8,WEIGHTS=81,FRAC=12,SAMPLES=2048,EPOCHS=4;
constexpr int ONE=1<<FRAC;
class Model {
    uint32_t alphabet;unsigned w;
    std::array<uint32_t,4> history{};
    std::array<int16_t,WEIGHTS> weights{};
    std::array<int64_t,HIDDEN> base{};
    static std::array<int32_t,INPUTS> features(const std::array<uint32_t,4>&h,uint32_t prefix,unsigned depth,unsigned width,uint32_t a){
        std::array<int32_t,INPUTS> x{};
        for(unsigned i=0;i<4;++i)x[i]=int32_t(uint64_t(h[i])*2*ONE/(a-1))-ONE;
        x[4]=depth?int32_t(uint64_t(prefix-(1u<<depth))*2*ONE/(1u<<depth))-ONE:0;
        x[5]=depth*ONE/width;x[6]=width*ONE/16;x[7]=h[0]==0?ONE:0;return x;
    }
public:
    Model(const Symbols &symbols,uint32_t a):alphabet(a),w(a2_tree::width(a)){
        fann_disable_seed_rand();
        unsigned layers[]={INPUTS,HIDDEN,1};
        std::unique_ptr<fann,decltype(&fann_destroy)> ann(fann_create_standard_array(3,layers),fann_destroy);
        if(!ann)throw std::bad_alloc();
        fann_set_activation_function_hidden(ann.get(),FANN_LINEAR_PIECE_SYMMETRIC);
        fann_set_activation_steepness_hidden(ann.get(),1.0f);
        fann_set_activation_function_output(ann.get(),FANN_SIGMOID);
        fann_set_activation_steepness_output(ann.get(),0.5f);
        fann_set_training_algorithm(ann.get(),FANN_TRAIN_RPROP);
        if(fann_get_total_connections(ann.get())!=WEIGHTS)throw std::runtime_error("Neural FANN architecture");
        std::vector<fann_connection> connections(WEIGHTS);fann_get_connection_array(ann.get(),connections.data());
        std::mt19937 random(0x4e4e31);
        for(auto &c:connections)c.weight=(int(random()%2001)-1000)/10000.0f;
        fann_set_weight_array(ann.get(),connections.data(),WEIGHTS);
        uint64_t nbits=uint64_t(symbols.size())*w;unsigned n=unsigned(std::min<uint64_t>(nbits,SAMPLES));
        std::unique_ptr<fann_train_data,decltype(&fann_destroy_train)> data(fann_create_train(n,INPUTS,1),fann_destroy_train);
        if(!data)throw std::bad_alloc();
        for(unsigned i=0;i<n;++i){
            uint64_t index=uint64_t(i)*nbits/n;size_t word=index/w;unsigned d=index%w;
            std::array<uint32_t,4> h{};for(unsigned j=0;j<4;++j)if(word>j)h[j]=symbols[word-j-1];
            uint32_t prefix=(1u<<d)|(uint32_t(symbols[word])>>(w-d));auto x=features(h,prefix,d,w,a);
            for(unsigned j=0;j<INPUTS;++j)data->input[i][j]=float(x[j])/ONE;
            data->output[i][0]=(symbols[word]>>(w-d-1))&1;
        }
        fann_train_on_data(ann.get(),data.get(),EPOCHS,0,0.0f);
        fann_get_connection_array(ann.get(),connections.data());
        std::array<bool,WEIGHTS> present{};
        for(const auto &c:connections){
            unsigned index;
            if(c.to_neuron>=INPUTS+1&&c.to_neuron<INPUTS+1+HIDDEN&&c.from_neuron<=INPUTS)
                index=(c.to_neuron-INPUTS-1)*(INPUTS+1)+c.from_neuron;
            else if(c.to_neuron==INPUTS+HIDDEN+2&&c.from_neuron>=INPUTS+1&&c.from_neuron<=INPUTS+HIDDEN+1)
                index=HIDDEN*(INPUTS+1)+c.from_neuron-INPUTS-1;
            else throw std::runtime_error("Neural connection mapping");
            if(index>=WEIGHTS||present[index]||!std::isfinite(c.weight))throw std::runtime_error("Neural weight");
            weights[index]=int16_t(std::clamp<long>(std::lround(double(c.weight)*ONE),-32768,32767));present[index]=true;
        }
        for(bool p:present)if(!p)throw std::runtime_error("Missing neural weight");
    }
    Model(Reader &r,uint32_t a):alphabet(a),w(a2_tree::width(a)){
        for(uint32_t v:{INPUTS,HIDDEN,FRAC,SAMPLES,EPOCHS,WEIGHTS})if(r.get<uint32_t>()!=v)throw std::runtime_error("Neural configuration");
        for(auto &v:weights){uint32_t u=r.get<uint16_t>();v=int16_t(u<32768?int(u):int(u)-65536);}
    }
    void save(Bytes &b)const{for(uint32_t v:{INPUTS,HIDDEN,FRAC,SAMPLES,EPOCHS,WEIGHTS})put<uint32_t>(b,v);for(auto v:weights)put<uint16_t>(b,uint16_t(v));}
    void begin_word(){auto x=features(history,1,0,w,alphabet);
        for(unsigned h=0;h<HIDDEN;++h){base[h]=int64_t(weights[h*9+8])*ONE;for(unsigned i:{0u,1u,2u,3u,6u,7u})base[h]+=int64_t(weights[h*9+i])*x[i];}
    }
    uint32_t predict(uint32_t prefix,unsigned depth){
        auto x=features(history,prefix,depth,w,alphabet);int64_t output=int64_t(weights[80])*ONE;
        for(unsigned h=0;h<HIDDEN;++h){int64_t sum=base[h]+int64_t(weights[h*9+4])*x[4]+int64_t(weights[h*9+5])*x[5];
            int32_t activation=int32_t(std::clamp<int64_t>(sum/ONE,-ONE,ONE));output+=int64_t(weights[72+h])*activation;}
        return zpaq_mix::squash(int(std::clamp<int64_t>((output/ONE)/64,-2048,2047)));
    }
    void observe(unsigned,uint32_t,unsigned){}
    void finish_word(uint32_t s){for(unsigned i=3;i>0;--i)history[i]=history[i-1];history[0]=s;}
};
} // namespace neural_codec
