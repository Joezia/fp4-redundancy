// Fixed complete PAQ-style context-mixing and CABAC-model adaptations.
#pragma once
#include "third_party/ZPAQ/mix.hpp"
#include "third_party/OpenH264/tables.hpp"
namespace context_models {
constexpr uint32_t SLOTS=65536;
struct Counter{uint16_t zero=1,one=1;uint32_t probability()const{return uint64_t(one)*32768/(zero+one);}void add(unsigned b){if(zero+one>=32768){zero=(zero+1)/2;one=(one+1)/2;}if(b)++one;else ++zero;}};
struct Entry{uint64_t key=0;Counter count;};
class Paq {
    std::vector<Counter> marginal;
    std::vector<Entry> conditional;
    std::array<uint32_t,16> weights{};
    uint32_t previous=0;unsigned w;
    Entry *last=nullptr;int log0=0,log1=0,prediction=0;
public:
    Paq(const Symbols&,uint32_t a):marginal(a),conditional(SLOTS),w(a2_tree::width(a)){weights.fill(32768);}
    Paq(Reader &r,uint32_t a):Paq(Symbols{},a){for(uint32_t v:{SLOTS,32u,32768u})if(r.get<uint32_t>()!=v)throw std::runtime_error("PAQ configuration");}
    void save(Bytes &b)const{for(uint32_t v:{SLOTS,32u,32768u})put<uint32_t>(b,v);}
    void begin_word(){}
    uint32_t predict(uint32_t prefix,unsigned d){
        uint64_t key=(uint64_t(previous)<<17)|prefix;
        auto &e=conditional[(key*0x9e3779b97f4a7c15ULL)>>48];
        if(e.key!=key){e.key=key;e.count=Counter{};}last=&e;
        log0=zpaq_mix::stretch(marginal.at(prefix).probability());log1=zpaq_mix::stretch(e.count.probability());
        prediction=zpaq_mix::mix2(weights[d],log0,log1);return zpaq_mix::squash(prediction);
    }
    void observe(unsigned bit,uint32_t prefix,unsigned d){weights[d]=zpaq_mix::update_mix2(weights[d],log0,log1,prediction,bit);marginal[prefix].add(bit);last->count.add(bit);}
    void finish_word(uint32_t s){previous=s;}
};
class Cabac {
    struct State{uint64_t key=0;uint8_t index=0,mps=0;};
    std::vector<State> states;
    State *last=nullptr;uint32_t previous=0;
public:
    Cabac(const Symbols&,uint32_t):states(SLOTS){}
    Cabac(Reader &r,uint32_t a):Cabac(Symbols{},a){for(uint32_t v:{SLOTS,2u,416u})if(r.get<uint32_t>()!=v)throw std::runtime_error("CABAC model configuration");}
    void save(Bytes &b)const{for(uint32_t v:{SLOTS,2u,416u})put<uint32_t>(b,v);}
    void begin_word(){}
    uint32_t predict(uint32_t prefix,unsigned){
        uint64_t key=(uint64_t(previous)<<17)|prefix;
        auto &s=states[(key*0x9e3779b97f4a7c15ULL)>>48];
        if(s.key!=key){s=State{};s.key=key;}last=&s;
        uint32_t lps=uint32_t(openh264::g_kuiCabacRangeLps[s.index][2])*32768/416;
        return s.mps?32768-lps:lps;
    }
    void observe(unsigned bit,uint32_t,unsigned){bool mps=bit==last->mps;auto old=last->index;if(!mps&&old==0)last->mps^=1;last->index=openh264::g_kuiStateTransTable[old][mps?1:0];}
    void finish_word(uint32_t s){previous=s;}
};
} // namespace context_models
