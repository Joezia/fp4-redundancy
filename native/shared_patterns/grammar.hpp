#pragma once
#include "common.hpp"
namespace patterns {
struct Grammar {std::vector<std::vector<uint32_t>> rules;std::vector<uint32_t> root;};
inline Stream grammar_stream(const Grammar& g,size_t count){Stream s{count,{}, {}};var(s.model,g.rules.size());
    for(const auto& rule:g.rules){var(s.model,rule.size());for(auto v:rule)var(s.model,v);}var(s.model,g.root.size());for(auto v:g.root)var(s.data,v);return s;}
inline Words grammar_decode(const Stream& stream,unsigned alphabet){Reader m{stream.model.data(),stream.model.size()},r{stream.data.data(),stream.data.size()};
    auto nr=var(m);if(nr>stream.count)throw std::runtime_error("C grammar has too many rules");Grammar g;g.rules.resize(nr);
    size_t all_symbols=0;for(auto& rhs:g.rules){auto n=var(m);if(!n||n>stream.count || all_symbols+n>stream.count*4)throw std::runtime_error("Invalid grammar rule length");
        all_symbols+=n;for(unsigned j=0;j<n;++j){auto v=var(m);if(v>=alphabet+nr)throw std::runtime_error("Undefined grammar symbol");rhs.push_back(v);}}
    auto roots=var(m);m.end();if(!roots||roots>stream.count)throw std::runtime_error("Invalid grammar root length");
    for(unsigned j=0;j<roots;++j){auto v=var(r);if(v>=alphabet+nr)throw std::runtime_error("Invalid root symbol");g.root.push_back(v);}r.end();
    // Validate arbitrary rule numbering, acyclicity and expansion lengths before decoding.
    std::vector<uint8_t> mark(nr,0);std::vector<size_t> lengths(nr,0);
    for(unsigned id=0;id<nr;++id)if(!mark[id]){
        std::vector<std::pair<unsigned,size_t>> stack{{id,0}};mark[id]=1;
        while(!stack.empty()){
            auto& top=stack.back();auto node=top.first;
            if(top.second==g.rules[node].size()) {mark[node]=2;stack.pop_back();continue;}
            auto v=g.rules[node][top.second];
            if(v<alphabet){++lengths[node];++top.second;}
            else {auto child=v-alphabet;if(mark[child]==1)throw std::runtime_error("Cyclic grammar");
                if(!mark[child]){mark[child]=1;stack.emplace_back(child,0);continue;}
                lengths[node]+=lengths[child];++top.second;}
            if(lengths[node]>stream.count)throw std::runtime_error("Grammar expansion exceeds component");
        }
    }
    size_t total=0;for(auto v:g.root){total+=v<alphabet?1:lengths[v-alphabet];if(total>stream.count)throw std::runtime_error("Grammar root expansion exceeds component");}
    if(total!=stream.count)throw std::runtime_error("Grammar does not cover original component");Words out;out.reserve(total);
    std::vector<uint32_t> stack(g.root.rbegin(),g.root.rend());
    while(!stack.empty()){auto v=stack.back();stack.pop_back();if(v<alphabet)out.push_back(v);else{
        const auto& rhs=g.rules[v-alphabet];stack.insert(stack.end(),rhs.rbegin(),rhs.rend());}}
    return out;
}
inline Stream repair_encode(const Words& s,unsigned alphabet){
    struct Node {uint32_t value;int prev,next;bool alive=true;};std::vector<Node> nodes;
    for(size_t i=0;i<s.size();++i)nodes.push_back({s[i],int(i)-1,i+1<s.size()?int(i+1):-1});
    struct Pair {std::set<int> positions;uint64_t generation=0;};
    struct Entry {size_t count;uint64_t key,generation;bool evaluated;
        bool operator<(const Entry& b)const{return count!=b.count?count<b.count:key!=b.key?key>b.key:evaluated<b.evaluated;}};
    std::unordered_map<uint64_t,Pair> pairs;std::priority_queue<Entry> heap;
    auto schedule=[&](uint64_t key,Pair& p){++p.generation;if(p.positions.size()>1)heap.push({p.positions.size(),key,p.generation,false});};
    auto remove=[&](int i){if(i<0||nodes[i].next<0)return;auto key=pair_key(nodes[i].value,nodes[nodes[i].next].value);auto& p=pairs[key];p.positions.erase(i);schedule(key,p);};
    auto add=[&](int i){if(i<0||nodes[i].next<0)return;auto key=pair_key(nodes[i].value,nodes[nodes[i].next].value);auto& p=pairs[key];p.positions.insert(i);schedule(key,p);};
    for(size_t i=0;i+1<s.size();++i)add(i);Grammar g;
    while(!heap.empty()){
        auto e=heap.top();heap.pop();auto& p=pairs[e.key];if(e.generation!=p.generation)continue;
        if(!e.evaluated){size_t frequency=0;int consumed=-1;for(int i:p.positions)if(i!=consumed){++frequency;consumed=nodes[i].next;}
            if(frequency>1)heap.push({frequency,e.key,e.generation,true});continue;}
        if(e.count<2)break;auto code=uint32_t(alphabet+g.rules.size());g.rules.push_back({uint32_t(e.key>>32),uint32_t(e.key)});
        std::vector<int> occurrences(p.positions.begin(),p.positions.end());
        for(int a:occurrences){if(!nodes[a].alive||nodes[a].next<0||pair_key(nodes[a].value,nodes[nodes[a].next].value)!=e.key)continue;
            int b=nodes[a].next,before=nodes[a].prev,after=nodes[b].next;
            remove(before);remove(a);remove(b);nodes[a].value=code;nodes[a].next=after;if(after>=0)nodes[after].prev=a;nodes[b].alive=false;
            add(before);add(a);}
    }
    for(int i=0;i>=0;i=nodes[i].next)g.root.push_back(nodes[i].value);return grammar_stream(g,s.size());
}
Stream sequitur_encode(const Words&,unsigned);
}
