#include "common.hpp"
namespace mathematical {
std::pair<Words,uint32_t> bwt(const Words& s){size_t n=s.size();if(!n)throw std::runtime_error("Empty BWT block");
    std::vector<unsigned> order(n),rank(n),next(n),shifted(n),counts(std::max<size_t>(65536,n),0);
    for(auto v:s)++counts[v];for(unsigned i=1;i<65536;++i)counts[i]+=counts[i-1];for(size_t i=n;i--;)order[--counts[s[i]]]=i;
    unsigned classes=1;rank[order[0]]=0;for(size_t i=1;i<n;++i){if(s[order[i]]!=s[order[i-1]])++classes;rank[order[i]]=classes-1;}
    for(size_t length=1;length<n&&classes<n;length*=2){std::fill(counts.begin(),counts.begin()+classes,0);
        for(size_t i=0;i<n;++i){shifted[i]=(order[i]+n-length)%n;++counts[rank[shifted[i]]];}
        for(unsigned i=1;i<classes;++i)counts[i]+=counts[i-1];for(size_t i=n;i--;)order[--counts[rank[shifted[i]]]]=shifted[i];
        unsigned c=1;next[order[0]]=0;for(size_t i=1;i<n;++i){auto a=order[i-1],b=order[i];if(rank[a]!=rank[b]||rank[(a+length)%n]!=rank[(b+length)%n])++c;next[b]=c-1;}rank.swap(next);classes=c;
    }
    Words last(n);uint32_t primary=0;for(size_t i=0;i<n;++i){last[i]=s[(order[i]+n-1)%n];if(!order[i])primary=i;}return {last,primary};
}
Words inverse_bwt(const Words& last,uint32_t primary){size_t n=last.size();if(primary>=n)throw std::runtime_error("BWT primary outside block");
    std::vector<unsigned> counts(65536,0),starts(65536),lf(n);for(auto v:last)++counts[v];unsigned sum=0;
    for(unsigned v=0;v<65536;++v){starts[v]=sum;sum+=counts[v];}std::fill(counts.begin(),counts.end(),0);
    for(size_t i=0;i<n;++i)lf[i]=starts[last[i]]+counts[last[i]]++;
    Words out(n);unsigned row=primary;for(size_t i=n;i--;){out[i]=last[row];row=lf[row];}return out;
}
struct Fenwick {std::vector<unsigned> tree;explicit Fenwick(size_t n):tree(n+1,0){}
    void add(size_t i,int value){for(++i;i<tree.size();i+=i&-i)tree[i]+=value;}
    unsigned sum(size_t i)const{unsigned v=0;for(++i;i;i-=i&-i)v+=tree[i];return v;}
    size_t select(unsigned rank)const{size_t index=0,step=1;while(step*2<tree.size())step*=2;
        for(;step;step/=2){auto next=index+step;if(next<tree.size()&&tree[next]<=rank){rank-=tree[next];index=next;}}return index;}
};
Words mtf(const Words& s,unsigned alphabet,bool decode){size_t n=s.size();Fenwick f(n+alphabet);std::vector<size_t> positions(alphabet);Words at(n+alphabet),out(n);
    for(unsigned v=0;v<alphabet;++v){positions[v]=n+v;at[n+v]=v;f.add(n+v,1);}size_t front=n;
    for(size_t i=0;i<n;++i){unsigned symbol;size_t pos;
        if(decode){if(s[i]>=alphabet)throw std::runtime_error("MTF rank outside alphabet");pos=f.select(s[i]);symbol=at[pos];out[i]=symbol;}
        else{symbol=s[i];if(symbol>=alphabet)throw std::runtime_error("MTF symbol outside alphabet");pos=positions[symbol];out[i]=f.sum(pos)-1;}
        f.add(pos,-1);--front;f.add(front,1);positions[symbol]=front;at[front]=symbol;
    }return out;
}
I32 lifting(const I32& input,bool cdf,bool inverse){I32 out=input;size_t n=input.size();std::vector<size_t> levels;
    for(size_t len=n;len>1;len=(len+1)/2)levels.push_back(len);if(inverse)std::reverse(levels.begin(),levels.end());
    for(auto len:levels){size_t nl=(len+1)/2,nh=len/2;I32 low(nl),high(nh);
        if(!inverse){for(size_t i=0;i<nl;++i)low[i]=out[2*i];
            for(size_t i=0;i<nh;++i)high[i]=checked32(int64_t(out[2*i+1])-(cdf?floor_shift(int64_t(low[i])+low[std::min(i+1,nl-1)],1):low[i]));
            for(size_t i=0;i<nl;++i)low[i]=checked32(int64_t(low[i])+(cdf?floor_shift(int64_t(high[i?i-1:0])+high[std::min(i,nh-1)]+2,2):(i<nh?floor_shift(high[i],1):0)));
            std::copy(low.begin(),low.end(),out.begin());std::copy(high.begin(),high.end(),out.begin()+nl);
        }else{std::copy(out.begin(),out.begin()+nl,low.begin());std::copy(out.begin()+nl,out.begin()+len,high.begin());
            for(size_t i=0;i<nl;++i)low[i]=checked32(int64_t(low[i])-(cdf?floor_shift(int64_t(high[i?i-1:0])+high[std::min(i,nh-1)]+2,2):(i<nh?floor_shift(high[i],1):0)));
            for(size_t i=0;i<nh;++i)high[i]=checked32(int64_t(high[i])+(cdf?floor_shift(int64_t(low[i])+low[std::min(i+1,nl-1)],1):low[i]));
            for(size_t i=0;i<nl;++i)out[2*i]=low[i];for(size_t i=0;i<nh;++i)out[2*i+1]=high[i];
        }
    }return out;
}
Stream exact_encode(const Words& s,unsigned bits,unsigned method){Bytes control;std::vector<Stream> streams;
    if(method==BWT_MTF){put<uint32_t>(control,65536);Words ranks;ranks.reserve(s.size());
        for(size_t i=0;i<s.size();i+=65536){auto last=bwt(Words(s.begin()+i,s.begin()+std::min(i+65536,s.size())));put<uint32_t>(control,last.second);
            auto part=mtf(last.first,1u<<bits,false);ranks.insert(ranks.end(),part.begin(),part.end());}streams.push_back(entropy_encode(ranks));
    }else if(method==BITPLANE){put<uint32_t>(control,bits);for(unsigned p=bits;p--;){Words plane(s.size());for(size_t i=0;i<s.size();++i)plane[i]=(s[i]>>p)&1;streams.push_back(entropy_encode(plane));}}
    else{put<uint32_t>(control,256);I32 coefficients;coefficients.reserve(s.size());for(size_t i=0;i<s.size();i+=256){I32 block(s.begin()+i,s.begin()+std::min(i+256,s.size()));auto c=lifting(block,method==CDF53,false);coefficients.insert(coefficients.end(),c.begin(),c.end());}streams.push_back(encode_i32(coefficients));}
    return join(s.size(),1u<<bits,control,streams);
}
Words exact_decode(const Stream& s,unsigned bits,unsigned method){auto parts=split(s,1u<<bits);Reader r{parts.control.data(),parts.control.size()};Words out;out.reserve(s.count);
    if(method==BWT_MTF){if(r.get<uint32_t>()!=65536||parts.streams.size()!=1||parts.streams[0].count!=s.count)throw std::runtime_error("Invalid BWT block layout");auto ranks=entropy_decode(parts.streams[0]);
        for(size_t i=0;i<s.count;i+=65536){auto primary=r.get<uint32_t>();auto last=mtf(Words(ranks.begin()+i,ranks.begin()+std::min(i+65536,size_t(s.count))),1u<<bits,true);auto block=inverse_bwt(last,primary);out.insert(out.end(),block.begin(),block.end());}
    }else if(method==BITPLANE){if(r.get<uint32_t>()!=bits||parts.streams.size()!=bits)throw std::runtime_error("Invalid bit-plane layout");out.resize(s.count,0);
        for(unsigned p=0;p<bits;++p){if(parts.streams[p].count!=s.count)throw std::runtime_error("Bit-plane length differs");auto plane=entropy_decode(parts.streams[p]);for(size_t i=0;i<s.count;++i){if(plane[i]>1)throw std::runtime_error("Invalid bit-plane symbol");out[i]|=uint16_t(plane[i]<<(bits-1-p));}}
    }else{if(r.get<uint32_t>()!=256||parts.streams.size()!=1||parts.streams[0].count!=s.count*2)throw std::runtime_error("Invalid lifting layout");auto c=decode_i32(parts.streams[0]);
        for(size_t i=0;i<c.size();i+=256){auto block=lifting(I32(c.begin()+i,c.begin()+std::min(i+256,c.size())),method==CDF53,true);for(auto v:block){if(v<0||unsigned(v)>=(1u<<bits))throw std::runtime_error("Lifting reconstructed codeword outside alphabet");out.push_back(v);}}
    }r.end();return out;
}
}
