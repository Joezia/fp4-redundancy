// Complete codeword-domain adaptations. All compensation is exact modulo 2^width.
#pragma once
#include "common.hpp"
namespace predictive {
enum Method { DELTA=10, XOR=11, DPCM=12, ADPCM=13, JPEGLS=14, PNG=15, FPC=16,
              SZ=17, LPC=18, LTP=19, MOTION=20, INTRA=21 };
inline int64_t prior(const Words& s,size_t i,size_t lag) { return i>=lag?s[i-lag]:0; }
inline int64_t spatial(const Words& s,size_t i,size_t cols,unsigned mode) {
    size_t y=i/cols,x=i%cols;
    int64_t a=x?s[i-1]:0,b=y?s[i-cols]:0,c=(x&&y)?s[i-cols-1]:0;
    if(mode==0)return 0;
    if(mode==1)return a;
    if(mode==2)return b;
    if(mode==3)return (a+b)/2;
    auto p=a+b-c,pa=std::abs(p-a),pb=std::abs(p-b),pc=std::abs(p-c);
    return pa<=pb&&pa<=pc?a:pb<=pc?b:c; // PNG Paeth tie rules.
}
inline int64_t intra(const Words& s,size_t i,size_t cols,unsigned mode) {
    size_t y=i/cols,x=i%cols,y0=y/8*8,x0=x/8*8,dy=y-y0,dx=x-x0;
    auto top=[&](int64_t at)->int64_t {return y0?s[(y0-1)*cols+size_t(std::clamp<int64_t>(at,0,cols-1))]:0;};
    auto left=[&](size_t at)->int64_t {return x0?s[at*cols+x0-1]:0;};
    if(mode==0)return top(x);
    if(mode==1)return left(y);
    if(mode==2)return top(x0+dx+dy+1); // diagonal down-left, replicated right edge.
    if(dx>dy)return top(x0+dx-dy-1);
    if(dy>dx)return left(y0+dy-dx-1);
    return y0&&x0?s[(y0-1)*cols+x0-1]:0;
}
inline uint64_t distance(uint16_t value,int64_t prediction,unsigned bits) {
    unsigned r=wrap(int64_t(value)-prediction,bits),range=1u<<bits;
    return std::min(r,range-r);
}
// FLAC autocorrelation/Levinson-Durbin recursion adapted from lpc.c (BSD-3-Clause).
// Copyright (C) 2000-2009 Josh Coalson; 2011-2025 Xiph.Org Foundation.
// The full upstream license is retained in third_party/FLAC-COPYING.Xiph.
inline std::vector<int16_t> lpc_fit(const Words& s,unsigned bits) {
    constexpr unsigned order=8; double autoc[order+1]={},lpc[order]={};
    size_t n=std::min<size_t>(s.size(),8192);int64_t center=int64_t(1)<<(bits-1);
    for(size_t i=0;i<n;++i)for(unsigned j=0;j<=order && j<=i;++j)
        autoc[j]+=double(int64_t(s[i])-center)*double(int64_t(s[i-j])-center);
    double err=autoc[0];std::vector<int16_t> q(order,0);
    for(unsigned i=0;i<order && err>autoc[0]*1e-12 && err>0;++i) {
        double r=-autoc[i+1];for(unsigned j=0;j<i;++j)r-=lpc[j]*autoc[i-j];r/=err;
        r=std::clamp(r,-0.999,0.999); lpc[i]=r;
        unsigned j=0;for(;j<(i>>1);++j){auto tmp=lpc[j];lpc[j]+=r*lpc[i-1-j];lpc[i-1-j]+=r*tmp;}
        if(i&1)lpc[j]+=lpc[j]*r;
        err*=1-r*r;
        for(j=0;j<=i;++j)q[j]=int16_t(std::clamp<long>(std::lround(-lpc[j]*4096),-32768,32767));
    }
    return q;
}
inline int64_t linear(const Words& s,size_t i,const std::vector<int16_t>& q,unsigned bits) {
    int64_t sum=0,center=int64_t(1)<<(bits-1);
    for(size_t j=0;j<q.size();++j)sum+=int64_t(q[j])*(i>j?int64_t(s[i-j-1])-center:0);
    return center+floor_shift(sum,12);
}
struct Parameters { Bytes encoded; std::vector<int16_t> q; };
inline Parameters choose(unsigned method,const Words& s,unsigned bits,size_t cols) {
    Parameters p;
    if(method==LPC) {
        p.q=lpc_fit(s,bits);for(auto v:p.q)put<int16_t>(p.encoded,v);
    } else if(method==PNG) {
        for(size_t row=0;row<s.size();row+=cols) {
            uint64_t best=UINT64_MAX;unsigned mode=0;
            for(unsigned m=0;m<5;++m){uint64_t cost=0;for(size_t i=row;i<std::min(row+cols,s.size());++i)cost+=distance(s[i],spatial(s,i,cols,m),bits);
                if(cost<best){best=cost;mode=m;}}
            p.encoded.push_back(mode);
        }
    } else if(method==LTP) {
        // One-tap pitch/lag + fitted gain, not an Opus/SILK bitstream or five-tap codec.
        for(size_t base=0;base<s.size();base+=256) {
            uint64_t best=UINT64_MAX;unsigned lag=1;int16_t gain=4096;int64_t center=int64_t(1)<<(bits-1);
            std::vector<unsigned> lags;for(unsigned l=1;l<=128;++l)lags.push_back(l);
            for(unsigned l=256;l<=4096;l*=2)lags.push_back(l);
            for(auto l:lags) {
                if(l>base && base)continue;
                int64_t xy=0,xx=0;
                for(size_t i=base;i<std::min(base+64,s.size());i+=4)if(i>=l) {
                    int64_t x=int64_t(s[i-l])-center,y=int64_t(s[i])-center;xy+=x*y;xx+=x*x;
                }
                int16_t g=xx?int16_t(std::clamp<int64_t>(xy*4096/xx,-4096,4096)):0;
                uint64_t cost=0;for(size_t i=base;i<std::min(base+64,s.size());i+=4)
                    cost+=distance(s[i],center+floor_shift(int64_t(g)*(i>=l?int64_t(s[i-l])-center:0),12),bits);
                if(cost<best){best=cost;lag=l;gain=g;}
            }
            put<uint16_t>(p.encoded,lag);put<int16_t>(p.encoded,gain);
        }
    } else if(method==INTRA) {
        size_t rows=(s.size()+cols-1)/cols;
        for(size_t y=0;y<rows;y+=8)for(size_t x=0;x<cols;x+=8) {
            uint64_t best=UINT64_MAX;unsigned mode=0;
            for(unsigned m=0;m<4;++m) {
                uint64_t cost=0;for(size_t yy=y;yy<std::min(y+8,rows);++yy)for(size_t xx=x;xx<std::min(x+8,cols);++xx) {
                    size_t i=yy*cols+xx;if(i<s.size())cost+=distance(s[i],intra(s,i,cols,m),bits);
                }
                if(cost<best){best=cost;mode=m;}
            }
            p.encoded.push_back(mode);
        }
    } else if(method==MOTION) {
        size_t rows=(s.size()+cols-1)/cols;
        for(size_t frame=0;frame<rows;frame+=16)
          for(size_t by=0;by<std::min<size_t>(16,rows-frame);by+=4)for(size_t x=0;x<cols;x+=16) {
            int bestx=0,besty=0;uint64_t best=UINT64_MAX;
            if(frame)for(int dy=-4;dy<=4;dy+=4)for(int dx=-8;dx<=8;dx+=4) {
                if(int64_t(x)+dx<0 || int64_t(x+std::min<size_t>(16,cols-x))+dx>int64_t(cols) || int64_t(by)+dy<0 ||
                   int64_t(by+std::min<size_t>(4,rows-frame-by))+dy>16)continue;
                uint64_t cost=0;
                for(size_t y=by;y<std::min(by+4,std::min<size_t>(16,rows-frame));++y)for(size_t xx=x;xx<std::min(x+16,cols);xx+=2) {
                    size_t i=(frame+y)*cols+xx,ref=(frame-16+y+dy)*cols+xx+dx;
                    if(i<s.size())cost+=distance(s[i],s[ref],bits);
                }
                if(cost<best){best=cost;bestx=dx;besty=dy;}
            }
            put<int8_t>(p.encoded,bestx);put<int8_t>(p.encoded,besty);
        }
    }
    return p;
}
inline Parameters read_parameters(unsigned method,const Bytes& b,size_t n,size_t cols) {
    Parameters p{b,{}};Reader r{b.data(),b.size()};
    size_t rows=(n+cols-1)/cols;
    if(method==LPC){for(unsigned i=0;i<8;++i)p.q.push_back(r.get<int16_t>());r.end();}
    else if(method==PNG){if(b.size()!=rows || std::any_of(b.begin(),b.end(),[](auto v){return v>4;}))throw std::runtime_error("Invalid PNG row modes");}
    else if(method==INTRA){if(b.size()!=((rows+7)/8)*((cols+7)/8) || std::any_of(b.begin(),b.end(),[](auto v){return v>3;}))throw std::runtime_error("Invalid intra modes");}
    else if(method==MOTION){if(b.size()!=((rows+3)/4)*((cols+15)/16)*2)throw std::runtime_error("Invalid motion vectors");}
    else if(method==LTP){if(b.size()!=((n+255)/256)*4)throw std::runtime_error("Invalid LTP parameters");
        while(r.pos<r.n){auto lag=r.get<uint16_t>();auto gain=r.get<int16_t>();if(!lag||lag>4096||gain<-4096||gain>4096)throw std::runtime_error("Invalid LTP lag/gain");}}
    else if(!b.empty())throw std::runtime_error("Unexpected prediction parameters");
    return p;
}
inline int64_t predict(unsigned method,const Words& s,size_t i,unsigned bits,size_t cols,const Parameters& p) {
    if(method==DELTA || method==XOR)return prior(s,i,1);
    if(method==DPCM)return 2*prior(s,i,1)-prior(s,i,2);
    if(method==PNG)return spatial(s,i,cols,p.encoded[i/cols]);
    if(method==SZ) {auto x=i%cols;return (x?s[i-1]:0)+(i>=cols?s[i-cols]:0)-(x&&i>=cols?s[i-cols-1]:0);}
    if(method==LPC)return linear(s,i,p.q,bits);
    if(method==LTP) {
        Reader r{p.encoded.data()+i/256*4,4};auto lag=r.get<uint16_t>();auto gain=r.get<int16_t>();
        int64_t center=int64_t(1)<<(bits-1);return center+floor_shift(int64_t(gain)*(i>=lag?int64_t(s[i-lag])-center:0),12);
    }
    if(method==INTRA)return intra(s,i,cols,p.encoded[((i/cols)/8)*((cols+7)/8)+(i%cols)/8]);
    if(method==MOTION) {
        size_t y=i/cols,x=i%cols,frame=y/16*16,index=((y/4)*((cols+15)/16)+x/16)*2;
        Reader r{p.encoded.data()+index,2};auto dx=r.get<int8_t>(),dy=r.get<int8_t>();
        if(!frame){if(dx||dy)throw std::runtime_error("Motion first frame has external reference");return 0;}
        int64_t yy=int64_t(frame)-16+int64_t(y%16)+dy,xx=int64_t(x)+dx;
        if(yy<int64_t(frame)-16||yy>=int64_t(frame)||xx<0||xx>=int64_t(cols))throw std::runtime_error("Motion reference outside prior frame");
        return s[size_t(yy)*cols+size_t(xx)];
    }
    throw std::runtime_error("Unknown B prediction");
}
}
