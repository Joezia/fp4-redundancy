#include "common.hpp"
#include "wide_predictors.hpp"
namespace grouped {
struct Writer{Bytes data;uint64_t bits=0;void put(uint64_t v,unsigned n){for(unsigned i=n;i;--i){if(bits%8==0)data.push_back(0);data.back()|=uint8_t(((v>>(i-1))&1)<<(7-bits%8));++bits;}}};
struct Bits{const Bytes& data;uint64_t n,p=0;uint64_t get(unsigned b){if(b>n-p)throw std::runtime_error("Grouped bitstream truncated");uint64_t v=0;while(b--){v=(v<<1)|((data[p/8]>>(7-p%8))&1);++p;}return v;}void end(){if(p!=n||(n+7)/8!=data.size()||(n%8&&(data.back()&((1u<<(8-n%8))-1))))throw std::runtime_error("Grouped trailing bits");}};
Stream rice(const Words& s,unsigned b){unsigned best=0;__uint128_t cost=~__uint128_t(0);
 for(unsigned k=0;k<=b;++k){__uint128_t n=__uint128_t(s.size())*(k+1);for(auto v:s)n+=v>>k;if(n<cost){cost=n;best=k;}}
 if(cost>uint64_t(64u<<20)*8)throw std::runtime_error("RESOURCE_LIMIT: Rice payload exceeds 64MiB component cap");Writer w;
 for(auto v:s){uint64_t q=v>>best;while(q>=32){w.put(0,32);q-=32;}w.put(0,q);w.put(1,1);w.put(v,best);}
 if(w.bits!=cost)throw std::runtime_error("Grouped Rice cost mismatch");Stream out{s.size(),{},w.data};put<uint32_t>(out.model,best);put<uint64_t>(out.model,w.bits);return out;
}
Words unrice(const Stream& s,unsigned b){Reader r{s.model.data(),s.model.size()};unsigned k=r.get<uint32_t>();auto n=r.get<uint64_t>();r.end();
 if(k>b||n>uint64_t(s.count)*(b+1)||n>uint64_t(64u<<20)*8)throw std::runtime_error("Grouped Rice bounds");Bits bits{s.data,n};Words out(s.count);
 for(auto& v:out){uint64_t q=0,limit=mask(b)>>k;while(!bits.get(1)){if(q==limit)throw std::runtime_error("Grouped Rice quotient overflow");++q;}v=(q<<k)|bits.get(k);}bits.end();return out;}
struct FPC{std::vector<uint64_t> f=std::vector<uint64_t>(4096),d=std::vector<uint64_t>(4096);unsigned h=0,dh=0,b;uint64_t last=0;
 explicit FPC(unsigned bits):b(bits){}uint64_t first()const{return f[h];}uint64_t second()const{return (last+d[dh])&mask(b);}
 void update(uint64_t v){auto step=(v-last)&mask(b);f[h]=v;d[dh]=step;h=((h<<6)^(v>>(b>16?b-16:0)))&4095;dh=((dh<<2)^(step>>(b>24?b-24:0)))&4095;last=v;}};
Stream fpc(const Words& s,unsigned b){FPC state(b);Stream out{s.size(),{},Bytes((s.size()+1)/2,0)};put<uint32_t>(out.model,2);
 for(size_t i=0;i<s.size();++i){auto a=s[i]^state.first(),c=s[i]^state.second();bool use=c<a;auto residual=use?c:a;unsigned bytes=residual?(64-__builtin_clzll(residual)+7)/8:0;
  unsigned code=bytes>=7?7:bytes,stored=code==7?8:code;out.data[i/2]|=uint8_t((use*8+code)<<((i%2)?0:4));word(out.data,residual,stored*8);state.update(s[i]);}return out;}
Words unfpc(const Stream& s,unsigned b){Reader model{s.model.data(),s.model.size()};if(model.get<uint32_t>()!=2)throw std::runtime_error("Grouped FPC format");model.end();
 Reader r{s.data.data(),s.data.size()};auto prefix=r.take((s.count+1)/2);if(s.count%2&&(prefix.back()&15))throw std::runtime_error("Grouped FPC padding");FPC state(b);Words out(s.count);
 for(size_t i=0;i<out.size();++i){unsigned p=(prefix[i/2]>>((i%2)?0:4))&15,n=(p&7)==7?8:p&7;if(n>b/8)throw std::runtime_error("Grouped FPC residual bytes");auto v=word(r,n*8);v^=(p&8)?state.second():state.first();if(v>mask(b))throw std::runtime_error("Grouped FPC value");out[i]=v;state.update(v);}r.end();return out;}
Stream lifting(const Words& s,unsigned b,bool cdf){std::vector<int64_t> all;for(size_t base=0;base<s.size();base+=256){std::vector<int64_t> v(s.begin()+base,s.begin()+std::min(base+256,s.size()));
  for(size_t n=v.size();n>1;n=(n+1)/2){size_t nl=(n+1)/2,nh=n/2;std::vector<int64_t> lo(nl),hi(nh);for(size_t i=0;i<nl;++i)lo[i]=v[2*i];
   for(size_t i=0;i<nh;++i)hi[i]=v[2*i+1]-(cdf?floor_shift(lo[i]+lo[std::min(i+1,nl-1)],1):lo[i]);
   for(size_t i=0;i<nl;++i)lo[i]+=cdf?floor_shift(hi[i?i-1:0]+hi[std::min(i,nh-1)]+2,2):(i<nh?floor_shift(hi[i],1):0);
   std::copy(lo.begin(),lo.end(),v.begin());std::copy(hi.begin(),hi.end(),v.begin()+nl);}
  all.insert(all.end(),v.begin(),v.end());}
 Words codes;unsigned width=1;for(auto v:all){uint64_t u=(uint64_t(v)<<1)^(v<0?UINT64_MAX:0);codes.push_back(u);if(u)width=std::max(width,unsigned(64-__builtin_clzll(u)));}
 // Existing arithmetic coder on fixed 16-bit limbs; output width is measured,
 // not assumed equal to the source word width. No integer clipping.
 unsigned limbs=(width+15)/16;Words serial;for(auto u:codes)for(unsigned j=0;j<limbs;++j)serial.push_back((u>>(16*j))&65535);
 auto out=entropy(serial,16);out.count=s.size();Bytes model;put<uint32_t>(model,256);put<uint32_t>(model,limbs);gblob(model,out.model);out.model=std::move(model);return out;}
Words unlifting(const Stream& s,unsigned b,bool cdf){Reader r{s.model.data(),s.model.size()};if(r.get<uint32_t>()!=256)throw std::runtime_error("Grouped lifting block");unsigned limbs=r.get<uint32_t>();if(!limbs||limbs>4)throw std::runtime_error("Grouped lifting limbs");auto model=gblob(r);r.end();auto values=entropy_decode({s.count*limbs,model,s.data},16);
 std::vector<int64_t> all;for(size_t i=0;i<s.count;++i){uint64_t u=0;for(unsigned j=0;j<limbs;++j)u|=values[i*limbs+j]<<(16*j);if(u>(uint64_t(1)<<49))throw std::runtime_error("Grouped lifting coefficient bound");all.push_back(u&1?-1-int64_t(u>>1):int64_t(u>>1));}
 Words out;for(size_t base=0;base<s.count;base+=256){std::vector<int64_t> v(all.begin()+base,all.begin()+std::min(base+256,s.count));std::vector<size_t> levels;for(size_t n=v.size();n>1;n=(n+1)/2)levels.push_back(n);
  for(auto it=levels.rbegin();it!=levels.rend();++it){size_t n=*it,nl=(n+1)/2,nh=n/2;std::vector<int64_t> lo(v.begin(),v.begin()+nl),hi(v.begin()+nl,v.begin()+n);
   for(size_t i=0;i<nl;++i)lo[i]-=cdf?floor_shift(hi[i?i-1:0]+hi[std::min(i,nh-1)]+2,2):(i<nh?floor_shift(hi[i],1):0);
   for(size_t i=0;i<nh;++i)hi[i]+=cdf?floor_shift(lo[i]+lo[std::min(i+1,nl-1)],1):lo[i];
   for(size_t i=0;i<nl;++i)v[2*i]=lo[i];for(size_t i=0;i<nh;++i)v[2*i+1]=hi[i];}
  for(auto v:v){if(v<0||uint64_t(v)>mask(b))throw std::runtime_error("Grouped lifting inverse range");out.push_back(v);}}
 return out;}
Stream encode_method(unsigned method,unsigned b,unsigned cols,const Words& s){
 if(b==64&&method!=1&&method!=10&&method!=11&&method!=16&&method!=30)throw std::runtime_error("Unsupported 64-bit grouped technique");
 if(method<=2)return entropy(s,b,method);if(method==3)return rice(s,b);if(method==16)return fpc(s,b);
 if(method==31||method==33||method==38)return legacy_match(method,b,s);
 if((method>=41&&method<=45)||method==61||method==62||method==63||method==65||(method>=68&&method<=75))return numeric_encode(method,b,s);
 if(method==30){Stream out{s.size(),{}, {}};put<uint32_t>(out.model,2);for(size_t i=0;i<s.size();){size_t j=i+1;while(j<s.size()&&s[j]==s[i])++j;var(out.data,j-i);word(out.data,s[i],b);i=j;}return out;}
 if(method==64||method==66)return lifting(s,b,method==64);
 if(method==67){Stream out{s.size(),{}, {}};put<uint32_t>(out.model,b);for(unsigned p=b;p--;){Words plane;for(auto v:s)plane.push_back((v>>p)&1);auto stream=entropy(plane,8);gblob(out.model,stream.model);put<uint64_t>(out.model,stream.data.size());out.data.insert(out.data.end(),stream.data.begin(),stream.data.end());}return out;}
 Words residuals(s.size());Bytes parameters;
 if(method==10||method==11){uint64_t previous=0;for(size_t i=0;i<s.size();++i){residuals[i]=method==10?(s[i]-previous)&mask(b):s[i]^previous;previous=s[i];}}
 else{auto p=wide_predictive::choose(method,s,b,cols);parameters=p.encoded;for(size_t i=0;i<s.size();++i)residuals[i]=wrap(int64_t(s[i])-wide_predictive::predict(method,s,i,b,cols,p),b);}
 auto out=method==17?sz_store(residuals,b):entropy(residuals,b);Bytes model;gblob(model,parameters);gblob(model,out.model);out.model=std::move(model);return out;
}
Words decode_method(unsigned method,unsigned b,unsigned cols,const Stream& s){
 if(b==64&&method!=1&&method!=10&&method!=11&&method!=16&&method!=30)throw std::runtime_error("Unsupported 64-bit grouped technique");
 if(method<=2)return entropy_decode(s,b,method);if(method==3)return unrice(s,b);if(method==16)return unfpc(s,b);
 if(method==80)return categorical_decode(b,s);
 if(method==31||method==33||method==38)return legacy_unmatch(method,b,s);
 if((method>=41&&method<=45)||method==61||method==62||method==63||method==65||(method>=68&&method<=75))return numeric_decode(method,b,s);
 if(method==30){Reader m{s.model.data(),s.model.size()};if(m.get<uint32_t>()!=2)throw std::runtime_error("Grouped RLE version");m.end();Reader r{s.data.data(),s.data.size()};Words out;while(out.size()<s.count){auto n=var(r),v=word(r,b);if(!n||n>s.count-out.size())throw std::runtime_error("Grouped RLE run");out.insert(out.end(),n,v);}r.end();return out;}
 if(method==64||method==66)return unlifting(s,b,method==64);
 if(method==67){Reader m{s.model.data(),s.model.size()},p{s.data.data(),s.data.size()};if(m.get<uint32_t>()!=b)throw std::runtime_error("Grouped bitplane width");Words out(s.count,0);for(unsigned bit=b;bit--;){auto model=gblob(m);auto data=p.take(m.get<uint64_t>());auto plane=entropy_decode({s.count,model,data},8);for(size_t i=0;i<s.count;++i){if(plane[i]>1)throw std::runtime_error("Grouped bitplane value");out[i]|=plane[i]<<bit;}}m.end();p.end();return out;}
 Reader r{s.model.data(),s.model.size()};auto parameters=gblob(r),model=gblob(r);r.end();auto residuals=method==17?sz_load({s.count,model,s.data},b):entropy_decode({s.count,model,s.data},b);Words out(s.count);
 if(method==10||method==11){uint64_t previous=0;for(size_t i=0;i<s.count;++i){out[i]=method==10?(residuals[i]+previous)&mask(b):residuals[i]^previous;previous=out[i];}}
 else{auto p=wide_predictive::read_parameters(method,parameters,s.count,cols);for(size_t i=0;i<s.count;++i)out[i]=wrap(int64_t(residuals[i])+wide_predictive::predict(method,out,i,b,cols,p),b);}return out;
}
} // namespace grouped
