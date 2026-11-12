"""Run named word-width configurations through the existing sample runner."""
import argparse,base64,csv,json,math,os,sqlite3,struct,subprocess,sys,time,zlib
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fp4_redundancy.experiment import _atomic_json,_now
DIR=ROOT/'experiments/grouped-units';PLAN=DIR/'plan.json'

def make_plan():
    original=json.loads((ROOT/'experiments/fp4-module/plan.json').read_text())
    audit=json.loads((DIR/'audit.json').read_text());jobs=[];equivalent=[]
    for e in audit:
        mid=e['method_id']
        if mid=='inter_frame_reference':equivalent.append({'method':mid,'equivalent_to':'motion_compensation','reason':e['notes']});continue
        for b in e['candidate_unit_bits']:
            p={'unit_bits':b}
            if mid=='vq_codeword':p.update(original['method_parameters'][mid])
            jobs.append({'method':mid,'unit_bits':b,'parameters':p,'output':f'runs/{mid}/grouped_units/unit{b}',
                         'phase':1 if b<=16 else 2})
        if mid=='block_dedup':equivalent.append({'method':mid,'unit_bits':[8,16,32],'equivalent_to':'block_dedup/unit4','reason':e['notes']})
        if mid in ('vcdiff','cdc_dedup'):equivalent.append({'method':mid,'unit_bits':[16,32],'equivalent_to':mid+'/unit8','reason':e['notes']})
    jobs.append({'method':'arithmetic','unit_bits':64,'parameters':{'unit_bits':64},'output':'runs/arithmetic/grouped_units/unit64','phase':2,'purpose':'matched zero-order control only'})
    plan={**original,'kind':'fp4_grouped_word_width','created_utc':_now(),'jobs':jobs,'equivalent':equivalent,
          'runtime':{'parallel_configurations':4,'workers_per_configuration':4,'inner_threads':1},
          'order':'all 4/8/16 configurations first; supported 32/64 second',
          'grouping':'consecutive original nibbles, first codeword low bits, little endian words, within original rows; exact row-tail zero padding with retained original byte length',
          'numeric_domain':'unit4 retains old plugin input semantics; unit8/16/32 numerical adapters use exact unsigned integer float64, same normalization/shapes/ranks/Q20 serialization; correction in original nibbles',
          'selection_policy':'report each fixed configuration; no best-of selector or unbilled configuration choice',
          'payload_persisted':False}
    with PLAN.open('x') as f:json.dump(plan,f,ensure_ascii=False,indent=2);f.write('\n')
    print(f"{len(jobs)} configurations; same {original['sample_chunks']} chunks / six complete matrices")

class Reader:
    def __init__(self,b):self.b=b;self.p=0
    def get(self,f):v=struct.unpack_from('<'+f,self.b,self.p);self.p+=struct.calcsize('<'+f);return v[0] if len(v)==1 else v
    def blob(self,fmt='I'):n=self.get(fmt);v=self.b[self.p:self.p+n];self.p+=n;return v

def flows(header,mid,b):
    """Derive inner stream boundaries from the actual saved decoder metadata."""
    part=next(c for c in header['components'] if c['role']=='packed_fp4');streams=part['streams'];g=part.get('grouping',{})
    outer=sum(s['bytes'] for s in streams);answer={'encoded_payload':outer};engine=g.get('inner',{}).get('engine')
    if len(streams)!=1:return {f'stream{i}':s['bytes'] for i,s in enumerate(streams)}
    model=base64.b64decode(streams[0]['model']['table_b64']);codec=None
    if engine=='wide':codec=struct.unpack_from('<I',model,4)[0];model=model[40:]
    if mid=='fpc_like_fp4':
        prefix=(streams[0]['count']+1)//2;return {'prefix':prefix,'xor_residual':outer-prefix}
    if mid=='adpcm_like_fp4':
        r=Reader(model);r.get('III');p=Reader(r.blob());p.blob();n=p.get('Q');return {'ima_decisions':n,'exact_mod_correction':outer-n}
    if mid in ('vq_lossless','pq_lossless','opq_lossless','rvq_lossless','aq_lossless'):
        r=Reader(model)
        if engine!='wide':r=Reader(model[12:12+struct.unpack_from('<I',model,8)[0]])
        dim,k,nt,exponent,workers=r.get('IIIhH');stages=1 if mid=='vq_lossless' else 2;d=4 if mid in ('pq_lossless','opq_lossless') else 8
        r.p+=(256 if mid=='opq_lossless' else 0)+stages*k*d*4;r.blob();r.blob();n=r.get('Q');return {'indices':n,'exact_correction':outer-n}
    if mid=='vq_codeword':
        if engine!='wide':model=model[12:12+struct.unpack_from('<I',model,8)[0]]
        r=Reader(model);r.get('I');r.get('HHHBBII');r.blob();r.blob();n=r.get('Q');return {'indices':n,'exact_correction':outer-n}
    from fp4_redundancy.methods.grouped_units import NUMERIC
    if mid in NUMERIC or (mid in ('cdf53','haar_lifting','bitplane','bwt_mtf') and engine!='wide'):
        r=Reader(model);r.get('II');r.blob();n=r.get('I');sizes=[]
        for i in range(n):r.get('Q');r.blob();sizes.append(r.get('Q'))
        labels=['coefficients_or_factors','exact_correction'] if mid in NUMERIC else [f'transform_stream{i}' for i in range(n)]
        if sum(sizes)!=outer:raise ValueError('Inner D payload accounting differs')
        return dict(zip(labels,sizes))
    if mid=='bitplane' and engine=='wide':
        r=Reader(model);n=r.get('I');sizes={}
        for p in reversed(range(n)):r.blob('Q');sizes[f'plane{p}']=r.get('Q')
        if sum(sizes.values())!=outer:raise ValueError('Plane accounting differs')
        return sizes
    labels={'cdf53':'integer_coefficients','haar_lifting':'integer_coefficients','delta':'mod_residual','xor':'xor_residual',
            'sz_like_lossless':'huffman_zstd_residual','rle':'run_lengths_and_symbols','lz77':'match_tokens','lzss':'flags_literals_matches',
            'repair':'grammar_roots','sequitur':'grammar_roots','cdc_dedup':'literal_blocks_and_references','block_dedup':'literal_blocks_and_references',
            'vcdiff':'vcdiff_instructions','rlz':'reference_matches_and_literals'}
    return {labels.get(mid,'encoded_payload'):outer}

def settings(mid,b):
    """Readable fixed settings; decoder parameters remain in the paid artifacts."""
    from fp4_redundancy.methods.grouped_units import NUMERIC
    v={'codewords_per_word':b//4,'input_domain':'original_plugin_semantics' if b==4 else 'unsigned_packed_word'}
    if mid in ('ppm','ctw'):v.update(context_order_words=3,history_fp4_codewords=3*b//4)
    if mid=='neural_entropy':v.update(history_words=4,history_fp4_codewords=b,network='8-8-1',parameter_precision='Q12')
    if mid=='paq':v.update(experts='order0/order1 whole-word prefix',previous_word_span=b//4)
    if mid=='cabac':v.update(context='previous word/current prefix; non-complete CABAC',previous_word_span=b//4)
    if mid=='fpc_like_fp4':v.update(table_entries_each=4096,prefix_bits_per_word=4,format_version=2 if b>=32 else 1)
    if mid=='rle':v.update(symbol_storage='original varint' if b==4 else f'{b}-bit unsigned little endian',run='LEB128')
    if mid=='bwt_mtf':v.update(internal_block_words=65536,block_fp4_codewords=65536*b//4,initial_mtf_alphabet=2**b)
    if mid in ('cdf53','haar_lifting'):v.update(block_words=256,block_fp4_codewords=256*b//4,intermediate='int64' if b==32 else 'int32')
    if mid in ('lz77','lzss'):v.update(window_words=(65535 if mid=='lz77' else 4096)*4//b,min_match_words=3,min_match_codewords=3*b//4)
    if mid=='jpeg_ls_like_fp4':v.update(near=0,sample_precision=b,destination='estimated size with bounded retry' if b>4 else 'original estimated capacity')
    if mid=='motion_compensation':v.update(block_words=[4,16],frame_rows=16,block_fp4_codewords=64*b//4)
    if mid=='intra_directional':v.update(block_words=[8,8],block_fp4_codewords=64*b//4)
    if mid in NUMERIC:
        shapes={'cp_lossless':([4,4,4],[2]),'tucker_lossless':([4,4,4],[2,2,2]),'tt_lossless':([4,4,4],[1,2,2,1]),
                'svd_lowrank':([8,8],[2]),'kronecker_lossless':([8,8,2,2,4,4],[1]),'circulant_lossless':([8,8],[8]),
                'sparse_coding':([8,16],[3]),'ksvd_lossless':([8,16],[3]),'mdct_lossless':([16,8],[8])}
        shape,ranks=shapes.get(mid,([8],[8]));words=64 if mid in ('cp_lossless','tucker_lossless','tt_lossless','svd_lowrank','kronecker_lossless','circulant_lossless') else 8
        v.update(shape_words=shape,ranks=ranks,words_per_block_or_step=words,fp4_codewords_per_block_or_step=words*b//4,
                 input_precision='float64',serialized_parameters='Q20 int32',correction_unit_bits=4 if b>4 else 'original complete codeword',
                 reconstruction='stored Q20 integer path; round positive halves upward, clamp to unsigned width; exact XOR')
    if mid=='vq_codeword':v.update(vector_words=4,vector_fp4_codewords=b,max_centers=4,objective='rate',representative='observed medoid',correction_unit_bits=4 if b>=16 else b)
    if mid=='mdct_lossless':v.update(window_words=16,hop_words=8,window_fp4_codewords=4*b)
    return v

def report(plan):
    rows=[];details=[];matrices=[];reference=None
    for job in plan['jobs']:
        out=ROOT/job['output'];path=out/'results.sqlite3'
        if not path.exists():continue
        mid,b=job['method'],job['unit_bits'];row={'method':mid,'unit_bits':b,'parameters':json.dumps(job['parameters'],sort_keys=True),'algorithm_settings':settings(mid,b),'status':'running',
             'chunks':0,'verified_chunks':0,'fp4_original_bytes':plan['sample_weight_bytes'],'fp4_payload_bytes':0,'fp4_metadata_bytes':0,'fp4_total_bytes':None,'fp4_payload_saving_fraction':None,'fp4_saving_fraction':None,'net_saving_fraction':None,
             'matched_baseline_unit_bits':b,'whole_target_total_bytes':0,'streams_bytes':{},'scope':'six_complete_adjacent_matrices','reasons':[]}
        groups={};hashes={};statuses={}
        with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as db:
            for raw,state in db.execute('SELECT result_json,state_zlib FROM chunks'):
                record=json.loads(raw);r=record['results'][0];row['chunks']+=1;status=r['status'];statuses[status]=statuses.get(status,0)+1
                if r.get('source_sha256') is not None:hashes[record['chunk_id']]=r['source_sha256']
                if status!='verified':
                    if r['reason'] not in row['reasons']:row['reasons'].append(r['reason'])
                    continue
                row['verified_chunks']+=1;part=next(c for c in r['components'] if c['role']=='packed_fp4');meta=part['direct_side_information_bytes']+r['shared_side_information_bytes']-(len(r['components'])-1)
                row['fp4_payload_bytes']+=part['payload_bytes'];row['fp4_metadata_bytes']+=meta;row['whole_target_total_bytes']+=r['total_bytes'];hashes[record['chunk_id']]=r['source_sha256']
                state=json.loads(zlib.decompress(state));h=state[mid]['side_information'];sub=flows(h,mid,b)
                if sum(sub.values())!=part['payload_bytes']:raise ValueError('Inner payload total differs')
                for k,v in sub.items():row['streams_bytes'][k]=row['streams_bytes'].get(k,0)+v
                g=next(c for c in h['components'] if c['role']=='packed_fp4').get('grouping')
                details.append({'method':mid,'unit_bits':b,'target':record['target_id'],'chunk_id':record['chunk_id'],'raw_chunk_bytes_limit':plan['chunk_bytes'],
                                'fp4_original_bytes':part['original_bytes'],'stored_nibbles':part['original_bytes']*2,
                                'logical_fp4_weights':math.prod(h['logical_shape']),'word_count':g['rows']*g['word_columns'] if g else part['original_bytes']*2,
                                'grouping':g,
                                'streams_bytes':sub,'fp4_payload_bytes':part['payload_bytes'],'fp4_metadata_bytes':meta,'fp4_total_bytes':part['payload_bytes']+meta,
                                'fp4_payload_saving_fraction':1-part['payload_bytes']/part['original_bytes'],
                                'encode_seconds':r['encode_seconds'],'decode_seconds':r['decode_seconds'],'verified':True})
                key=record['target_id'];v=groups.setdefault(key,{'method':mid,'unit_bits':b,'target':key,'chunks':0,'original':0,'payload':0,'metadata':0})
                v['chunks']+=1;v['original']+=part['original_bytes'];v['payload']+=part['payload_bytes'];v['metadata']+=meta
        row['statuses']=statuses
        resources=out.parent/(out.name+'-resources.txt');row['wall_seconds']=None;row['peak_process_rss_kib']=None
        if not resources.exists():resources=out/'resources.txt'
        if resources.exists():
            for line in resources.read_text().splitlines():
                if 'Maximum resident set size (kbytes):' in line:row['peak_process_rss_kib']=int(line.rsplit(':',1)[1])
                if 'Elapsed (wall clock) time' in line:
                    parts=line.split('): ',1)[1].strip().split(':');seconds=0
                    for p in parts:seconds=seconds*60+float(p)
                    row['wall_seconds']=seconds
            row['resource_log']=str(resources.relative_to(ROOT))
        execution=json.loads((DIR/'execution.json').read_text()) if (DIR/'execution.json').exists() else {}
        if row['wall_seconds'] is None:row['wall_seconds']=execution.get(f'{mid}/unit{b}',{}).get('wall_seconds')
        if row['chunks']==plan['sample_chunks']:
            row['status']='verified' if row['verified_chunks']==row['chunks'] else ','.join(sorted(k for k in statuses if k!='verified'))
            if len(hashes)==plan['sample_chunks']:
                if reference is None:reference=hashes
                elif hashes!=reference:raise ValueError('Configuration source hashes differ')
        if row['status']=='verified':
            if sum(v['original'] for v in groups.values())!=plan['sample_weight_bytes']:raise ValueError('Byte coverage differs')
            if reference is None:reference=hashes
            elif hashes!=reference:raise ValueError('Method source hashes differ')
            row['fp4_total_bytes']=row['fp4_payload_bytes']+row['fp4_metadata_bytes'];row['fp4_saving_fraction']=1-row['fp4_total_bytes']/row['fp4_original_bytes']
            row['fp4_payload_saving_fraction']=1-row['fp4_payload_bytes']/row['fp4_original_bytes']
        if not row['verified_chunks']:
            row['fp4_payload_bytes']=None;row['fp4_metadata_bytes']=None;row['whole_target_total_bytes']=None
        for g in groups.values():g['fp4_payload_saving_fraction']=1-g['payload']/g['original']
        rows.append(row);matrices.extend(groups.values())
    baseline={r['unit_bits']:r for r in rows if r['method']=='arithmetic' and r['status']=='verified'}
    for r in rows:
        if r['status']=='verified' and r['method']!='arithmetic' and r['unit_bits'] in baseline:r['net_saving_fraction']=1-r['fp4_total_bytes']/baseline[r['unit_bits']]['fp4_total_bytes']
    _atomic_json(DIR/'results.json',rows);_atomic_json(DIR/'chunk-details.json',details);_atomic_json(DIR/'matrix-results.json',matrices)
    if rows:
        with (DIR/'results.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
    lines=['# 连续 FP4 分组实验','',f"已完成 {sum(r['chunks']==plan['sample_chunks'] for r in rows)}/{len(plan['jobs'])} 个配置；每配置同一组6个完整矩阵、36块、{plan['sample_weight_bytes']} FP4 bytes。",
           'Payload节省率 = 1 − payload / 原始packed字节；总节省率 = 1 − (payload + 元数据) / 原始packed字节。负值为膨胀。相对算术净节省仍按完整总成本比较。scale独立处理，表中仅FP4。',
           'Payload保留全部实际编码流：熵编码位流、ANS重归一化输出及最终编码状态、FPC前缀、残差/纠错、索引、系数/因子等；频率表、符号映射、码本、模型参数和布局等按现有独立元数据计量。完整分类见[开发说明](../../docs/开发说明.md#payload与元数据)。',
           '不重新划分现有产物的字节边界。元数据并非全部可共享；字典、语法或参考可能携带原始信息，Payload节省率仅用于诊断，不能单独作为发现冗余的证据。',
           '逐配置展示，不生成未计费的“每目标最佳配置”组合；unit4保留原方法语义，数值法unit8起为packed整数域。']
    for title,field in (('Payload节省率（不含独立元数据）','fp4_payload_saving_fraction'),('总节省率（含全部元数据）','fp4_saving_fraction')):
        lines+=['','## '+title,'','| 方法 | 4 bit | 8 bit | 16 bit | 32 bit | 64 bit |','| --- | ---: | ---: | ---: | ---: | ---: |']
        for mid in sorted({r['method'] for r in rows}):
            cells=[]
            for b in (4,8,16,32,64):
                found=next((r for r in rows if r['method']==mid and r['unit_bits']==b),None)
                cells.append('—' if found is None else f"{found[field]:.2%}" if found['status']=='verified' else found['status'])
            lines.append('| '+' | '.join([mid,*cells])+' |')
    lines+=['','— 表示本轮范围外或等价配置，具体见支持清单；不表示零收益。','',
            '## 逐配置成本与收益','',
            '| 方法 | unit_bits | payload bytes | Payload节省率 | 元数据 bytes | 合计 bytes | 总节省率 | 相对同宽算术净节省 | 验证/限制 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in rows:
        show=lambda v:'—' if v is None else f'{v:.2%}'
        lines.append(f"| {r['method']} | {r['unit_bits']} | {r['fp4_payload_bytes'] if r['verified_chunks'] else '—'} | {show(r['fp4_payload_saving_fraction'])} | {r['fp4_metadata_bytes'] if r['verified_chunks'] else '—'} | {r['fp4_total_bytes'] if r['fp4_total_bytes'] is not None else '—'} | {show(r['fp4_saving_fraction'])} | {show(r['net_saving_fraction'])} | {r['status']} |")
    lines+=['','支持范围及等价项见[audit.md](audit.md)。[results.csv](results.csv)保存参数、状态与各流汇总；`chunk-details.json`是服务器本地块级诊断，保存实际word行宽、覆盖码字数、各流大小、编解码时间，不随Git上传；[matrix-results.json](matrix-results.json)保留逐矩阵结果。','',
            '具体资源限制：']
    for r in rows:
        if r['reasons']:lines.append(f"- {r['method']}/{r['unit_bits']}: {'; '.join(r['reasons'])}")
    for attempt in plan.get('retained_attempts',[]):lines.append(f"- 已修复并重跑 {attempt['method']}/{attempt['unit_bits']}：{attempt['reason']}；首次失败记录保留于 `{attempt['output']}`，当前结果使用 `-buffer-v2`。")
    (DIR/'report.md').write_text('\n'.join(lines)+'\n')
    _atomic_json(DIR/'validation.json',{'configurations':len(rows),'completed_configurations':sum(r['chunks']==plan['sample_chunks'] for r in rows),
                 'verified_configurations':sum(r['status']=='verified' for r in rows),'verified_chunk_pairs':sum(r['verified_chunks'] for r in rows),'same_source_hashes_on_complete_verified_configurations':True,
                 'same_source_hashes_including_resource_limited':True,
                 'fp4_bytes_per_configuration':plan['sample_weight_bytes'],'matrices_per_configuration':6,'full_model_run':False})

def run(plan):
    execution=json.loads((DIR/'execution.json').read_text()) if (DIR/'execution.json').exists() else {}
    env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'FP4_C_THREADS':'1','FP4_D_THREADS':'1','FP4_VQ_THREADS':'1','OMP_NUM_THREADS':'1'}
    def one(job):
        mid,b=job['method'],job['unit_bits'];key=f'{mid}/unit{b}';out=ROOT/job['output'];out.parent.mkdir(parents=True,exist_ok=True)
        if (out/'summary.json').exists():return key,execution.get(key,{'status':'existing_completed'})
        cmd=[sys.executable,'-m','fp4_redundancy','run',plan['snapshot'],'--output',str(out),'--methods',mid,'--chunk-bytes',str(plan['chunk_bytes']),
             '--selection',str(PLAN),'--workers',str(plan['runtime']['workers_per_configuration']),'--method-parameters',json.dumps({mid:job['parameters']})]
        if (out/'run.json').exists():cmd.append('--resume')
        start=time.monotonic();print(json.dumps({'event':'start','configuration':key,'utc':_now()}),flush=True)
        with (out.parent/(out.name+'-execution.log')).open('a') as log:
            q=subprocess.run(['/usr/bin/time','-v','-o',str(out.parent/(out.name+'-resources.txt')),*cmd],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        result={'exit_code':q.returncode,'wall_seconds':time.monotonic()-start,'finished_utc':_now(),'resources':str(out.parent/(out.name+'-resources.txt'))}
        print(json.dumps({'event':'finished','configuration':key,**result}),flush=True);return key,result
    for phase in (1,2):
        jobs=[j for j in plan['jobs'] if j['phase']==phase]
        priority={'ctw':0,'repair':1,'vq_codeword':2,'aq_lossless':3,'cp_lossless':4}
        jobs.sort(key=lambda j:(priority.get(j['method'],10),j['unit_bits'],j['method']))
        with ThreadPoolExecutor(max_workers=plan['runtime']['parallel_configurations']) as pool:
            for f in as_completed([pool.submit(one,j) for j in jobs]):
                key,result=f.result();execution[key]=result;_atomic_json(DIR/'execution.json',execution)
        report(plan)
        if any(not (ROOT/j['output']/'summary.json').exists() for j in jobs):raise RuntimeError('Interrupted configurations; inspect execution logs before the next phase')
    report(plan)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('plan','run','report'));a=p.parse_args()
    if a.command=='plan':make_plan()
    else:(run if a.command=='run' else report)(json.loads(PLAN.read_text()))
