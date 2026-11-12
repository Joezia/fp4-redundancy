"""Private word-input configurations of complete plugins; no stage composition API."""
import base64
import ctypes
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import struct
import zlib
from ..types import EncodedArtifact, Support, ResourceLimitError
from .native import ROOT, _buffer_library, _call, _json

NUMERIC=frozenset(('vq_lossless','pq_lossless','opq_lossless','rvq_lossless','aq_lossless',
 'dct_lossless','dst_lossless','mdct_lossless','pca_lossless','sparse_coding','ksvd_lossless',
 'svd_lowrank','cp_lossless','tucker_lossless','tt_lossless','kronecker_lossless','circulant_lossless'))
WIDE=frozenset(('huffman','arithmetic','rans','golomb_rice','delta','xor','dpcm_lossless','png_filtering',
 'fpc_like_fp4','sz_like_lossless','lpc_lossless','ltp_lossless','motion_compensation','intra_directional',
 'rle','cdf53','haar_lifting','bitplane'))

@lru_cache(maxsize=1)
def library():
    lib=_buffer_library(ROOT/'build/libfp4_grouped_units.so','fp4_units')
    for name in ('pack','unpack','vq_encode'):
        getattr(lib,'fp4_units_'+name).argtypes=lib.fp4_native_encode.argtypes
    return lib

@lru_cache(maxsize=1)
def jpeg_library():return _buffer_library(ROOT/'build/libfp4_grouped_jpeg.so','fp4_jpeg')

def jpeg_build_metadata():
    m=json.loads((ROOT/'build/grouped-jpeg-build.json').read_text())
    for p,h in m['dependencies_sha256'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise RuntimeError('Grouped JPEG dependency changed')
    if hashlib.sha256((ROOT/'build/libfp4_grouped_jpeg.so').read_bytes()).hexdigest()!=m['library_sha256']:raise RuntimeError('Grouped JPEG binary changed')
    return m

def _jpeg_method():
    from .predictive import NativeB
    codec=NativeB('jpeg_ls_like_fp4');codec._library=jpeg_library;return codec

def build_metadata():
    m=json.loads((ROOT/'build/grouped-units-build.json').read_text());source=ROOT/'native/grouped_units';h=hashlib.sha256()
    for p in sorted(p for p in source.rglob('*') if p.is_file()):h.update(str(p.relative_to(source)).encode());h.update(p.read_bytes())
    if h.hexdigest()!=m['source_sha256']:raise RuntimeError('Grouped sources changed; run scripts/build_grouped_units.py')
    for p,key in (('build/libfp4_grouped_units.so','library_sha256'),('build/libfp4_entropy.so','a1_library_sha256')):
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=m[key]:raise RuntimeError('Grouped binary dependency changed')
    for p,digest in m['dependencies_sha256'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=digest:raise RuntimeError('Grouped source dependency changed')
    return m

def bits(parameters):
    b=parameters.get('unit_bits',4)
    if type(b) is not int or b not in (4,8,16,32,64):raise ValueError('unit_bits must be 4/8/16/32/64')
    return b

def support(method,target,parameters):
    try:b=bits(parameters)
    except ValueError as e:return Support(False,str(e))
    p={k:v for k,v in parameters.items() if k!='unit_bits'}
    if b==4:return method.check_support(target,p)
    mid=method.spec.id
    if method.exponent_split:return Support(False,'Grouped FP4 does not combine with exponent separation')
    if not any(c.role=='packed_fp4' for c in target.components):return Support(False,'unit_bits applies only to packed FP4 components')
    if mid=='block_dedup':return Support(False,'EQUIVALENT: same fixed original-byte boundaries and packed literals preserve matches and size')
    if mid=='inter_frame_reference':return Support(False,'EQUIVALENT: use motion_compensation with the same unit_bits')
    if mid in ('vcdiff','cdc_dedup') and b>8:return Support(False,'EQUIVALENT: the true packed-byte configuration is unit_bits=8')
    if b==64 and mid not in ('delta','xor','fpc_like_fp4','rle','arithmetic'):return Support(False,'64-bit first round is limited to exact bit/integer methods and their matched arithmetic control')
    if b==32 and mid not in WIDE|NUMERIC|{'vq_codeword'}:return Support(False,'This implementation is bounded to 16-bit symbols; 32-bit is outside its first-round capability')
    # Validation of method-specific parameters is retained. A categorical packed
    # word has no IEEE format and does not need inferred E2M1 semantics.
    mapped=replace(target,components=tuple(replace(c,role='other',storage_format='F8_E4M3') if c.role=='packed_fp4' else c for c in target.components))
    result=method.check_support(mapped,p)
    if not result.applicable:return result
    try:
        for c in target.components:
            if c.role=='packed_fp4':_geometry(target,c,b)
    except ValueError as e:return Support(False,str(e))
    return Support(True,'Complete packed-word plugin configuration; scales retain original independent representation')

def _call_units(function,*args):
    lib=library()
    try:return _call(function,*args,_library=lib)
    except ValueError as e:
        if 'RESOURCE_LIMIT:' in str(e):raise ResourceLimitError(str(e)) from e
        raise

def _geometry(target,component,b):
    length=component.length
    shape=target.structure.get('storage_shapes',{}).get(component.id)
    if shape and len(shape)==2 and shape[1]>0 and length%shape[1]==0:row_bytes=shape[1]
    elif len(target.logical_shape)==2 and length==target.logical_shape[0]*((target.logical_shape[1]+1)//2):row_bytes=(target.logical_shape[1]+1)//2
    else:row_bytes=length
    relative=target.structure.get('chunk',{}).get('ranges',{}).get(component.id,{}).get('offset',0)
    if shape and len(shape)==2 and (length%shape[1] or relative%shape[1]):
        raise ValueError('Grouped 2D input must contain complete original rows')
    rows=length//row_bytes;word_bytes=b//8;grouped_row_bytes=((row_bytes+word_bytes-1)//word_bytes)*word_bytes
    valid=target.logical_shape[-1] if len(target.logical_shape)==2 and (target.logical_shape[-1]+1)//2==row_bytes else row_bytes*2
    return {'unit_bits':b,'raw_row_bytes':row_bytes,'rows':rows,'word_columns':grouped_row_bytes//word_bytes,
            'grouped_bytes':rows*grouped_row_bytes,'nibble_order':component.storage_format,
            'word_order':'first_codeword_in_low_nibble','word_endian':'little',
            'valid_codewords_per_row':valid,'stored_nibbles_per_row':row_bytes*2,
            'native_padding_nibbles_per_row':row_bytes*2-valid,'zero_pad_bytes_per_row':grouped_row_bytes-row_bytes}

def encode(method,target,components,parameters):
    b=bits(parameters);p={k:v for k,v in parameters.items() if k!='unit_bits'}
    if b==4:return method.encode(target,components,p)
    accepted=support(method,target,parameters)
    if not accepted.applicable:raise ValueError(accepted.reason)
    if set(components)!={c.id for c in target.components}:raise ValueError('Grouped component set differs')
    payload=bytearray();parts=[];lib=library()
    for component in target.components:
        raw=components[component.id]
        if len(raw)!=component.length:raise ValueError('Grouped input length differs')
        if component.role=='packed_fp4':
            g=_geometry(target,component,b)
            packed=_call_units(lib.fp4_units_pack,b,int(component.storage_format.endswith('high_first')),g['raw_row_bytes'],raw,len(raw))
            if len(packed)!=g['grouped_bytes']:raise ValueError('Grouped packing length differs')
            native=(b>16) or method.spec.id in NUMERIC|{'rle','lz77','lzss','cdc_dedup'}
            native=native or (method.spec.id=='vq_codeword' and b==16)
            if native:
                if method.spec.id=='vq_codeword':
                    from .codeword_vq import options
                    o=options(p);flags=o['vector_length']|(o['centers']<<8)|(('shared','conditioned','auto').index(o['correction_model'])<<16)|((o['objective']=='rate')<<18)
                    frame=_call_units(lib.fp4_units_vq_encode,b,flags,g['word_columns'],packed,len(packed))
                else:frame=_call_units(lib.fp4_native_encode,method.codec,b,g['word_columns'],packed,len(packed))
                magic,codec,width,cols,count,nm,nd=struct.unpack_from('<IIIIQQQ',frame)
                if magic!=0x31505747 or codec!=method.codec or width!=b or len(frame)!=40+nm+nd:raise ValueError('Grouped native frame differs')
                streams=[{'count':count,'bytes':nd,'model':{'backend':'grouped-native-v1','table_b64':base64.b64encode(frame[:40+nm]).decode(),'payload_bits':nd*8}}]
                data=frame[40+nm:];inner={'engine':'wide'}
            else:
                fmt='F8_E4M3' if b==8 else 'BF16'
                part=replace(component,role='other',storage_format=fmt,length=len(packed))
                mapped=replace(target,logical_shape=(g['rows'],g['word_columns']),components=(part,),
                    structure={**target.structure,'storage_shapes':{component.id:[g['rows'],g['word_columns']]}})
                codec=_jpeg_method() if method.spec.id=='jpeg_ls_like_fp4' else method
                try:artifact=codec.encode(mapped,{component.id:packed},p)
                except ValueError as e:
                    if 'RESOURCE_LIMIT:' in str(e):raise ResourceLimitError(str(e)) from e
                    raise
                header=json.loads(artifact.side_information);source=header['components'][0]
                streams=source['streams'];data=artifact.payload
                inner={k:v for k,v in source.items() if k not in ('id','streams')};inner['engine']='legacy'
                if method.spec.id=='jpeg_ls_like_fp4':inner['engine']='jpeg-buffer-v2'
            g['inner']=inner
        else:
            artifact=method.encode(replace(target,components=(component,)),{component.id:raw},p)
            header=json.loads(artifact.side_information);source=header['components'][0];streams=source['streams'];data=artifact.payload
            g={'unit_bits':8,'inner':{k:v for k,v in source.items() if k not in ('id','streams')}};g['inner']['engine']='legacy'
        parts.append({'id':component.id,'role':component.role,'format':component.storage_format,'bytes':len(raw),
            'crc32':zlib.crc32(raw),'split':False,'grouping':g,'streams':streams})
        payload.extend(data)
    header={'version':1,'method':method.spec.id,'backend':method.backend+'-grouped-v1','unit_bits':b,
            'target_id':target.id,'logical_shape':target.logical_shape,'components':parts}
    for key in ('chunk','tensor_names','storage_shapes','scale_block_shape'):
        if key in target.structure:header[key]=target.structure[key]
    return EncodedArtifact(bytes(payload),_json(header))

def decode(method,artifact):
    h=json.loads(artifact.side_information)
    if h['version']!=1 or h['method']!=method.spec.id or h['backend']!=method.backend+'-grouped-v1':raise ValueError('Grouped method/version differs')
    bits({'unit_bits':h['unit_bits']});offset=0;out={};lib=library()
    for part in h['components']:
        n=sum(s['bytes'] for s in part['streams'])
        if n<0 or offset+n>len(artifact.payload) or part['id'] in out:raise ValueError('Grouped payload layout')
        data=artifact.payload[offset:offset+n];offset+=n;g=part['grouping'];inner=dict(g['inner']);engine=inner.pop('engine')
        if engine=='wide':
            prefix=base64.b64decode(part['streams'][0]['model']['table_b64'],validate=True)
            if struct.unpack_from('<III',prefix)!=(0x31505747,method.codec,g['unit_bits']):raise ValueError('Grouped native method differs')
            frame=prefix+data;raw=_call_units(lib.fp4_native_decode,frame,len(frame))
        elif engine in ('legacy','jpeg-buffer-v2'):
            inner.update(id=part['id'],streams=part['streams'])
            header={'version':1,'method':method.spec.id,'backend':method.backend,'components':[inner]}
            codec=_jpeg_method() if engine=='jpeg-buffer-v2' else method
            raw=codec.decode(EncodedArtifact(data,_json(header)))[part['id']]
        else:raise ValueError('Unknown grouped engine')
        if part['role']=='packed_fp4':
            if len(raw)!=g['grouped_bytes']:raise ValueError('Grouped decoded word bytes differ')
            raw=_call_units(lib.fp4_units_unpack,g['unit_bits'],int(part['format'].endswith('high_first')),g['raw_row_bytes'],raw,len(raw))
        if len(raw)!=part['bytes'] or zlib.crc32(raw)!=part['crc32']:raise ValueError('Grouped original bytes checksum differs')
        out[part['id']]=raw
    if offset!=len(artifact.payload):raise ValueError('Grouped trailing payload')
    return out
