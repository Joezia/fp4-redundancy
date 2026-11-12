"""Whole-word domains, exact tails/extremes, independent decode and matched costs."""
import base64
from dataclasses import replace
import json
import os
import random
import struct
import subprocess
import sys
import unittest
from fp4_redundancy.catalog import builtin_registry
from fp4_redundancy.methods.grouped_units import ROOT,build_metadata,library,_call_units,NUMERIC
from fp4_redundancy.runner import evaluate_target
from fp4_redundancy.types import Component,Target,EncodedArtifact

def target(raw,high=False,rows=3,columns=13):
    return Target('odd_rows',(rows,columns),(Component('weight','packed_fp4','absent',0,len(raw),'fp4_packed_'+('high_first' if high else 'low_first')),),
        {'storage_shapes':{'weight':[rows,(columns+1)//2]},'fp4_value_format':'E2M1FN'})

def params(mid,b):
    p={'unit_bits':b}
    if mid=='vq_codeword':p.update(vector_length=4,centers=4,objective='rate',correction_model='auto')
    return p

class GroupedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.registry=builtin_registry();cls.audit=json.loads((ROOT/'experiments/grouped-units/audit.json').read_text());build_metadata()

    def test_all_supported_domains_extremes_repetition_odd_rows_and_tails(self):
        rng=random.Random(84);patterns=[bytes(21),bytes([255])*21,bytes([0,255,128,127,1,254,0])*3,bytes(rng.randrange(256) for _ in range(21))]
        checked=0
        for entry in self.audit:
            mid=entry['method_id']
            for b in entry['candidate_unit_bits']:
                if b==4:continue
                p=params(mid,b)
                for high in (False,True):
                    for raw in patterns:
                        t=target(raw,high);m=self.registry.create(mid)
                        self.assertTrue(m.check_support(t,p).applicable,(mid,b))
                        a=m.encode(t,{'weight':raw},p)
                        self.assertEqual(self.registry.create(mid).decode(a),{'weight':raw},(mid,b))
                        h=json.loads(a.side_information);g=h['components'][0]['grouping']
                        self.assertEqual(g['word_columns'],(7+b//8-1)//(b//8))
                        self.assertEqual(g['native_padding_nibbles_per_row'],1)
                        self.assertEqual(a.total_bytes,len(a.payload)+len(a.side_information));checked+=1
        self.assertGreater(checked,1000)

    def test_packing_order_row_isolation_and_all_uint64_bits(self):
        lib=library();raw=bytes.fromhex('1032547698badcfeff')*3
        for b in (8,16,32,64):
            packed=_call_units(lib.fp4_units_pack,b,0,9,raw,len(raw))
            row=((9+b//8-1)//(b//8))*(b//8)
            self.assertEqual(packed[:9],raw[:9]);self.assertEqual(packed[9:row],bytes(row-9))
            self.assertEqual(_call_units(lib.fp4_units_unpack,b,0,9,packed,len(packed)),raw)
            changed=bytearray(packed)
            if row>9:
                changed[9]=1
                with self.assertRaises(ValueError):_call_units(lib.fp4_units_unpack,b,0,9,bytes(changed),len(changed))
        raw=struct.pack('<QQQQQ',0,2**64-1,2**63,2**63-1,1);t=target(raw,rows=1,columns=80)
        for mid in ('arithmetic','delta','xor','fpc_like_fp4','rle'):
            a=self.registry.create(mid).encode(t,{'weight':raw},{'unit_bits':64})
            self.assertEqual(self.registry.create(mid).decode(a),{'weight':raw})

    def test_original_four_bit_artifact_unchanged_and_limits_are_explicit(self):
        raw=bytes.fromhex('1032ff8077')*3;t=target(raw,rows=3,columns=10)
        for entry in self.audit:
            mid=entry['method_id'];m=self.registry.create(mid);p=params(mid,4);original={k:v for k,v in p.items() if k!='unit_bits'}
            if mid=='inter_frame_reference':continue
            self.assertEqual(m.encode(t,{'weight':raw},p),m.encode(t,{'weight':raw},original),mid)
        for mid,b in (('adpcm_like_fp4',32),('jpeg_ls_like_fp4',32),('ppm',32),('pq_lossless',64),('block_dedup',8),('vcdiff',16)):
            self.assertFalse(self.registry.create(mid).check_support(t,params(mid,b)).applicable,(mid,b))
        broken=replace(t,structure={**t.structure,'chunk':{'ranges':{'weight':{'offset':1}}}})
        self.assertFalse(self.registry.create('jpeg_ls_like_fp4').check_support(broken,{'unit_bits':16}).applicable)

    def test_joint_symbol_not_nibble_and_sparse_rans_resource_status(self):
        # Equal nibble marginals, deterministic adjacent pairs: real 16-bit joint
        # frequencies have two symbols, not the original 16-nibble alphabet.
        raw=bytes.fromhex('10325476')*20;t=target(raw,rows=1,columns=len(raw)*2)
        m=self.registry.create('arithmetic');a=m.encode(t,{'weight':raw},{'unit_bits':16});part=json.loads(a.side_information)['components'][0]
        self.assertEqual(part['streams'][0]['count'],len(raw)//2)
        self.assertEqual(m.decode(a),{'weight':raw})
        raw=b''.join(struct.pack('<I',v) for v in range(65537));t=target(raw,rows=1,columns=len(raw)*2)
        result=evaluate_target(self.registry,'rans',t,{'unit_bits':32},components={'weight':raw})
        self.assertEqual(result['status'],'resource_limited');self.assertIsNone(result['total_bytes']);self.assertIn('65536',result['reason'])

    def test_matched_arithmetic_word_width_and_new_process_decode(self):
        raw=bytes(range(21));t=target(raw)
        result=evaluate_target(self.registry,'ppm',t,{'unit_bits':16},components={'weight':raw})
        self.assertEqual(result['status'],'verified');self.assertEqual(result['baseline_comparison']['parameters'],{'unit_bits':16})
        for mid in ('delta','fpc_like_fp4','vq_codeword','opq_lossless','ksvd_lossless','cp_lossless','mdct_lossless'):
            p=params(mid,32);a=self.registry.create(mid).encode(t,{'weight':raw},p)
            script="import sys,json;from fp4_redundancy.catalog import builtin_registry;from fp4_redundancy.types import EncodedArtifact;r=json.load(sys.stdin);print(builtin_registry().create(r['m']).decode(EncodedArtifact(bytes.fromhex(r['p']),bytes.fromhex(r['h'])))['weight'].hex())"
            q=subprocess.run([sys.executable,'-c',script],input=json.dumps({'m':mid,'p':a.payload.hex(),'h':a.side_information.hex()}),text=True,capture_output=True,env={**os.environ,'PYTHONPATH':str(ROOT/'src')})
            self.assertEqual(q.returncode,0,q.stderr);self.assertEqual(bytes.fromhex(q.stdout),raw)
            with self.assertRaises(ValueError):self.registry.create(mid).decode(EncodedArtifact(a.payload[:-1],a.side_information))

    def test_jpeg_high_entropy_packed_bytes_can_expand_past_library_estimate(self):
        rng=random.Random(1);raw=rng.randbytes(512*256);t=target(raw,rows=512,columns=512)
        m=self.registry.create('jpeg_ls_like_fp4');a=m.encode(t,{'weight':raw},{'unit_bits':8})
        self.assertGreater(len(a.payload),len(raw)+1024)
        self.assertEqual(self.registry.create('jpeg_ls_like_fp4').decode(a),{'weight':raw})
        h=json.loads(a.side_information);self.assertEqual(h['components'][0]['grouping']['inner']['engine'],'jpeg-buffer-v2')
