"""B mechanisms, causal state, independent decoding, and existing result plumbing."""

import base64
import ctypes
import json
import os
import random
import sqlite3
import struct
import subprocess
import sys
import unittest
import zlib

import test_checkpoint
from fp4_redundancy.catalog import builtin_registry
from fp4_redundancy.experiment import run_experiment
from fp4_redundancy.methods.native import ROOT, MAGIC, NativeA, _call, library as a_library
from fp4_redundancy.methods.predictive import METHODS, NativeB, build_metadata, component_columns
from fp4_redundancy.types import Target, Component, EncodedArtifact


def target(raw, fmt="BF16", columns=None):
    bits = 4 if fmt.startswith("fp4") else 16 if fmt == "BF16" else 8
    shape = (len(raw)*8//bits,)
    structure = {}
    if columns:
        shape = (shape[0]//columns, columns)
        structure = {"storage_shapes": {"weight": [shape[0], columns//2 if bits==4 else columns]}}
    return Target("synthetic",shape,(Component("weight","packed_fp4" if bits==4 else "other",
                  "missing",0,len(raw),fmt),),structure)


def model(artifact):
    stream = json.loads(artifact.side_information)["components"][0]["streams"][0]
    m = base64.b64decode(stream["model"]["table_b64"])
    version, alphabet, columns, np = struct.unpack_from("<IIII",m)
    pos = 16 + np
    nm, = struct.unpack_from("<I",m,pos)
    assert pos + 4 + nm + 8 == len(m)
    return version, alphabet, columns, m[16:pos], m[pos+4:pos+4+nm], stream["count"]


def ac_words(payload, table, count):
    frame = struct.pack("<IQIIIIQQQ",MAGIC,count*2,1,5,0,1,count,len(table),len(payload))+table+payload
    raw = _call(a_library().fp4_native_decode,frame,len(frame))
    return list(struct.unpack("<"+"H"*count,raw))


@unittest.skipUnless((ROOT/"build/libfp4_predictive.so").is_file(),"Build B codecs first")
class PredictiveTests(unittest.TestCase):
    def test_residuals_use_exact_a1_arithmetic_and_wrap(self):
        values = [65535,0,32768,0,1,65535,65535,32767]
        raw = struct.pack("<8H",*values)
        for method in ("delta","xor","dpcm_lossless"):
            a = NativeB(method).encode(target(raw),{"weight":raw},{})
            _,alphabet,_,params,table,n = model(a)
            expected=[]
            for i,v in enumerate(values):
                prev=values[i-1] if i else 0
                pred=2*prev-(values[i-2] if i>=2 else 0) if method=="dpcm_lossless" else prev
                expected.append(v^pred if method=="xor" else (v-pred)&65535)
            self.assertEqual(ac_words(a.payload,table,n),expected)
            self.assertEqual(alphabet,65536)
            self.assertEqual(params,b"")
        ramp=bytes(range(256))*16
        a=NativeB("delta").encode(target(ramp,"F8_E4M3"),{"weight":ramp},{})
        b=NativeA("arithmetic").encode(target(ramp,"F8_E4M3"),{"weight":ramp},{})
        self.assertLess(a.total_bytes,b.total_bytes)

    def test_adpcm_retains_quantized_feedback_and_pays_exact_compensation(self):
        values=[32768,32769,40000,0,65535,32768]*17
        raw=struct.pack("<"+"H"*len(values),*values)
        a=NativeB("adpcm_like_fp4").encode(target(raw),{"weight":raw},{})
        _,_,_,params,correction_table,n=model(a)
        length,=struct.unpack_from("<I",params)
        qtable=params[4:4+length]
        qbytes,=struct.unpack_from("<Q",params,4+length)
        codes=ac_words(a.payload[:qbytes],qtable,n)
        corrections=ac_words(a.payload[qbytes:],correction_table,n)
        # Independent IMA decoder; step/index state uses quantized reconstruction only.
        steps=[7,8,9,10,11,12,13,14,16,17,19,21,23,25,28,31,34,37,41,45,50,55,60,66,73,80,88,97,107,118,
               130,143,157,173,190,209,230,253,279,307,337,371,408,449,494,544,598,658,724,796,876,963,
               1060,1166,1282,1411,1552,1707,1878,2066,2272,2499,2749,3024,3327,3660,4026,4428,4871,5358,
               5894,6484,7132,7845,8630,9493,10442,11487,12635,13899,15289,16818,18500,20350,22385,24623,27086,29794,32767]
        index=value=0
        for original,code,residue in zip(values,codes,corrections):
            step=steps[index]
            diff=(step>>3)+((step>>2) if code&1 else 0)+((step>>1) if code&2 else 0)+(step if code&4 else 0)
            value=max(-32768,min(32767,value+(-diff if code&8 else diff)))
            index=max(0,min(88,index+[-1,-1,-1,-1,2,4,6,8][code&7]))
            self.assertEqual((residue+value+32768)&65535,original)
        self.assertTrue(any(corrections))
        self.assertGreater(len(qtable),0)
        self.assertEqual(NativeB("adpcm_like_fp4").decode(a),{"weight":raw})

    def test_fpc_reference_vector_and_jpegls_native_stream(self):
        raw=b"\x11"
        a=NativeB("fpc_like_fp4").encode(target(raw,"fp4_packed_low_first"),{"weight":raw},{})
        # First 1 uses FCM=0, one residual byte; second 1 uses DFCM=1, zero bytes.
        self.assertEqual(a.payload,b"\x18\x01")
        self.assertEqual(NativeB("fpc_like_fp4").decode(a),{"weight":raw})
        raw=bytes(range(256))*2
        a=NativeB("jpeg_ls_like_fp4").encode(target(raw,"F8_E4M3",16),{"weight":raw},{})
        self.assertEqual(a.payload[4:6],b"\xff\xd8")
        self.assertIn(b"\xff\xf7",a.payload)  # JPEG-LS SOF55, not arithmetic payload.
        self.assertEqual(a.payload[-2:],b"\xff\xd9")

    def test_causal_spatial_modes_parameters_and_all_special_patterns(self):
        rng=random.Random(117)
        # Matching strips with both horizontal and vertical displacement, plus partial edge blocks.
        rows,cols=35,37
        raw=bytes(rng.randrange(256) for _ in range(rows*cols))
        for method in METHODS:
            codec=NativeB(method);t=target(raw,"F8_E4M3",cols)
            a=codec.encode(t,{"weight":raw},{})
            self.assertEqual(NativeB(method).decode(a),{"weight":raw})
            _,alphabet,width,params,table,_=model(a)
            self.assertEqual((alphabet,width),(256,cols))
            if method=="lpc_lossless":self.assertEqual(len(params),16)
            if method=="ltp_lossless":self.assertEqual(len(params),4*((len(raw)+255)//256))
            if method=="motion_compensation":self.assertEqual(len(params),2*((rows+3)//4)*((cols+15)//16))
            if method=="intra_directional":self.assertEqual(len(params),((rows+7)//8)*((cols+7)//8))
            if method=="png_filtering":self.assertEqual(len(params),rows)
            if method=="sz_like_lossless":self.assertEqual(a.payload[8:12],bytes.fromhex("28b52ffd"))
            for payload in (a.payload[:-1],a.payload+b"extra",bytes([a.payload[0]^255])+a.payload[1:]):
                with self.assertRaises(ValueError,msg=method):codec.decode(EncodedArtifact(payload,a.side_information))
        # A true 2-D displacement is selected, rather than an alias of Delta.
        previous=bytes(rng.randrange(256) for _ in range(16*cols))
        shifted=bytes(previous[((y+4)%16)*cols+(x+4)%cols] for y in range(16) for x in range(cols))
        raw=previous+shifted
        a=NativeB("motion_compensation").encode(target(raw,"F8_E4M3",cols),{"weight":raw},{})
        params=model(a)[3]
        self.assertEqual(params[4*((cols+15)//16)*2:][:2],bytes([4,4]))
        self.assertEqual(NativeB("motion_compensation").decode(a),{"weight":raw})

    def test_sz3_retains_huffman_zstd_and_exact_lorenzo_feedback(self):
        values=[(i*7311)&65535 for i in range(63)]
        raw=struct.pack("<63H",*values)
        a=NativeB("sz_like_lossless").encode(target(raw,columns=9),{"weight":raw},{})
        _,_,_,_,table,n=model(a)
        length,=struct.unpack_from("<Q",a.payload)
        zstd=ctypes.CDLL("libzstd.so.1")
        zstd.ZSTD_decompress.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t]
        zstd.ZSTD_decompress.restype=ctypes.c_size_t
        output=ctypes.create_string_buffer(length)
        self.assertEqual(zstd.ZSTD_decompress(output,length,a.payload[8:],len(a.payload)-8),length)
        self.assertEqual(length,struct.unpack_from("<Q",table,len(table)-8)[0])
        table=table[:-8]
        frame=struct.pack("<IQIIIIQQQ",MAGIC,n*2,0,5,0,1,n,len(table),length)+table+output.raw
        decoded=_call(a_library().fp4_native_decode,frame,len(frame))
        residuals=struct.unpack("<63H",decoded)
        for i,v in enumerate(values):
            prediction=(values[i-1] if i%9 else 0)+(values[i-9] if i>=9 else 0)-(values[i-10] if i>=9 and i%9 else 0)
            self.assertEqual(residuals[i],(v-prediction)&65535)
        self.assertEqual(NativeB("sz_like_lossless").decode(a),{"weight":raw})

    def test_fresh_process_decodes_without_target_or_training_cache(self):
        raw=bytes(range(256))+b"\x00\x80\xff\xff\x01\x7f"
        for method in METHODS:
            a=NativeB(method).encode(target(raw),{"weight":raw},{})
            script="""import json,sys
from fp4_redundancy.methods.predictive import NativeB
from fp4_redundancy.types import EncodedArtifact
r=json.load(sys.stdin)
print(NativeB(r['method']).decode(EncodedArtifact(bytes.fromhex(r['p']),bytes.fromhex(r['s'])))['weight'].hex())
"""
            run=subprocess.run([sys.executable,"-c",script],input=json.dumps({"method":method,"p":a.payload.hex(),"s":a.side_information.hex()}),
                               text=True,capture_output=True,env={**os.environ,"PYTHONPATH":str(ROOT/"src")})
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertEqual(bytes.fromhex(run.stdout),raw)

    def test_physical_rows_parallel_comparison_and_saved_state(self):
        fixture=test_checkpoint.CheckpointTests();fixture.setUp()
        try:
            from fp4_redundancy.checkpoint import load_checkpoint
            from fp4_redundancy.chunks import iter_target_chunks
            for t in load_checkpoint(fixture.snapshot).targets:
                for chunk in iter_target_chunks(t,2050):
                    for c in chunk.components:
                        columns=component_columns(chunk,c)
                        self.assertGreater(columns,0)
            output=fixture.root/"b_run"
            summary=run_experiment(fixture.snapshot,output,workers=2,chunk_bytes=2050,methods=list(METHODS))
            self.assertEqual(summary["status"],"completed")
            self.assertEqual(summary["processed_fraction"],1)
            config=json.loads((output/"run.json").read_text())["configuration"]
            self.assertEqual(config["predictive_build"]["library_sha256"],build_metadata()["library_sha256"])
            for result in summary["methods"].values():
                self.assertEqual(set(result["statuses"]),{"verified"})
                pair=result["baseline_comparison"]
                self.assertEqual(pair["kind"],"complete_scheme")
                self.assertEqual(pair["net_saving_bytes"],pair["baseline_total_bytes"]-result["total_bytes"])
                self.assertEqual(result["saving_bytes"],result["verified_original_bytes"]-result["total_bytes"])
            with sqlite3.connect(output/"results.sqlite3") as db:
                blob,=db.execute("SELECT state_zlib FROM chunks LIMIT 1").fetchone()
                states=json.loads(zlib.decompress(blob))
                self.assertEqual(set(states),set(METHODS)|{"arithmetic"})
            again=run_experiment(fixture.snapshot,output,workers=2,chunk_bytes=2050,methods=list(METHODS),resume=True)
            self.assertEqual(again["completed_chunks"],summary["completed_chunks"])
        finally:fixture.tearDown()
