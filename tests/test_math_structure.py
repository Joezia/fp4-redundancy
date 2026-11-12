"""D complete pipelines: independent mechanism checks, serialized state, exact bytes and costs."""
import base64
import json
import math
import os
import random
import sqlite3
import struct
import subprocess
import sys
import unittest
import zlib

import test_checkpoint
from test_shared_patterns import Reader, target
from test_predictive import ac_words
from fp4_redundancy.catalog import builtin_registry
from fp4_redundancy.experiment import run_experiment
from fp4_redundancy.methods.math_structure import NativeD, METHODS, EXACT, build_metadata
from fp4_redundancy.methods.native import ROOT
from fp4_redundancy.types import EncodedArtifact, Target


def streams(a):
    h=json.loads(a.side_information);part=h["components"][0]["streams"][0]
    r=Reader(base64.b64decode(part["model"]["table_b64"]));version,alphabet=r.get("<II");assert version==1
    control=r.take(r.get("<I")[0]);count=r.get("<I")[0];offset=0;result=[]
    for _ in range(count):
        n=r.get("<Q")[0];model=r.take(r.get("<I")[0]);size=r.get("<Q")[0]
        result.append((n,model,a.payload[offset:offset+size]));offset+=size
    bits=r.get("<Q")[0];assert bits==len(a.payload)*8 and offset==len(a.payload) and r.p==len(r.b)
    return control,result,alphabet


def i32_words(stream):
    n,model,payload=stream;words=ac_words(payload,model,n);out=[]
    for lo,hi in zip(words[::2],words[1::2]):
        u=lo+(hi<<16);out.append(-1-(u>>1) if u&1 else u>>1)
    return out


def state(a):
    control,parts,alphabet=streams(a);r=Reader(control);version,exponent,threads=r.get("<IhH");assert version==1
    shape=[r.get("<I")[0] for _ in range(r.get("<I")[0])]
    ranks=[r.get("<I")[0] for _ in range(r.get("<I")[0])]
    parameters=[r.get("<i")[0] for _ in range(r.get("<I")[0])]
    n=r.get("<Q")[0];assert r.p==len(r.b)
    coefficients=i32_words(parts[0]);assert len(coefficients)==n
    errors=ac_words(parts[1][2],parts[1][1],parts[1][0])
    return dict(exponent=exponent,threads=threads,shape=shape,ranks=ranks,parameters=parameters,coefficients=coefficients,errors=errors)


def bf16(values):return b"".join(struct.pack("<I",struct.unpack("<I",struct.pack("<f",v))[0]>>16)[:2] for v in values)


def reference_lifting(values,cdf):
    out=list(values);n=len(out)
    while n>1:
        low=out[:n:2];high=[out[2*i+1]-((low[i]+low[min(i+1,len(low)-1)])//2 if cdf else low[i]) for i in range(n//2)]
        for i in range(len(low)):
            low[i]+=((high[max(i-1,0)]+high[min(i,len(high)-1)]+2)//4 if cdf else high[i]//2 if i<len(high) else 0)
        out[:n]=low+high;n=(n+1)//2
    return out


@unittest.skipUnless((ROOT/"build/libfp4_math_structure.so").is_file(),"Build D codecs first")
class MathematicalTests(unittest.TestCase):
    def test_bwt_mtf_known_inverse_and_primaries(self):
        raw=b"BANANA";a=NativeD("bwt_mtf").encode(target(raw),{"weight":raw},{})
        control,parts,_=streams(a)
        self.assertEqual(struct.unpack("<II",control),(65536,3))
        self.assertEqual(ac_words(parts[0][2],parts[0][1],parts[0][0]),[78,0,67,67,0,0])
        self.assertEqual(NativeD("bwt_mtf").decode(a),{"weight":raw})
        for raw in (b"a",b"aaaaaaa",b"abababab",bytes(range(256))*257):
            a=NativeD("bwt_mtf").encode(target(raw),{"weight":raw},{})
            control,_,_=streams(a)
            self.assertEqual(len(control),4+4*((len(raw)+65535)//65536))
            self.assertEqual(NativeD("bwt_mtf").decode(a),{"weight":raw})

    def test_two_integer_liftings_match_signed_floor_reference(self):
        rng=random.Random(21)
        for method,cdf in (("cdf53",True),("haar_lifting",False)):
            for values in ([1,2,3,4],[65535,0,65535,0,1],[rng.randrange(65536) for _ in range(257)]):
                raw=struct.pack("<"+"H"*len(values),*values);a=NativeD(method).encode(target(raw,"BF16"),{"weight":raw},{})
                control,parts,_=streams(a)
                self.assertEqual(control,struct.pack("<I",256))
                expected=[]
                for i in range(0,len(values),256):expected.extend(reference_lifting(values[i:i+256],cdf))
                self.assertEqual(i32_words(parts[0]),expected)
                self.assertEqual(NativeD(method).decode(a),{"weight":raw})
            self.assertEqual(NativeD(method).spec.fidelity,"native")

    def test_bitplane_is_local_complete_inverse_with_independent_models(self):
        raw=bytes([0x80,0x01,0xff]);a=NativeD("bitplane").encode(target(raw),{"weight":raw},{})
        control,parts,_=streams(a);self.assertEqual(struct.unpack("<I",control),(8,))
        planes=[ac_words(p,m,n) for n,m,p in parts]
        self.assertEqual(planes[0],[1,0,1]);self.assertEqual(planes[-1],[0,1,1])
        self.assertEqual(len(parts),8)
        self.assertEqual(NativeD("bitplane").decode(a),{"weight":raw})
        raw=bytes.fromhex("10fe8023");t=target(raw,"fp4_packed_high_first")
        a=NativeD("bitplane").encode(t,{"weight":raw},{})
        self.assertEqual(len(streams(a)[1]),4)
        self.assertEqual(NativeD("bitplane").decode(a),{"weight":raw})

    def test_transforms_save_basis_mean_window_coefficients_and_correction(self):
        rng=random.Random(11);values=[-0.0]+[rng.uniform(-2,2) for _ in range(18)];raw=bf16(values)
        for method,parameter_count,coefficient_count in (("dct_lossless",64,24),("dst_lossless",64,24),("pca_lossless",72,24),("mdct_lossless",144,32)):
            a=NativeD(method).encode(target(raw,"BF16"),{"weight":raw},{})
            q=state(a)
            self.assertEqual(len(q["parameters"]),parameter_count)
            self.assertEqual(len(q["coefficients"]),coefficient_count)
            self.assertEqual(len(q["errors"]),len(values))
            self.assertTrue(any(q["errors"]))
            self.assertEqual(NativeD(method).decode(a),{"weight":raw})
            if method=="mdct_lossless":
                window=q["parameters"][:16];unit=1<<20
                for i in range(8):self.assertLess(abs(window[i]**2+window[i+8]**2-unit**2),2_000_000)
        ambiguous=Target("unknown",(8,),target(bytes(4),"fp4_packed_low_first").components)
        self.assertFalse(NativeD("dct_lossless").check_support(ambiguous,{}).applicable)
        self.assertTrue(NativeD("cdf53").check_support(ambiguous,{}).applicable)

    def test_mdct_exact_inverse_path_uses_stored_window_and_overlap(self):
        for size in (1,7,8,9,17,39):
            values=[math.sin(i*.37) for i in range(size)];values[0]+=1;values[-1]-=.5
            raw=bf16(values);a=NativeD("mdct_lossless").encode(target(raw,"BF16"),{"weight":raw},{})
            q=state(a);unit=1<<20;w=q["parameters"][:16];cosines=q["parameters"][16:];c=q["coefficients"]
            nv=(size+7)//8;acc=[0]*((nv+2)*8)
            for f in range(nv+1):
                for i in range(16):acc[f*8+i]+=sum(c[f*8+k]*cosines[i*8+k] for k in range(8))*w[i]
            # Integer reconstruction should closely match normalized input BEFORE
            # projection/XOR. Native checks separately prove floating TDAC exactly.
            native_values=[struct.unpack("<f",b"\0\0"+raw[i:i+2])[0] for i in range(0,len(raw),2)]
            for i,v in enumerate(native_values):self.assertLess(abs(acc[8+i]/(4*unit**3)*2**q["exponent"]-v),1e-4)
            self.assertEqual(NativeD("mdct_lossless").decode(a),{"weight":raw})

    def test_sparse_actual_support_and_learned_dictionary(self):
        raw=bytes.fromhex("02000000");a=NativeD("sparse_coding").encode(target(raw,"fp4_packed_low_first"),{"weight":raw},{})
        q=state(a);self.assertEqual(q["shape"],[8,16]);self.assertEqual(q["ranks"],[3])
        self.assertEqual(q["coefficients"],[1,0,1<<19])
        self.assertEqual(q["errors"],[0]*8)
        rng=random.Random(9);raw=bf16([math.sin(.17*(i+1)*(j+1))+.1*(i==j%8) for j in range(64) for i in range(8)])
        fixed=state(NativeD("sparse_coding").encode(target(raw,"BF16"),{"weight":raw},{}))
        learned=NativeD("ksvd_lossless").encode(target(raw,"BF16"),{"weight":raw},{})
        q=state(learned);self.assertEqual(len(q["parameters"]),128);self.assertNotEqual(q["parameters"],fixed["parameters"])
        offset=0;nnz=[]
        for _ in range(64):
            count=q["coefficients"][offset];offset+=1;nnz.append(count);self.assertLessEqual(count,3)
            atoms=q["coefficients"][offset:offset+2*count:2];self.assertEqual(len(set(atoms)),count)
            self.assertTrue(all(0<=v<16 for v in atoms));offset+=2*count
        self.assertEqual(offset,len(q["coefficients"]));self.assertTrue(any(v>1 for v in nnz))
        self.assertEqual(NativeD("ksvd_lossless").decode(learned),{"weight":raw})

    def test_distinct_factor_shapes_ranks_and_core_lengths(self):
        raw=bf16([((i%7)-3)/8 for i in range(73)])
        configurations={"svd_lowrank":([8,8],[2],64),"cp_lossless":([4,4,4],[2],48),"tucker_lossless":([4,4,4],[2,2,2],64),
                        "tt_lossless":([4,4,4],[1,2,2,1],64),"kronecker_lossless":([8,8,2,2,4,4],[1],40),"circulant_lossless":([8,8],[8],16)}
        for method,(shape,ranks,count) in configurations.items():
            a=NativeD(method).encode(target(raw,"BF16"),{"weight":raw},{})
            q=state(a)
            self.assertEqual(q["shape"],shape);self.assertEqual(q["ranks"],ranks);self.assertEqual(len(q["coefficients"]),count)
            self.assertEqual(q["parameters"],[])
            self.assertEqual(NativeD(method).decode(a),{"weight":raw})

    def test_independent_process_restores_special_patterns_and_rejects_truncation(self):
        raw=bytes.fromhex("00000080807f80ff817fc1ff0100ffff")+bytes(range(64))
        for method in METHODS:
            a=NativeD(method).encode(target(raw,"BF16"),{"weight":raw},{})
            script="""import json,sys
from fp4_redundancy.methods.math_structure import NativeD
from fp4_redundancy.types import EncodedArtifact
r=json.load(sys.stdin)
print(NativeD(r['m']).decode(EncodedArtifact(bytes.fromhex(r['p']),bytes.fromhex(r['s'])))['weight'].hex())
"""
            run=subprocess.run([sys.executable,"-c",script],text=True,capture_output=True,input=json.dumps({"m":method,"p":a.payload.hex(),"s":a.side_information.hex()}),
                               env={**os.environ,"PYTHONPATH":str(ROOT/"src")})
            self.assertEqual(run.returncode,0,run.stderr);self.assertEqual(bytes.fromhex(run.stdout),raw)
            for payload in (a.payload[:-1],a.payload+b"extra"):
                with self.assertRaises(ValueError):NativeD(method).decode(EncodedArtifact(payload,a.side_information))

    def test_parallel_run_full_cost_both_benefits_states_and_resume(self):
        f=test_checkpoint.CheckpointTests();f.setUp()
        try:
            output=f.root/"d_run";methods=list(METHODS)
            summary=run_experiment(f.snapshot,output,workers=2,chunk_bytes=2050,methods=methods)
            self.assertEqual(summary["status"],"completed")
            config=json.loads((output/"run.json").read_text())["configuration"]
            self.assertEqual(config["math_structure_build"]["library_sha256"],build_metadata()["library_sha256"])
            for value in summary["methods"].values():
                self.assertEqual(set(value["statuses"]),{"verified"})
                self.assertEqual(value["total_bytes"],value["payload_bytes"]+value["side_information_bytes"])
                self.assertEqual(value["saving_bytes"],value["verified_original_bytes"]-value["total_bytes"])
                pair=value["baseline_comparison"];self.assertEqual(pair["kind"],"complete_scheme")
                self.assertEqual(pair["net_saving_bytes"],pair["baseline_total_bytes"]-pair["method_total_bytes"])
            with sqlite3.connect(output/"results.sqlite3") as db:
                blob,=db.execute("SELECT state_zlib FROM chunks LIMIT 1").fetchone();states=json.loads(zlib.decompress(blob))
                self.assertEqual(set(states),set(methods)|{"arithmetic"})
            again=run_experiment(f.snapshot,output,workers=2,chunk_bytes=2050,methods=methods,resume=True)
            self.assertEqual(again["completed_chunks"],summary["completed_chunks"])
        finally:f.tearDown()
