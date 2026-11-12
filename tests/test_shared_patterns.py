"""C algorithms: mechanism checks as well as exact byte restoration and cost accounting."""
import base64
from collections import Counter
import json
import os
import random
import sqlite3
import struct
import subprocess
import sys
import unittest
from unittest.mock import patch
import zlib

import test_checkpoint
from test_predictive import ac_words
from fp4_redundancy.catalog import builtin_registry
from fp4_redundancy.experiment import run_experiment
from fp4_redundancy.methods.native import ROOT
from fp4_redundancy.methods.predictive import NativeB
from fp4_redundancy.methods.shared_patterns import METHODS, NativeC, InterFrameReference, build_metadata
from fp4_redundancy.types import Target, Component, EncodedArtifact


def target(raw, fmt="F8_E4M3"):
    return Target("synthetic",(len(raw),),(Component("weight","packed_fp4" if fmt.startswith("fp4") else "other",
                  "missing",0,len(raw),fmt),),{"fp4_value_format":"E2M1FN"})


class Reader:
    def __init__(self, b):self.b,self.p=b,0
    def var(self):
        v=shift=0
        while True:
            byte=self.b[self.p];self.p+=1;v|=(byte&127)<<shift
            if byte<128:return v
            shift+=7
    def take(self,n):b=self.b[self.p:self.p+n];self.p+=n;return b
    def get(self,fmt):n=struct.calcsize(fmt);return struct.unpack(fmt,self.take(n))


def parameters(artifact):
    h=json.loads(artifact.side_information)
    stream=h["components"][0]["streams"][0]
    m=base64.b64decode(stream["model"]["table_b64"])
    version,alphabet,n=struct.unpack_from("<III",m)
    assert version==1 and len(m)==12+n+8
    return m[12:12+n],alphabet,stream["count"]


def grammar(a):
    b,alphabet,n=parameters(a);r=Reader(b);rules=[]
    for _ in range(r.var()):rules.append([r.var() for _ in range(r.var())])
    roots=r.var();assert r.p==len(b);p=Reader(a.payload);root=[p.var() for _ in range(roots)];assert p.p==len(a.payload)
    return rules,root,alphabet


def quantizer(a,method):
    b,alphabet,n=parameters(a);r=Reader(b)
    dim,k,nt,exponent,workers=r.get("<IIIhH")
    rotation=r.get("<64i") if method=="opq_lossless" else None
    m=1 if method=="vq_lossless" else 2
    book_dim=4 if method in ("pq_lossless","opq_lossless") else 8
    books=r.get("<"+"i"*(m*k*book_dim))
    itable=r.take(r.get("<I")[0]);etable=r.take(r.get("<I")[0]);ni=r.get("<Q")[0]
    assert r.p==len(b)
    indices=ac_words(a.payload[:ni],itable,((n+7)//8)*m)
    errors=ac_words(a.payload[ni:],etable,n)
    return dict(dim=dim,k=k,nt=nt,exponent=exponent,workers=workers,rotation=rotation,
                books=books,indices=indices,errors=errors)


@unittest.skipUnless((ROOT/"build/libfp4_shared_patterns.so").is_file(),"Build C codecs first")
class SharedPatternTests(unittest.TestCase):
    def test_rle_and_lz_known_tokens_dictionary_growth_and_kwkwk(self):
        raw=b"AAAABBBCCDAA"
        a=NativeC("rle").encode(target(raw),{"weight":raw},{})
        self.assertEqual(a.payload,bytes([4,65,3,66,2,67,1,68,2,65]))
        raw=b"ABABABA"
        for method in ("lz77","lz78","lzss","lzw"):
            a=NativeC(method).encode(target(raw),{"weight":raw},{})
            self.assertEqual(NativeC(method).decode(a),{"weight":raw})
            self.assertLess(len(a.payload),len(raw)+5)
            if method=="lz77":
                r=Reader(a.payload);triples=[]
                while r.p<len(r.b):triples.append(tuple(r.var() for _ in range(3)))
                self.assertEqual(triples,[(0,0,65),(0,0,66),(2,5,65536)])
            if method=="lz78":self.assertEqual(a.payload,bytes([0,65,0,66,1,66,3,65]))
            if method=="lzss":self.assertEqual(a.payload,bytes([3,65,66,18,0]))
            if method=="lzw":
                # Welch example includes the not-yet-defined next code (KwKwK).
                bits="".join(f"{c:09b}" for c in (65,66,256,258))
                bits+="0"*(-len(bits)%8)
                self.assertEqual(a.payload,bytes(int(bits[i:i+8],2) for i in range(0,len(bits),8)))
        rng=random.Random(19);raw=bytes(rng.randrange(16) for _ in range(9000))
        a=NativeC("lzw").encode(target(raw),{"weight":raw},{})
        self.assertEqual(NativeC("lzw").decode(a),{"weight":raw})

    def test_repair_matches_independent_global_pair_selection(self):
        rng=random.Random(71)
        for raw in (b"aaaaaaa",b"abcdbcabcdbc",bytes(rng.randrange(4) for _ in range(111))):
            a=NativeC("repair").encode(target(raw),{"weight":raw},{})
            actual,root,alphabet=grammar(a);seq=list(raw);expected=[]
            while True:
                counts={}
                for pair in set(zip(seq,seq[1:])):
                    i=0;count=0
                    while i+1<len(seq):
                        if tuple(seq[i:i+2])==pair:count+=1;i+=2
                        else:i+=1
                    counts[pair]=count
                if not counts or max(counts.values())<2:break
                pair=min(counts,key=lambda p:(-counts[p],p));code=alphabet+len(expected);expected.append(list(pair));out=[];i=0
                while i<len(seq):
                    if tuple(seq[i:i+2])==pair:out.append(code);i+=2
                    else:out.append(seq[i]);i+=1
                seq=out
            self.assertEqual(actual,expected)
            self.assertEqual(root,seq)

    def test_sequitur_online_grammar_invariants(self):
        for raw in (b"abcdbcabcd",b"a"*129,b"abbbabcbb",b"abcdefgh"*43,bytes(range(32))*7):
            a=NativeC("sequitur").encode(target(raw),{"weight":raw},{})
            rules,root,alphabet=grammar(a)
            uses=Counter(v for rhs in rules+[root] for v in rhs if v>=alphabet)
            self.assertTrue(all(uses[alphabet+i]>=2 for i in range(len(rules))))
            seen={}
            for ri,rhs in enumerate(rules+[root]):
                for i,pair in enumerate(zip(rhs,rhs[1:])):
                    if pair in seen:
                        oldri,oldi=seen[pair]
                        self.assertTrue(oldri==ri and abs(oldi-i)==1,"Nonoverlapping duplicate digram")
                    else:seen[pair]=(ri,i)
            self.assertEqual(NativeC("sequitur").decode(a),{"weight":raw})

    def test_dedup_discovers_variable_boundaries_and_emits_real_references(self):
        rng=random.Random(11);pattern=bytes(rng.randrange(256) for _ in range(4096));raw=pattern+b"x"+pattern
        for method in ("block_dedup","cdc_dedup"):
            source=(pattern*3) if method=="block_dedup" else raw
            a=NativeC(method).encode(target(source),{"weight":source},{})
            r=Reader(a.payload);lengths=[];references=0;dictionary=[];restored=bytearray()
            while r.p<len(r.b):
                n=r.var()
                if n:
                    chunk=r.take(n);dictionary.append(chunk);restored.extend(chunk);lengths.append(n)
                else:restored.extend(dictionary[r.var()]);references+=1
            self.assertEqual(bytes(restored),source)
            self.assertGreater(references,0)
            if method=="cdc_dedup":
                self.assertGreater(len(set(lengths)),1)
                self.assertTrue(all(64<=n<=1024 for n in lengths[:-1]))

    def test_references_are_selected_paid_self_contained_and_alias_reuses_b(self):
        raw=b"the internal reference 12345 "*80
        for method in ("vcdiff","rlz"):
            a=NativeC(method).encode(target(raw),{"weight":raw},{})
            b,_,_=parameters(a);r=Reader(b);n=r.var()
            self.assertGreater(n,0)
            self.assertEqual(r.take(n),raw[:n])
            self.assertEqual(r.p,len(b))
            self.assertEqual(NativeC(method).decode(a),{"weight":raw})
            if method=="vcdiff":self.assertEqual(a.payload[:3],bytes.fromhex("d6c3c4"))
        t=target(raw);source={"weight":raw}
        original=NativeB("motion_compensation").encode(t,source,{})
        alias=InterFrameReference().encode(t,source,{})
        self.assertEqual(alias.payload,original.payload)
        self.assertEqual(json.loads(alias.side_information)["components"],json.loads(original.side_information)["components"])
        self.assertEqual(InterFrameReference().decode(alias),source)
        self.assertEqual(InterFrameReference().spec.implementation_version,NativeB("motion_compensation").spec.implementation_version)

    def test_fp4_training_is_e2m1_numeric_and_preserves_negative_zero(self):
        # Low-first complete codes: 0,7,8,15,1,9,2,10. E2M1 values, not code indices.
        raw=bytes.fromhex("70f891a2");t=target(raw,"fp4_packed_low_first")
        a=NativeC("vq_lossless").encode(t,{"weight":raw},{})
        q=quantizer(a,"vq_lossless")
        self.assertEqual((q["dim"],q["k"],q["exponent"]),(8,1,3))
        self.assertEqual(q["books"],tuple(int(v/8*(1<<20)) for v in (0,6,0,-6,.5,-.5,1,-1)))
        self.assertEqual(q["errors"],[0,0,8,0,0,0,0,0])
        self.assertEqual(NativeC("vq_lossless").decode(a),{"weight":raw})
        ambiguous=Target(t.id,t.logical_shape,t.components)
        self.assertFalse(NativeC("vq_lossless").check_support(ambiguous,{}).applicable)
        self.assertTrue(NativeC("rle").check_support(ambiguous,{}).applicable)

    def test_opq_learns_rotation_and_all_c3_save_real_books_indices_corrections(self):
        rng=random.Random(1234);words=[]
        for _ in range(96):
            x,y=rng.uniform(-2,2),rng.uniform(-1,1)
            for v in (x,y,x+y,x-y,2*x+y,x+2*y,-x+y,2*y-x):words.append(struct.unpack("<I",struct.pack("<f",v))[0]>>16)
        raw=struct.pack("<"+"H"*len(words),*words)
        tables={}
        for method in ("vq_lossless","pq_lossless","opq_lossless","rvq_lossless","aq_lossless"):
            a=NativeC(method).encode(target(raw,"BF16"),{"weight":raw},{})
            q=quantizer(a,method);tables[method]=q["books"]
            self.assertEqual(q["nt"],96)
            self.assertEqual(q["k"],16)
            self.assertGreater(len(set(q["indices"])),1)
            self.assertTrue(any(q["errors"]))
            self.assertEqual(NativeC(method).decode(a),{"weight":raw})
            if method=="opq_lossless":
                matrix=q["rotation"];unit=1<<20
                self.assertGreater(max(abs(matrix[i*8+j]) for i in range(8) for j in range(8) if i!=j),unit//100)
                for i in range(8):
                    for j in range(8):self.assertLess(abs(sum(matrix[i*8+k]*matrix[j*8+k] for k in range(8))-(unit*unit if i==j else 0)),10_000_000)
        self.assertNotEqual(tables["rvq_lossless"],tables["aq_lossless"])
        for threads in (1,2):
            with patch.dict(os.environ,FP4_C_THREADS=str(threads)):
                a=NativeC("pq_lossless").encode(target(raw,"BF16"),{"weight":raw},{})
                q=quantizer(a,"pq_lossless")
                self.assertEqual(q["workers"],threads)
                self.assertEqual(q["books"],tables["pq_lossless"])

    def test_fresh_process_all_c_and_corrupt_artifacts(self):
        raw=(bytes.fromhex("000080007f80ff80ffff7fc1")+bytes(range(64)))*3
        for method in METHODS:
            a=NativeC(method).encode(target(raw,"BF16"),{"weight":raw},{})
            script="""import json,sys
from fp4_redundancy.methods.shared_patterns import NativeC
from fp4_redundancy.types import EncodedArtifact
r=json.load(sys.stdin)
print(NativeC(r['m']).decode(EncodedArtifact(bytes.fromhex(r['p']),bytes.fromhex(r['s'])))['weight'].hex())
"""
            p=subprocess.run([sys.executable,"-c",script],input=json.dumps({"m":method,"p":a.payload.hex(),"s":a.side_information.hex()}),
                             text=True,capture_output=True,env={**os.environ,"PYTHONPATH":str(ROOT/"src")})
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(bytes.fromhex(p.stdout),raw)
            for payload in (a.payload[:-1],a.payload+b"extra",bytes([a.payload[0]^255])+a.payload[1:]):
                with self.assertRaises(ValueError,msg=method):NativeC(method).decode(EncodedArtifact(payload,a.side_information))

    def test_parallel_checkpoint_accounting_states_and_resume(self):
        f=test_checkpoint.CheckpointTests();f.setUp()
        try:
            methods=list(METHODS)+["inter_frame_reference"];output=f.root/"c_run"
            result=run_experiment(f.snapshot,output,workers=2,chunk_bytes=2050,methods=methods)
            self.assertEqual(result["status"],"completed")
            config=json.loads((output/"run.json").read_text())["configuration"]
            self.assertEqual(config["shared_pattern_build"]["library_sha256"],build_metadata()["library_sha256"])
            self.assertIn("predictive_build",config)
            for v in result["methods"].values():
                self.assertEqual(set(v["statuses"]),{"verified"})
                self.assertEqual(v["total_bytes"],v["payload_bytes"]+v["side_information_bytes"])
                self.assertEqual(v["saving_bytes"],v["verified_original_bytes"]-v["total_bytes"])
                p=v["baseline_comparison"]
                self.assertEqual(p["kind"],"complete_scheme")
                self.assertEqual(p["net_saving_bytes"],p["baseline_total_bytes"]-p["method_total_bytes"])
            with sqlite3.connect(output/"results.sqlite3") as db:
                row,=db.execute("SELECT state_zlib FROM chunks LIMIT 1").fetchone()
                self.assertEqual(set(json.loads(zlib.decompress(row))),set(methods)|{"arithmetic"})
            again=run_experiment(f.snapshot,output,workers=2,chunk_bytes=2050,methods=methods,resume=True)
            self.assertEqual(again["completed_chunks"],result["completed_chunks"])
        finally:f.tearDown()
