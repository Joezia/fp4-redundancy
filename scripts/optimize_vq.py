"""Bounded VQ ablations and parameter grid on the exact existing 36 FP4 chunks."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import struct
import subprocess
import sys
import time
import tempfile
import zlib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from fp4_redundancy.experiment import _atomic_json, _now

EXPERIMENT=ROOT/"experiments/fp4-sample"
PLAN=EXPERIMENT/"plan.json"
KEY="vq_parameter_exploration"
SECTION="\n## VQ 参数探索\n"


def update_section(path, text):
    original=path.read_text() if path.exists() else ""
    path.write_text(original.split(SECTION,1)[0].rstrip()+SECTION+"\n"+text.strip()+"\n",encoding="utf-8")


def merge_json(name, value):
    path=EXPERIMENT/name
    document=json.loads(path.read_text()) if path.exists() else {}
    document[KEY]={**document.get(KEY,{}),**value}
    _atomic_json(path,document)


def merge_csv(name, rows):
    path=EXPERIMENT/(name+".csv")
    with path.open(newline="") as f:
        reader=csv.DictReader(f);fields=list(reader.fieldnames);original=[r for r in reader if not r.get("variant")]
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    if name=="results":
        if "payload_saving_fraction" not in fields:fields.append("payload_saving_fraction")
        for row in original+rows:
            row["payload_saving_fraction"]=1-float(row["payload_bytes"])/float(row["original_bytes"])
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(original+rows)


def refresh_sample_payload_table():
    """Refresh the original methods in the same historical report, keeping its prose."""
    with (EXPERIMENT/"results.csv").open(newline="") as f:
        rows=[r for r in csv.DictReader(f) if not r.get("variant")]
    path=EXPERIMENT/"report.md";text=path.read_text()
    start=text.index("| 方法 | 已处理块 | 状态 |")
    end=text.index("\n\n",start)
    lines=["| 方法 | 已处理块 | 状态 | Payload bytes | Payload节省率 | 元数据 bytes | 合计 bytes | 总节省率 | 相对算术净节省 |",
           "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in sorted(rows,key=lambda r:-float(r["saving_fraction"])):
        net="—" if not r["net_saving_fraction"] else f"{float(r['net_saving_fraction']):.2%}"
        lines.append(f"| {r['method_id']} | {r['completed_chunks']}/{r['sample_chunks']} | {r['status']} | {r['payload_bytes']} | {float(r['payload_saving_fraction']):.2%} | {r['metadata_bytes']} | {r['total_bytes']} | {float(r['saving_fraction']):.2%} | {net} |")
    path.write_text(text[:start]+"\n".join(lines)+text[end:],encoding="utf-8")


def prepare():
    base=ROOT/"experiments/fp4-sample/plan.json"
    old=json.loads(base.read_text())
    selection={key:old[key] for key in ("model_id","revision","chunk_bytes","chunks")}
    variants=[]
    for dim in (2,4,8):
        for centers in (4,16,64):
            variants.append({"id":f"d{dim}_k{centers}_rate_auto","method":"vq_codeword",
                             "parameters":{"vector_length":dim,"centers":centers,"objective":"rate","correction_model":"auto"}})
    for objective,mode in (("hamming","shared"),("rate","shared"),("rate","conditioned")):
        variants.append({"id":f"d8_k16_{objective}_{mode}","method":"vq_codeword",
                         "parameters":{"vector_length":8,"centers":16,"objective":objective,"correction_model":mode}})
    variants += [{"id":"numeric_vq_control","method":"vq_lossless","parameters":{}},
                 {"id":"arithmetic_control","method":"arithmetic","parameters":{}}]
    plan={"created_utc":_now(),"snapshot":old["snapshot"],"base_plan_sha256":hashlib.sha256(base.read_bytes()).hexdigest(),
          "sample_chunks":old["sample_chunks"],"sample_original_bytes":old["sample_original_bytes"],
          "selection":selection,"variants":variants,"parallel_variants":4,"workers_per_variant":4,
          "coverage":"subset","selection_policy":"Report every configuration; best-on-these-36 exploration, no held-out claim",
          "payload_persisted":False}
    if KEY in old:raise ValueError("VQ parameter plan already exists in the sample plan")
    old[KEY]=plan
    _atomic_json(PLAN,old)
    update_section(EXPERIMENT/"plan.md","""

沿用原 FP4 抽样的同36个块、64 KiB上限、原序与独立scale，总原始大小2,242,368 bytes。

- 原数值VQ和零阶算术编码各重跑一次作对照，另存新目录；不覆盖原结果。
- 一个新完整插件 vq_codeword：实际完整码字向量 medoid，packed 码本、索引、精确 XOR 纠错；C++ 实现，复用原 A1 算术编码。
- 原序分层确定性抖动抽取至多2048个训练向量，五轮分配/medoid更新，解码不训练。
- 主网格：向量长度2/4/8 × 码本上限4/16/64，共9组，使用预计码长目标和纠错模型auto。
- 消融：8维/16中心的 Hamming＋共享纠错、码长目标＋共享纠错、码长目标＋条件纠错，共3组。
- 条件纠错按中心×位置统计；每表先比较估计收益与表成本，低收益时用共享模型。auto再按完整产物实际可变成本比较共享/条件方案。
- 无原样或算术-only回退；共享回退仍保留VQ码本、索引和纠错完整流程。
- 所有方法逐字节验证，记录完整大小、索引/纠错/码本/元数据成本及原始/净收益；按方法存 runs/<方法>/vq_optimization/<配置>/。
- 最多4组×4进程并行，内部线程1；不运行全模型。
- 保存全部配置结果。“最佳”只表示这36块上的探索最优，不声称独立验证或全模型最优。
""")
    print(json.dumps({"variants":len(variants),"chunks":36,"bytes":old["sample_original_bytes"]}))


def output_for(variant):
    return ROOT/"runs"/variant["method"]/"vq_optimization"/variant["id"]


def blob(data,pos):
    n,=struct.unpack_from("<I",data,pos);pos+=4
    return data[pos:pos+n],pos+n


def statistics(output,method):
    stats={"weight_index_bytes":0,"weight_correction_bytes":0,"weight_codebook_bytes":0,
           "weight_symbols":0,"weight_shared_components":0,"weight_conditioned_components":0,
           "weight_single_center_components":0,
           "weight_active_contexts":0,"weight_used_centers_min":None,"weight_used_centers_max":None}
    hashes={};pairs=0
    with sqlite3.connect(f"file:{output/'results.sqlite3'}?mode=ro",uri=True) as db:
        for cid,record,compressed in db.execute("SELECT chunk_id,result_json,state_zlib FROM chunks"):
            result=json.loads(record)["results"][0]
            assert result["status"]=="verified" and result["bitwise_verified"]
            hashes[cid]=result["source_sha256"];pairs+=1
            if method=="arithmetic":continue
            h=json.loads(zlib.decompress(compressed))[method]["side_information"]
            for component in h["components"]:
                if component["role"]!="packed_fp4":continue
                stream=component["streams"][0];data=base64.b64decode(stream["model"]["table_b64"])
                control,_=blob(data,8)
                if method=="vq_codeword":
                    _,dim,requested,k,mode,objective,nt,nv=struct.unpack_from("<IHHHBBII",control)
                    book,pos=blob(control,20);imodel,pos=blob(control,pos)
                    ni,=struct.unpack_from("<Q",control,pos);emodel,pos=blob(control,pos+8)
                    global_model,p=blob(emodel,16);ng,=struct.unpack_from("<I",emodel,p);p+=4
                    for _ in range(ng):
                        flag=emodel[p];p+=1
                        if flag:_,p=blob(emodel,p);stats["weight_active_contexts"]+=1
                    assert p==len(emodel)
                    stats["weight_conditioned_components" if mode else "weight_shared_components"]+=1
                else:
                    dim,k,nt,exponent,threads=struct.unpack_from("<IIIhH",control)
                    book=control[16:16+k*dim*4];imodel,pos=blob(control,16+len(book));emodel,pos=blob(control,pos)
                    ni,=struct.unpack_from("<Q",control,pos)
                stats["weight_symbols"]+=stream["count"];stats["weight_index_bytes"]+=ni
                stats["weight_single_center_components"]+=k==1
                stats["weight_correction_bytes"]+=stream["bytes"]-ni;stats["weight_codebook_bytes"]+=len(book)
                stats["weight_used_centers_min"]=min(k,stats["weight_used_centers_min"]) if stats["weight_used_centers_min"] is not None else k
                stats["weight_used_centers_max"]=max(k,stats["weight_used_centers_max"]) if stats["weight_used_centers_max"] is not None else k
    stats["verified_pairs"]=pairs
    if stats["weight_symbols"]:
        stats["weight_index_bits_per_symbol"]=stats["weight_index_bytes"]*8/stats["weight_symbols"]
        stats["weight_correction_bits_per_symbol"]=stats["weight_correction_bytes"]*8/stats["weight_symbols"]
    else:
        stats["weight_index_bits_per_symbol"]=stats["weight_correction_bits_per_symbol"]=None
    return stats,hashes


def report(plan):
    rows=[];reference=None;components=[]
    for variant in plan["variants"]:
        output=output_for(variant);path=output/"summary.json"
        if not path.exists():continue
        summary=json.loads(path.read_text());method=variant["method"];m=summary["methods"][method]
        if m["statuses"]!={"verified":36}:
            raise RuntimeError(f"Unverified sample: {variant['id']}: {m['statuses']}")
        stats,hashes=statistics(output,method)
        if reference is None:reference=hashes
        assert reference==hashes,"Source samples differ across configurations"
        assert summary["status"]=="completed_subset" and summary["selection_processed_fraction"]==1
        base=m.get("baseline_comparison",{})
        row={"variant":variant["id"],"method_id":method,"verified_chunks":summary["completed_chunks"],
             "status":summary["status"],"completed_chunks":summary["completed_chunks"],"sample_chunks":plan["sample_chunks"],
             "chunk_bytes_limit":plan["selection"]["chunk_bytes"],"statuses":json.dumps(m["statuses"],sort_keys=True),
             "original_bytes":m["verified_original_bytes"],"payload_bytes":m["payload_bytes"],
             "metadata_bytes":m["side_information_bytes"],"shared_metadata_bytes":m["shared_side_information_bytes"],
             "total_bytes":m["total_bytes"],"saving_fraction":m["saving_fraction_on_verified_bytes"],
             "payload_saving_fraction":1-m["payload_bytes"]/m["verified_original_bytes"],
             "net_saving_fraction":base.get("net_saving_fraction"),**stats}
        assert row["payload_bytes"]+row["metadata_bytes"]==row["total_bytes"]
        for key,group in m["by_component_format_layout"].items():components.append({"variant":variant["id"],"method_id":method,"component":key,**group})
        assert sum(g["direct_total_bytes"] for g in m["by_component_format_layout"].values())+row["shared_metadata_bytes"]==row["total_bytes"]
        rows.append(row)
    if not rows:return
    for name,data in (("results",rows),("components",components)):
        merge_csv(name,data)
    refresh_sample_payload_table()
    numeric=next((r for r in rows if r["method_id"]=="vq_lossless"),None)
    control=next((r for r in rows if r["method_id"]=="arithmetic"),None)
    if numeric:
        old=json.loads((ROOT/"runs/vq_lossless/fp4_sample/summary.json").read_text())["methods"]["vq_lossless"]
        assert numeric["total_bytes"]==old["total_bytes"],"Original VQ changed"
    if control:
        old=json.loads((ROOT/"runs/arithmetic/fp4_sample/summary.json").read_text())["methods"]["arithmetic"]
        assert control["total_bytes"]==old["total_bytes"],"Arithmetic control changed"
    best=min((r for r in rows if r["method_id"]=="vq_codeword"),key=lambda r:r["total_bytes"],default=None)
    def pct(x):return "—" if x is None else f"{x:.2%}"
    lines=[f"更新于 {_now()}；完成 {len(rows)}/{len(plan['variants'])} 组。样本与配置见[方案](plan.md)。",
           "本节字节数含独立scale；Payload节省率排除独立元数据，总节省率和相对算术净节省保留完整成本。正值节省，负值膨胀，仅为本批36块结果。","",
           "| 配置 | Payload bytes | Payload节省率 | 元数据 bytes | 合计 bytes | 总节省率 | 相对算术净节省 | FP4 索引 bit/码字 | FP4 纠错 bit/码字 |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in sorted(rows,key=lambda r:r["total_bytes"]):
        index="—" if r["weight_index_bits_per_symbol"] is None else f"{r['weight_index_bits_per_symbol']:.3f}"
        correction="—" if r["weight_correction_bits_per_symbol"] is None else f"{r['weight_correction_bits_per_symbol']:.3f}"
        lines.append(f"| {r['variant']} | {r['payload_bytes']} | {pct(r['payload_saving_fraction'])} | {r['metadata_bytes']} | {r['total_bytes']} | {pct(r['saving_fraction'])} | {pct(r['net_saving_fraction'])} | {index} | {correction} |")
    if best and numeric and control:
        lines += ["",f"本批最优配置：{best['variant']}；相比原数值VQ，完整产物减少 {pct(1-best['total_bytes']/numeric['total_bytes'])}。",
                  f"该配置有 {best['weight_single_center_components']}/36 个FP4块最终仅用一个中心，主要避免无效索引开销；尚不能解释为发现显著向量共享收益。",
                  "该配置在同一批样本上挑选，未做独立泛化验证；所有候选均保留，不将候选间收益相加。",
                  "auto 的共享/条件选择是插件内部已计费的选择，仍保留VQ完整流程。未加入原样或算术-only回退。",
                  "详见 [results.csv](results.csv)；其中码本字节是元数据的子项，不能重复相加。权重/scale直接成本见 [components.csv](components.csv)，共有头部另列并计入完整大小。"]
        by_id={r["variant"]:r for r in rows}
        shared=by_id.get("d8_k16_rate_shared");conditional=by_id.get("d8_k16_rate_conditioned")
        if shared and conditional:
            lines += ["", "### 开销观察", "",
                      f"- 8维/16中心强制条件模型比共享模型少 {shared['payload_bytes']-conditional['payload_bytes']} bytes 数据流，"
                      f"但多 {conditional['metadata_bytes']-shared['metadata_bytes']} bytes 元数据，完整产物反而增加 {conditional['total_bytes']-shared['total_bytes']} bytes。",
                      "- Hamming码字VQ与原数值VQ之间同时改变了训练域、中心形式、码本精度和训练采样数量，不能把其全部差异仅归因于packed码本。",
                      "- 增加中心可以减小纠错流，但会增加索引和模型成本；本批最佳配置更偏向少中心。"]
    update_section(EXPERIMENT/"report.md","\n".join(lines))
    merge_json("validation.json",{"completed_variants":len(rows),"verified_method_chunk_pairs":sum(r['verified_pairs'] for r in rows),
                    "identical_source_hashes":True,"original_numeric_vq_unchanged":numeric is not None,
                    "original_arithmetic_unchanged":control is not None,"costs_reconciled":True,
                    "best_on_sample":best["variant"] if best else None,"source_sha256_by_chunk":reference})


def run(plan):
    env={**os.environ,"PYTHONPATH":str(ROOT/"src"),"FP4_VQ_THREADS":"1","FP4_C_THREADS":"1","FP4_D_THREADS":"1"}
    def one(variant):
        output=output_for(variant);output.parent.mkdir(parents=True,exist_ok=True)
        before=time.monotonic();print(json.dumps({"start":variant["id"],"utc":_now()}),flush=True)
        with tempfile.TemporaryDirectory(prefix="fp4-vq-selection-") as temporary:
            selection=Path(temporary)/"selection.json";_atomic_json(selection,plan["selection"])
            command=[sys.executable,"-m","fp4_redundancy","run",plan["snapshot"],"--output",str(output),
                     "--selection",str(selection),"--chunk-bytes","65536","--methods",variant["method"],
                     "--workers",str(plan["workers_per_variant"]),"--method-parameters",json.dumps({variant["method"]:variant["parameters"]})]
            if (output/"run.json").exists():command.append("--resume")
            with (output.parent/(variant["id"]+".log")).open("a") as f:
                outcome=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        value={"exit_code":outcome.returncode,"wall_seconds":time.monotonic()-before,"finished_utc":_now()}
        print(json.dumps({"finished":variant["id"],**value}),flush=True);return value
    execution={}
    with ThreadPoolExecutor(max_workers=plan["parallel_variants"]) as pool:
        futures={pool.submit(one,v):v["id"] for v in plan["variants"]}
        for future in as_completed(futures):
            execution[futures[future]]=future.result();merge_json("execution.json",execution);report(plan)
    if any(v["exit_code"] for v in execution.values()):raise RuntimeError("A VQ experiment failed; see method logs")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=("plan","run","report"));args=parser.parse_args()
    if args.command=="plan":prepare()
    else:
        plan=json.loads(PLAN.read_text())[KEY];(run if args.command=="run" else report)(plan)
