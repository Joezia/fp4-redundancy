"""Plot committed aggregate results only; no checkpoint or codec execution."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'experiments/grouped-units'


def main():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit('Optional plotting dependency missing: install matplotlib') from error
    rows=json.loads((DIRECTORY/'results.json').read_text())
    indexed={(r['method'],r['unit_bits']):r for r in rows}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    payload_color='#2474a6';metadata_color='#e5a23d'
    methods=['arithmetic','ctw','paq','fpc_like_fp4','rle','vq_codeword','dct_lossless','svd_lowrank']
    selected=[indexed[(m,4)] for m in methods]
    if any(r['status']!='verified' for r in selected):raise ValueError('Overview needs verified configurations')
    fig,ax=plt.subplots(figsize=(10,5.5))
    for y,r in enumerate(selected):
        p=r['fp4_payload_bytes']/r['fp4_original_bytes'];m=r['fp4_metadata_bytes']/r['fp4_original_bytes']
        ax.barh(y,p,color=payload_color,label='Payload' if y==0 else None)
        ax.barh(y,m,left=p,color=metadata_color,label='Metadata' if y==0 else None)
        ax.text(p+m+.06,y,f"total saving {r['fp4_saving_fraction']:.2%}",va='center',fontsize=9)
    ax.set_yticks(range(len(methods)),methods);ax.invert_yaxis();ax.axvline(1,color='#555555',linestyle='--',linewidth=1)
    ax.set_xlim(0,max(r['fp4_total_bytes']/r['fp4_original_bytes'] for r in selected)+1.5)
    ax.set_xlabel('Encoded bytes / original packed FP4 bytes (1 = original size)')
    ax.set_title('Selected methods, 4-bit units: payload and metadata costs\nSix complete expert matrices; FP4 weights only; all required costs included')
    ax.legend(loc='lower right');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(DIRECTORY/'cost-overview.png',dpi=150);plt.close(fig)

    widths=[4,8,16,32,64];fig,axes=plt.subplots(1,2,figsize=(11,4.7),sharey=True)
    for ax,method in zip(axes,['arithmetic','rans']):
        for x,b in enumerate(widths):
            r=indexed.get((method,b))
            if r is None or r['status']!='verified':
                label='outside scope' if r is None else r['status']
                ax.text(x,.08,label.replace('_','\n'),ha='center',va='bottom',fontsize=8,color='#555555');continue
            p=r['fp4_payload_bytes']/r['fp4_original_bytes'];m=r['fp4_metadata_bytes']/r['fp4_original_bytes']
            ax.bar(x,p,color=payload_color,label='Payload' if x==0 else None)
            ax.bar(x,m,bottom=p,color=metadata_color,label='Metadata' if x==0 else None)
            ax.text(x,p+m+.05,f'{p+m:.2f}x',ha='center',fontsize=9)
        ax.set_xticks(range(len(widths)),[str(b) for b in widths]);ax.set_xlim(-.6,4.6);ax.set_xlabel('unit_bits');ax.set_title(method)
        ax.axhline(1,color='#555555',linestyle='--',linewidth=1);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.set_ylim(0,3.7)
    axes[0].set_ylabel('Encoded bytes / original packed FP4 bytes');axes[0].legend(loc='upper left')
    fig.suptitle('Wide-symbol entropy coding: same six matrices and outer chunks, FP4 only')
    fig.text(.5,.01,'Arithmetic 64-bit is a matched control. Missing/limited rANS cases have no size estimate.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(DIRECTORY/'entropy-unit-costs.png',dpi=150);plt.close(fig)
    print('Wrote cost-overview.png and entropy-unit-costs.png from existing results.json')


if __name__=='__main__':main()
