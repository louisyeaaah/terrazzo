"""Small executor-batch sweep: prefill batch vs prefill chunk.

Admission cap (16), completion batch (16), scheduling and no-prefix-cache are
fixed; only BatchGenerator's prefill_batch_size / prefill_step_size change.

    python bench/batch_sweep.py --model .hf-models/qwen3.5-2b
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'bench'))
import co_design_ablation as A
PY=sys.executable
def wait(base):
    for _ in range(180):
        try:
            with urllib.request.urlopen(base+'/health',timeout=2) as r:
                if r.status==200:return True
        except Exception: pass
        time.sleep(1)
    return False
def start(model,port,pbatch,step,max_seqs=None):
    log=open(f'/tmp/batch_sweep_{pbatch}_{step}.log','wb')
    cmd=[PY,'-m','terrazzo','serve','--model',model,'--host','127.0.0.1','--port',str(port),
         '--profile','agent','--no-thinking','--no-prefix-cache','--scheduler-policy','fifo',
         '--completion-batch-size','16','--prefill-batch-size',str(pbatch),
         '--prefill-step-size',str(step)]
    if max_seqs is not None:
        cmd += ['--max-num-seqs', str(max_seqs)]
    # Do not inherit experiment toggles such as TERRAZZO_AUTO_CAP from the
    # caller: a fixed-cap run must actually be fixed.
    env={k: v for k, v in os.environ.items() if not k.startswith("TERRAZZO_")}
    env.update({'TERRAZZO_ADMIT_SETTLE_MS': '50',
                'TERRAZZO_STACK_DUMP': '1',
                'TERRAZZO_AUTO_CAP': '0'})
    p=subprocess.Popen(cmd,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,env=env)
    base=f'http://127.0.0.1:{port}'
    if not wait(base):
        p.kill(); log.close(); raise RuntimeError(f'start failed {pbatch}/{step}')
    return {'proc':p,'log':log,'base':base,'model':'terrazzo'}
def stop(srv):
    try: srv['proc'].terminate(); srv['proc'].wait(timeout=10)
    except Exception:
        try: srv['proc'].kill()
        except Exception: pass
    try: srv['log'].close()
    except Exception: pass
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='.hf-models/qwen3.5-2b')
    ap.add_argument('--sessions',type=int,default=16)
    ap.add_argument('--turns',type=int,default=2)
    ap.add_argument('--max-tokens',type=int,default=64)
    ap.add_argument('--system-tokens',type=int,default=2000)
    ap.add_argument('--port-base',type=int,default=9130)
    ap.add_argument('--max-num-seqs',type=int,default=None)
    ap.add_argument('--grid',default='8:512,4:512,16:512,8:256,8:1024,4:256')
    ap.add_argument('--out',default=str(ROOT/'bench'/'batch_sweep_results.json'))
    ap.add_argument('--expected-completion-batch',type=int,default=16)
    args=ap.parse_args(); model=str(Path(args.model).resolve())
    systems,shared=A.build_systems('shared',args.sessions,args.system_tokens)
    rows=[]
    for i,item in enumerate(args.grid.split(',')):
        pb,step=[int(x) for x in item.split(':')]
        port=args.port_base+i
        print(f'[batch] prefill_batch={pb} step={step} port={port}',flush=True)
        srv=start(model,port,pb,step,args.max_num_seqs)
        try:
            A.chat_stream(srv['base'],'terrazzo',[{'role':'user','content':'warm up'}],8)
            t0=time.perf_counter()
            s=A.run_workload(srv,systems,args.turns,args.max_tokens,
                             user_template='请总结规范要点 {session}（第 {turn} 轮）')
            wall=time.perf_counter()-t0
            with urllib.request.urlopen(srv['base']+'/stats', timeout=10) as r:
                stats=json.loads(r.read().decode())
            executor=stats.get('executor') or {}
            auto=stats.get('auto_concurrency') or {}
            effective={'max_num_seqs':stats.get('max_num_seqs'),
                       'executor':executor,'auto_concurrency_enabled':auto.get('enabled')}
            if args.max_num_seqs is not None and stats.get('max_num_seqs') != args.max_num_seqs:
                raise RuntimeError(f"cap mismatch: requested {args.max_num_seqs}, "
                                   f"effective {stats.get('max_num_seqs')}")
            if executor.get('completion_batch_size') != args.expected_completion_batch:
                raise RuntimeError(f"completion batch mismatch: {executor}")
            if executor.get('prefill_batch_size') != pb or executor.get('prefill_step_size') != step:
                raise RuntimeError(f"prefill config mismatch: {executor}")
            if auto.get('enabled') is not False:
                raise RuntimeError(f"auto_concurrency not explicitly disabled: {auto}")
            rows.append({'prefill_batch':pb,'prefill_step':step,
                         'max_num_seqs':args.max_num_seqs,'sessions':args.sessions,
                         'effective':effective,
                         'wall_s':round(wall,2),'aggregate_tok_s':s['aggregate_tok_s'],
                         'ttft_p50_s':s['ttft_p50_s'],'ttft_p95_s':s['ttft_p95_s'],
                         'completion_p95_s':s['total_p95_s'],'tokens':s['output_tokens'],
                         'mlx_peak_mb':s['mlx_peak_mb'],'loadavg':list(os.getloadavg())})
            print(f"  tok/s={s['aggregate_tok_s']} ttft={s['ttft_p50_s']} p95={s['total_p95_s']} peak={s['mlx_peak_mb']}",flush=True)
        finally:
            stop(srv)
    out={'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'model':model,
         'args':vars(args),'rows':rows}
    Path(args.out).write_text(json.dumps(out,indent=1))
    print('\n| prefill batch | step | tok/s | TTFT p50 | TTFT p95 | completion p95 | peak MB |')
    print('|---:|---:|---:|---:|---:|---:|---:|')
    for r in rows:
        print(f"| {r['prefill_batch']} | {r['prefill_step']} | {r['aggregate_tok_s']} | {r['ttft_p50_s']} | {r['ttft_p95_s']} | {r['completion_p95_s']} | {r['mlx_peak_mb']} |")
    print('wrote',args.out)
if __name__=='__main__': main()
