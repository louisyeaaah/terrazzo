"""A/B: conservative prefix reuse (auto) vs calibrated certificate (certified).

Both servers prewarm the same 2k system prefix; only --prefix-reuse changes.
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
def start(model,port,policy):
    log=open(f'/tmp/cert_ab_{policy}.log','wb')
    cmd=[PY,'-m','terrazzo','serve','--model',model,'--host','127.0.0.1','--port',str(port),
         '--profile','agent','--no-thinking','--prefix-reuse',policy,
         '--completion-batch-size','16','--prefill-batch-size','8','--prefill-step-size','512']
    p=subprocess.Popen(cmd,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,
                       env={**os.environ,'TERRAZZO_ADMIT_SETTLE_MS':'50'})
    base=f'http://127.0.0.1:{port}'
    if not wait(base):
        p.kill(); log.close(); raise RuntimeError('start failed '+policy)
    return {'proc':p,'log':log,'base':base,'model':'terrazzo'}
def prewarm(srv,system):
    req=urllib.request.Request(srv['base']+'/v1/prewarm',data=json.dumps({'system':system}).encode(),
                               headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=900) as r: return json.loads(r.read().decode())
def stats(srv):
    with urllib.request.urlopen(srv['base']+'/stats',timeout=10) as r: return json.loads(r.read().decode())
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
    ap.add_argument('--port-base',type=int,default=9190)
    ap.add_argument('--out',default=str(ROOT/'bench'/'certificate_ab_results.json'))
    args=ap.parse_args(); model=str(Path(args.model).resolve())
    systems,_=A.build_systems('shared',args.sessions,args.system_tokens)
    shared=systems[0] if systems else ''
    rows=[]
    for i,policy in enumerate(['auto','certified']):
        srv=start(model,args.port_base+i,policy)
        try:
            pw=prewarm(srv,shared)
            A.chat_stream(srv['base'],'terrazzo',[{'role':'user','content':'warm up'}],8)
            t0=time.perf_counter()
            w=A.run_workload(srv, systems,args.turns,args.max_tokens,
                             user_template='请总结规范要点 {session}（第 {turn} 轮）')
            wall=time.perf_counter()-t0
            st=stats(srv)
            rows.append({'policy':policy,'prewarm_s':pw.get('elapsed_s'),
                         'wall_s':round(wall,2),'tok_s':w['aggregate_tok_s'],
                         'ttft_p50_s':w['ttft_p50_s'],'ttft_p95_s':w['ttft_p95_s'],
                         'completion_p95_s':w['total_p95_s'],'tokens':w['output_tokens'],
                         'prefix_cache':st.get('prefix_cache'),
                         'last_cost':st.get('cost_model',{}).get('last_cost'),
                         'prefill_calibration_skips':st.get('cost_model',{}).get('prefill_calibration_skips'),
                         'loadavg':list(os.getloadavg())})
        finally:
            stop(srv)
    Path(args.out).write_text(json.dumps({'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                          'args':vars(args),'rows':rows},indent=1))
    print(json.dumps(rows,indent=1))
if __name__=='__main__': main()
