import json, sys, time, threading, urllib.request
from pathlib import Path
sys.path.insert(0,'bench')
import co_design_ablation as A
MODEL='/Users/yezhipeng/project/dsh-other/deepinfer-study/.hf-cache/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25'
system = A.build_systems('shared',1,2000)[1]
A_suffix='请用一句话总结规范一。'
B_suffix='请用一句话总结规范二。'

def post(base,msgs,rid,mt=32,stream=False):
    body={'model':'terrazzo','messages':msgs,'max_tokens':mt,'temperature':0,'stream':stream,'request_id':rid}
    req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    if stream:
        return urllib.request.urlopen(req,timeout=120)
    with urllib.request.urlopen(req,timeout=120) as r: return json.loads(r.read().decode())

def ask(base,rid,suffix,mt=32):
    msgs=[{'role':'system','content':system},{'role':'user','content':suffix}]
    d=post(base,msgs,rid,mt,False); ch=d['choices'][0]
    return {'rid':rid,'finish':ch['finish_reason'],'out':d.get('usage',{}).get('completion_tokens'),
            'text':ch['message']['content'],'usage':d.get('usage')}

def wait(base):
    for _ in range(180):
        try:
            with urllib.request.urlopen(base+'/health',timeout=2) as r:
                if r.status==200:return True
        except Exception:pass
        time.sleep(1)
    return False

cache=A.start_server('b',MODEL,9100,system,Path('/tmp/cache_correctness_cache.log'))
nocache=A.start_server('a',MODEL,9101,None,Path('/tmp/cache_correctness_nocache.log'))
try:
    cb, nb = cache['base'], nocache['base']
    print('prewarm_s',cache['prewarm_s'])
    seq={}
    for name,suf in [('A',A_suffix),('B',B_suffix)]:
        c=ask(cb,'cache-'+name,suf); n=ask(nb,'nocache-'+name,suf)
        seq[name]=(c,n)
        print(name,'T',c['finish'],c['out'],repr(c['text'][:100]))
        print(name,'N',n['finish'],n['out'],repr(n['text'][:100]))
        print(name,'MATCH',c['text']==n['text'])
    # cache A again after B (snapshot immutability + no cross-branch contamination)
    a2=ask(cb,'cache-A2',A_suffix)
    print('A after B MATCH',a2['text']==seq['A'][0]['text'],'finish',a2['finish'])
    # concurrent two branches
    def concurrent(tag_prefix, base):
        results={}
        def run(tag,suf):
            results[tag]=ask(base,tag,suf)
        ts=[threading.Thread(target=run,args=(f'{tag_prefix}{i}',A_suffix if i%2==0 else B_suffix)) for i in range(4)]
        [t.start() for t in ts]; [t.join() for t in ts]
        return results
    cc=concurrent('cc',cb); nn=concurrent('nn',nb)
    print('concurrent cached vs no-cache')
    for i in range(4):
        kc=f'cc{i}'; kn=f'nn{i}'; suf='A' if i%2==0 else 'B'
        ref=seq[suf][0]['text']
        print(kc,suf,'out',cc[kc]['out'],'match_serial',cc[kc]['text']==ref,'match_nocache',cc[kc]['text']==nn[kn]['text'])
        print('  C',repr(cc[kc]['text'][:100]))
        print('  N',repr(nn[kn]['text'][:100]))
        print('  S',repr(ref[:100]))

    def ask_msgs(base,rid,msgs,mt=32):
        d=post(base,msgs,rid,mt,False); ch=d['choices'][0]
        return {'finish':ch['finish_reason'],'out':d.get('usage',{}).get('completion_tokens'),
                'text':ch['message']['content']}
    def multi(base,rid,suf,follow):
        msgs=[{'role':'system','content':system},{'role':'user','content':suf}]
        r1=ask_msgs(base,rid+'-1',msgs)
        msgs.append({'role':'assistant','content':r1['text']})
        msgs.append({'role':'user','content':follow})
        r2=ask_msgs(base,rid+'-2',msgs)
        return r1,r2
    for suf,name in [(A_suffix,'A'),(B_suffix,'B')]:
        cr1,cr2=multi(cb,'mt-'+name,suf,'请再补充一句。'); nr1,nr2=multi(nb,'mt-'+name,suf,'请再补充一句。')
        print('multi',name,'turn1 match',cr1['text']==nr1['text'],'turn2 match',cr2['text']==nr2['text'],
              'lens',cr1['out'],cr2['out'],'inputs',cr1.get('in'),cr2.get('in'))
        print('  T2',repr(cr2['text'][:100]))
    # abort a long request, then verify cache still serves A correctly
    rid='cancel-me'
    stream=post(cb,[{'role':'system','content':system},{'role':'user','content':'写一段很长的文本'}],rid,256,True)
    first=stream.readline(); print('stream first',first[:80])
    ab=urllib.request.Request(cb+'/v1/abort',data=json.dumps({'request_id':rid}).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(ab,timeout=10) as r: print('abort',r.read().decode())
    stream.close()
    time.sleep(0.2)
    a3=ask(cb,'cache-A3',A_suffix)
    print('A after abort MATCH',a3['text']==seq['A'][0]['text'],'finish',a3['finish'])
    with urllib.request.urlopen(cb+'/stats',timeout=5) as r:
        st=json.loads(r.read().decode())
    print('stats cache',json.dumps(st.get('prefix_cache') or st.get('cost_model') or {},indent=1)[:600])
finally:
    A.stop_server(cache); A.stop_server(nocache)
