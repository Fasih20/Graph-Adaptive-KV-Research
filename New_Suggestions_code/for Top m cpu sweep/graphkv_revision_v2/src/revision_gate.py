"""Fail-closed exact-prefix and shifted-prefix GPU correctness gate.

No segment splicing, no CacheBlend claims. Compare cold reference with
independently populated exact-prefix and shifted-prefix requests.
"""
import hashlib, importlib.metadata, json, math, time, uuid
from dataclasses import asdict
from pathlib import Path
import requests
from isolated_runtime import IsolatedRuntime
from real_cache_client import RealCacheClient
from sanity_controls import _evidence_since, _log_retrieved_tokens, _wait_lmcache_quiescent
from revision_io import atomic, digest

def environment_key(runtime):
    versions={}
    for name in ('vllm','lmcache','torch','transformers'):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]='missing'
    source=Path(__file__).parent
    code={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in
          ('revision_gate.py','real_cache_client.py','isolated_runtime.py')}
    return {'runtime':asdict(runtime),'versions':versions,'gate_protocol':2,'code':code}

def client_for(runtime,tokenizer):
    return RealCacheClient(tokenizer=tokenizer,model=runtime.model,
        vllm_base_url=f'http://{runtime.vllm_host}:{runtime.vllm_port}',
        lmcache_http_url=f'http://{runtime.lmcache_host}:{runtime.lmcache_http_port}',
        lmcache_metrics_url=f'http://{runtime.lmcache_host}:{runtime.lmcache_prometheus_port}/metrics',
        blend_separator=runtime.blend_separator,gpu_id=runtime.gpu_id)

def exact_ids(tokenizer,text):
    # Same construction for population and demand; no independent truncation defaults.
    bos=[] if tokenizer.bos_token_id is None else [int(tokenizer.bos_token_id)]
    return bos+list(map(int,tokenizer.encode(text,add_special_tokens=False)))

def sample(client,ids):
    r=requests.post(client.vllm_base_url+'/v1/completions',json={
        'model':client.model,'prompt':ids,'temperature':0,'seed':43,
        'max_tokens':8,'min_tokens':8,'ignore_eos':True,'logprobs':5,
        'skip_special_tokens':False,'stream':False},timeout=180)
    r.raise_for_status(); item=r.json()
    if 'error' in item: raise RuntimeError(str(item['error']))
    c=item['choices'][0]; lp=c.get('logprobs') or {}
    return {'text':c['text'],'tokens':lp.get('tokens'),
            'logprobs':lp.get('token_logprobs'),'usage':item.get('usage'),
            'prompt_hash':digest(ids),'prompt_tokens':len(ids)}

def equivalent(a,b,tolerance):
    if not a.get('tokens') or not b.get('tokens'): return False
    if a['tokens']!=b['tokens']: return False
    x,y=a.get('logprobs'),b.get('logprobs')
    return bool(x and y and len(x)==len(y) and all(
        u is not None and v is not None and math.isfinite(u) and math.isfinite(v)
        and abs(u-v)<=tolerance for u,v in zip(x,y)))

def run_gate(root,runtime,tokenizer,tolerance=.001):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    report={'passed':False,'status':'running','environment':environment_key(runtime),
            'scope':'exact-prefix only; shifted chunks must miss; no CacheBlend',
            'logprob_tolerance':tolerance,'conditions':[]}
    atomic(root/'reuse_gate.json',report)
    prompts=[
        'Aster station reports that the unique launch code is cobalt. '*24+'The launch code is',
        'Birch archive lists the inventor as Mira Chen and the year as 1982. '*24+'The inventor is',
        'Cedar observatory measures the altitude in meters using a calibrated instrument. '*24+'The unit is',
    ]
    ids=[exact_ids(tokenizer,t) for t in prompts]
    shifted=[exact_ids(tokenizer,'Different preceding article number '+str(i)+'. '*30+t)
             for i,t in enumerate(prompts)]
    if max(map(len,ids+shifted))+8>runtime.max_model_len: raise ValueError('Gate prompts exceed model context')
    try:
        # Two fresh processes per pair avoid earlier gate requests becoming cache prefixes.
        for index,(plain,moved) in enumerate(zip(ids,shifted)):
            cold_dir=root/f'case_{index}_cold'; warm_dir=root/f'case_{index}_warm'
            with IsolatedRuntime(runtime,cold_dir) as server:
                server.start('gate-cold-'+uuid.uuid4().hex[:8]); c=client_for(runtime,tokenizer)
                try:
                    shifted_ref=sample(c,moved); ref=sample(c,plain)
                finally: c.close()
            with IsolatedRuntime(runtime,warm_dir) as server:
                server.start('gate-warm-'+uuid.uuid4().hex[:8]); c=client_for(runtime,tokenizer)
                try:
                    c.completion(plain); _wait_lmcache_quiescent(c)
                    offset,_=_evidence_since(warm_dir/'lmcache.log',0)
                    actual=sample(c,plain); _wait_lmcache_quiescent(c)
                    offset,logs=_evidence_since(warm_dir/'lmcache.log',offset)
                    reused=_log_retrieved_tokens(logs)
                    moved_actual=sample(c,moved); _wait_lmcache_quiescent(c)
                    _,shift_logs=_evidence_since(warm_dir/'lmcache.log',offset)
                    shift_reused=_log_retrieved_tokens(shift_logs)
                finally: c.close()
            shared=0
            for x,y in zip(plain,moved):
                if x!=y: break
                shared+=1
            condition={'case':index,'cold':ref,'warm':actual,'shifted_cold':shifted_ref,
                       'shifted_after_population':moved_actual,'retrieved_tokens':reused,
                       'shifted_retrieved_tokens':shift_reused,'shared_prefix_tokens':shared,
                       'retrieval_evidence':logs,'shifted_evidence':shift_logs}
            condition['passed']=bool(reused>0 and shift_reused<=shared
                and equivalent(ref,actual,tolerance) and equivalent(shifted_ref,moved_actual,tolerance))
            report['conditions'].append(condition); atomic(root/'reuse_gate.json',report)
            print(f'Cache gate {index+1}/3: {condition["passed"]}',flush=True)
        report['passed']=all(c['passed'] for c in report['conditions'])
        report['status']='passed' if report['passed'] else 'failed_or_inconclusive'
    except Exception as exc:
        report['status']='error'; report['error']=str(exc); raise
    finally: atomic(root/'reuse_gate.json',report)
    return report

def require_gate(path,runtime):
    report=json.loads(Path(path).read_text())
    if not report.get('passed') or report.get('environment')!=environment_key(runtime):
        raise RuntimeError('Dedicated reuse gate missing/failed/stale. Run gate on this exact environment first.')
    return digest(report['environment'])
