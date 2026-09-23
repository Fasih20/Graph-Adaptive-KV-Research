#!/usr/bin/env python3
"""V2: prepare -> drift -> gate -> run -> summary. No silent gate override."""
import argparse, csv, hashlib, json, logging, os, sys, time, uuid
from dataclasses import asdict, fields
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from adaptive_policies import TraceEvent, OfflineAdaptivePolicy, OnlineAdaptivePolicy
from isolated_runtime import RuntimeConfig, IsolatedRuntime
from revision_io import atomic, digest, fingerprint, Backup
from revision_policy import Policy, edges, fit, NAMES
from revision_gate import exact_ids,client_for,run_gate,environment_key
from native_evidence_gate import require_gate
from revision_execution import execute_event

def source_hash():
    return digest({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(ROOT.rglob('*.py')) if '__pycache__' not in str(p)})

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','drift','gate','run','summary'])
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--model',default='Qwen/Qwen2.5-1.5B-Instruct')
    p.add_argument('--dataset',choices=['hotpot','2wiki','musique','multifield'],default='hotpot')
    p.add_argument('--gpu',default='0'); p.add_argument('--seed',type=int,default=43)
    p.add_argument('--events',type=int,default=60); p.add_argument('--train-events',type=int,default=120)
    p.add_argument('--dev-events',type=int,default=60); p.add_argument('--max-documents',type=int,default=30)
    p.add_argument('--top-m',type=int,nargs='+',default=[20])
    p.add_argument('--k',type=int,nargs='+',default=[6])
    p.add_argument('--policies',nargs='+',choices=NAMES,default=list(NAMES))
    p.add_argument('--modes',nargs='+',choices=['sync','async'],default=['async'])
    p.add_argument('--lead-ms',nargs='+',type=float,default=[0])
    p.add_argument('--block-events',type=int,default=30)
    p.add_argument('--repetitions',type=int,default=1)
    p.add_argument('--max-arms',type=int,default=64,help='Refuse larger jobs before any server starts')
    p.add_argument('--backup-every',type=int,default=5)
    p.add_argument('--backup-remote',default=os.getenv('GRAPHKV_BACKUP_REMOTE'))
    p.add_argument('--port-base',type=int,default=18000)
    p.add_argument('--l1-gb',type=float,default=.5)
    p.add_argument('--max-model-len',type=int,default=4096)
    return p

def prepare(a):
    from transformers import AutoConfig,AutoTokenizer
    from chunk_store import ChunkStore
    root=a.output_dir; root.mkdir(parents=True,exist_ok=True)
    spec={k:getattr(a,k) for k in ('model','dataset','seed','events','train_events','dev_events','max_documents')}
    spec['source']=source_hash()
    path=root/'prepared.json'
    if path.exists():
        if json.loads(path.read_text())['spec']!=spec: raise ValueError('Prepared configuration mismatch; new output required')
        Backup(root,a.backup_remote).save('prepared-resume')
        print('Prepared data already exists'); return
    config=AutoConfig.from_pretrained(a.model)
    revision=getattr(config,'_commit_hash',None)
    tokenizer=AutoTokenizer.from_pretrained(a.model,revision=revision)
    store=ChunkStore(a.dataset,tokenizer=tokenizer,model_config=config,top_m=20,
        max_documents=a.max_documents,max_chars_per_document=16000,
        document_chunk_tokens=384,document_chunk_overlap_tokens=64,
        embedding_device='cpu',seed=a.seed).prepare()
    traces={name:store.generate_events(name,count,offset) for name,count,offset in
            [('train',a.train_events,0),('dev',a.dev_events,1),('test',a.events,2)]}
    store.save(root/'workload',traces)
    np.save(root/'similarity.npy',store.sim_matrix,allow_pickle=False)
    atomic(path,{'spec':spec,'resolved_revision':revision,
                 'trace_semantics':'controlled synthetic; not production RAG',
                 'boundary_semantics':'LongBench context record; not constituent article'})
    Backup(root,a.backup_remote).save('prepared')

def load(a):
    root=a.output_dir
    meta=json.loads((root/'prepared.json').read_text())
    chunks=[json.loads(x) for x in (root/'workload/chunks.jsonl').read_text().splitlines()]
    sim=np.load(root/'similarity.npy',allow_pickle=False)
    traces={}
    for split in ('train','dev','test'):
        with (root/f'workload/trace_{split}.csv').open() as f:
            traces[split]=[TraceEvent(**{k:int(v) if k in ('event_id','step_id','current_chunk_id','target_chunk_id') else v
                                        for k,v in row.items()}) for row in csv.DictReader(f)]
    return meta,chunks,sim,traces

def runtime(a,meta):
    b=a.port_base
    return RuntimeConfig(model=meta['spec']['model'],model_revision=meta['resolved_revision'],gpu_id=a.gpu,
        vllm_host='127.0.0.1',vllm_port=b,lmcache_host='127.0.0.1',lmcache_port=b+1,
        lmcache_http_port=b+2,lmcache_prometheus_port=b+3,max_model_len=a.max_model_len,
        block_size=16,chunk_size=16,gpu_memory_utilization=.82,l1_size_gb=a.l1_gb,
        startup_timeout_s=900,seed=a.seed,max_num_seqs=2)

def graph_for(chunks,sim,m,structure=True):
    return edges(sim,[c['document_id'] for c in chunks],[c['position'] for c in chunks],m,structure)

def drift(a,meta,chunks,sim,traces):
    from document_graph import build_graph_document_aware
    from graph_algorithms import get_prefetch_graph
    report=[]; fits={}
    for m in a.top_m:
        raw=graph_for(chunks,sim,m); fitted=fit(traces['train'],traces['dev'],raw,sim,a.k)
        fits[str(m)]=fitted
        old_adj,old_raw,_=build_graph_document_aware(np.zeros((len(chunks),1)),sim,
            [c['document_id'] for c in chunks],[c['position'] for c in chunks],logging.getLogger(),top_m=m)
        for k in a.k:
            # Frozen weights isolate implementation drift; refitting is reported separately.
            for variant,weights in [('fixed',(.7,.3)),('offline',fitted['weights'])]:
                old=OfflineAdaptivePolicy(old_raw,weights)
                new=Policy('offline',raw,sim,weights)
                rows=[]
                for e in traces['test']:
                    before=old.predict(e.current_chunk_id,k).ids; after=new.predict(e.current_chunk_id,k).ids
                    historical=get_prefetch_graph(old_adj,e.current_chunk_id,k) if variant=='fixed' else before
                    rows.append({'event_id':e.event_id,'old':before,'new':after,'historical_fixed':historical,
                                 'old_hit':e.target_chunk_id in before,'new_hit':e.target_chunk_id in after})
                report.append({'m':m,'k':k,'variant':variant,'weights':list(weights),
                    'changed_ordered_predictions':sum(r['old']!=r['new'] for r in rows),
                    'historical_fixed_changed':sum(list(r['historical_fixed'])!=r['new'] for r in rows),
                    'old_recall':float(np.mean([r['old_hit'] for r in rows])),
                    'new_recall':float(np.mean([r['new_hit'] for r in rows]))})
                atomic(a.output_dir/f'drift/M{m}_K{k}_{variant}.json',rows)
    atomic(a.output_dir/'drift_report.json',{'comparisons':report,
        'scope':'CPU prediction replay on prepared workload, fixed weights; NOT generated-answer F1',
        'warning':'Exact submitted-number attribution requires original embeddings/trace/code and saved event comparison.'})
    atomic(a.output_dir/'fitted_revision.json',fits)
    print(json.dumps(report,indent=2))

def run(a,meta,chunks,sim,traces,rc):
    gate_hash=require_gate(a.output_dir/'gate/reuse_gate.json',rc)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(rc.model,revision=rc.model_revision)
    ids=[exact_ids(tokenizer,c['text']) for c in chunks]
    if max(map(len,ids))+1>rc.max_model_len: raise ValueError('Chunk exceeds context')
    spec={k:v for k,v in vars(a).items() if k not in ('command','backup_remote','backup_every','output_dir')}
    spec.update(source=source_hash(),gate=gate_hash,prepared=meta,environment=environment_key(rc))
    # Reject changed configuration before launching any server.
    fingerprint(a.output_dir,spec)
    backup=Backup(a.output_dir,a.backup_remote)
    fitted={}; graphs={}
    for m in a.top_m:
        raw=graph_for(chunks,sim,m); graphs[(m,True)]=raw; graphs[(m,False)]=graph_for(chunks,sim,m,False)
        fitted[m]=fit(traces['train'],traces['dev'],raw,sim,a.k)
    atomic(a.output_dir/'fitted_revision.json',fitted)
    jobs=[]
    # Independent sequence blocks reset both cache and online weights by design.
    for rep in range(a.repetitions):
        for m in a.top_m:
            for k in a.k:
                for name in a.policies:
                    # Controls do not depend on M; no-prefetch also does not depend on K.
                    if name in ('no_prefetch','cosine') and m!=a.top_m[0]: continue
                    if name=='no_prefetch' and k!=a.k[0]: continue
                    for mode in (['none'] if name=='no_prefetch' else a.modes):
                        for lead in a.lead_ms:
                            for start in range(0,len(traces['test']),a.block_events):
                                jobs.append((rep,m,k,name,mode,lead,start))
    np.random.default_rng(a.seed).shuffle(jobs)
    if len(jobs)>a.max_arms:
        raise ValueError(f'{len(jobs)} isolated blocks exceeds --max-arms={a.max_arms}; reduce matrix or explicitly raise budget')
    print(f'Planned: {len(jobs)} isolated blocks, {sum(min(a.block_events,len(traces["test"])-j[-1]) for j in jobs)} events. Each block launches fresh servers.',flush=True)
    atomic(a.output_dir/'execution_order.json',jobs)
    backup.save('start')
    for job_number,(rep,m,k,name,mode,lead,start) in enumerate(jobs,1):
        key=f'r{rep}_M{m}_K{k}_{name}_{mode}_L{lead:g}_b{start}'
        block=a.output_dir/'blocks'/key; done=block/'COMPLETE.json'
        if done.exists():
            saved=json.loads(done.read_text())
            data=json.loads((block/'events.json').read_text())
            if saved['events_hash']!=digest(data): raise RuntimeError('Corrupt completed block '+key)
            continue
        policy=Policy(name,graphs[(m,name!='semantic_only')],sim,fitted[m]['weights'],fitted[m]['eta'])
        attempt=block/('attempt-'+uuid.uuid4().hex[:8]); attempt.mkdir(parents=True,exist_ok=True)
        rows=[]; events=traces['test'][start:start+a.block_events]
        print(f'Block {job_number}/{len(jobs)}: {key}; {len(events)} events',flush=True)
        try:
            with IsolatedRuntime(rc,attempt) as server:
                server.start(key); fg=client_for(rc,tokenizer); bg=client_for(rc,tokenizer)
                try:
                    for i,e in enumerate(events):
                        tick=time.perf_counter_ns(); pred=policy.predict(e.current_chunk_id,k)
                        policy_ms=(time.perf_counter_ns()-tick)/1e6
                        result=execute_event(fg,bg,ids[e.current_chunk_id],
                            [(j,ids[j]) for j in pred.ids],ids[e.target_chunk_id],mode,lead)
                        update=policy.observe(e.target_chunk_id,pred)
                        row={**asdict(e),**result,'rep':rep,'m':m,'k':k,'policy':name,'mode':mode,
                             'lead_ms':lead,'block_start':start,'predicted_ids':pred.ids,
                             'prediction_hit':int(e.target_chunk_id in pred.ids),'policy_ms':policy_ms,
                             'application_with_policy_ms':result['application_request_ms']+policy_ms,
                             'update':update,'state_after':policy.state_dict(),
                             'target_prepared_before_arrival':e.target_chunk_id in result['completed_before_arrival'],
                             'prefetch_kv_bytes_estimate':sum(chunks[r['chunk_id']]['kv_bytes'] for r in result['population']),
                             'backend_cache_hit':None,
                             'metric_note':'cache attribution unavailable under concurrency; byte estimate is not measured transfer'}
                        rows.append(row); atomic(attempt/'progress.json',{'events':rows,'next_index':i+1})
                        if (i+1)%a.backup_every==0:
                            print(f'  {i+1}/{len(events)} events; current TTFT={result["arrival_to_first_token_ms"]:.2f} ms',flush=True)
                            backup.save(key+'-progress')
                    atomic(block/'events.json',rows)
                    atomic(done,{'events_hash':digest(rows),'events':len(rows),'config':digest(spec),
                                 'resume_semantics':'complete independent block; incomplete block reruns from cold'})
                finally: fg.close(); bg.close()
            backup.save(key+'-complete')
        except BaseException:
            backup.save(key+'-interrupted')
            raise
    summary(a.output_dir)
    atomic(a.output_dir/'COMPLETE.json',{'blocks':len(jobs),'fingerprint':digest(spec)})
    backup.save('complete')

def summary(root):
    import pandas as pd
    rows=[]
    for p in root.glob('blocks/*/COMPLETE.json'):
        data=json.loads((p.parent/'events.json').read_text())
        if json.loads(p.read_text())['events_hash']!=digest(data): raise ValueError('Corrupt block')
        rows.extend(data)
    if not rows: raise ValueError('No complete blocks')
    frame=pd.DataFrame(rows)
    # Make the overlap/load conditions explicit in the aggregate output.  The
    # event records already contained these fields, but the original pilot
    # summary hid them and therefore could not show *why* a lead time did or
    # did not help.
    frame['completed_before_arrival_count']=frame.completed_before_arrival.map(len)
    frame['population_completed_count']=frame.population.map(len)
    keys=['policy','m','k','mode','lead_ms']
    metrics=['prediction_hit','arrival_to_first_token_ms','arrival_to_complete_ms',
             'target_service_ttft_ms','application_to_first_token_ms',
             'application_with_policy_ms','cycle_including_drain_ms',
             'post_target_drain_ms','current_request_ms',
             'target_prepared_before_arrival','completed_before_arrival_count',
             'population_completed_count','prefetch_kv_bytes_estimate']
    table=frame.groupby(keys)[metrics].mean().reset_index()
    table['N']=frame.groupby(keys).size().values
    table['p95_ttft_ms']=frame.groupby(keys).arrival_to_first_token_ms.quantile(.95).values
    table['foreground_requests_per_second']=2000/table['cycle_including_drain_ms']
    table.to_csv(root/'headline_results.csv',index=False)
    # Paired bootstrap over independent blocks, averaging repeated executions first.
    deltas=[]; rng=np.random.default_rng(43)
    for group,part in frame.groupby(keys):
        name,m,k,mode,lead=group
        if name=='no_prefetch': continue
        for control in ('no_prefetch','cosine'):
            baseline=frame[(frame.policy==control)&(frame.lead_ms==lead)]
            if control=='cosine': baseline=baseline[(baseline.k==k)&(baseline['mode']==mode)]
            for metric in ['arrival_to_first_token_ms','target_service_ttft_ms',
                           'application_to_first_token_ms','application_with_policy_ms',
                           'cycle_including_drain_ms']:
                left=part.groupby(['block_start','event_id'])[metric].mean()
                right=baseline.groupby(['block_start','event_id'])[metric].mean()
                paired=(left-right).dropna(); blockmeans=paired.groupby(level=0).mean().to_numpy()
                if len(blockmeans)<2: low=high=None
                else:
                    draws=rng.choice(blockmeans,(2000,len(blockmeans)),replace=True).mean(axis=1)
                    low,high=map(float,np.quantile(draws,[.025,.975]))
                deltas.append(dict(zip(keys,group))|{'control':control,'metric':metric,'paired_events':len(paired),
                    'independent_blocks':len(blockmeans),'mean_delta':float(np.mean(blockmeans)),
                    'ci_low':low,'ci_high':high,'estimand':'equal-block mean; descriptive if blocks share source records'})
    pd.DataFrame(deltas).to_csv(root/'paired_deltas.csv',index=False)
    print(table.to_string(index=False))

def main():
    a=parser().parse_args(); logging.basicConfig(level=logging.INFO)
    if min(a.events,a.train_events,a.dev_events,a.block_events,a.repetitions,a.backup_every,*a.k,*a.top_m)<1:
        raise ValueError('Counts must be positive')
    if min(a.lead_ms)<0: raise ValueError('Lead time must be nonnegative')
    if a.command=='prepare': return prepare(a)
    if a.command=='summary': return summary(a.output_dir)
    meta,chunks,sim,traces=load(a)
    if a.command=='drift': return drift(a,meta,chunks,sim,traces)
    rc=runtime(a,meta)
    if a.command=='gate':
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(rc.model,revision=rc.model_revision)
        result=run_gate(a.output_dir/'gate',rc,tokenizer)
        Backup(a.output_dir,a.backup_remote).save('gate')
        if not result['passed']: raise SystemExit('Reuse gate failed or inconclusive. Do not benchmark yet.')
    else: run(a,meta,chunks,sim,traces,rc)

if __name__=='__main__': main()
