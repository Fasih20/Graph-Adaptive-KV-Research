#!/usr/bin/env python3
"""Local vLLM answer quality. Full prompt prefill, never spliced independent KV.

Shares V2 scoring with the systems ablations. Sentence nodes for Hotpot/2Wiki;
paragraph nodes for MuSiQue. Not a systems-latency experiment.
"""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import requests
from revision_io import atomic,digest,fingerprint,Backup
from revision_policy import Policy,edges,NAMES
from graphkv_quality.data import load_structured_sample,nodes_from_row
from graphkv_quality.graph import build_document_graph,predict_fixed,predict_adaptive
from graphkv_quality.scoring import exact_match,max_token_f1
from graphkv_quality.pipeline import PROMPT_TEMPLATE,equal_reference_token_contexts
from types import SimpleNamespace

def musique_row(raw):
    documents=[]; support=[]
    for p in raw['paragraphs']:
        title=f"{p['idx']}: {p['title']}"
        documents.append({'title':title,'sentences':[p['paragraph_text']]})
        if p['is_supporting']: support.append([title,0])
    return {'id':str(raw['id']),'question':raw['question'],
            'answers':[raw['answer'],*raw.get('answer_aliases',[])],
            'documents':documents,'support_pairs':support}

def sample_rows(dataset,split,count,seed,cache):
    if dataset!='musique': return load_structured_sample(dataset,split,count,seed,cache)
    path=cache/f'musique_{split}_{count}_{seed}.json'
    if path.exists(): return json.loads(path.read_text())
    from datasets import load_dataset
    stream=load_dataset('dgslibisey/MuSiQue',split=split,streaming=True).shuffle(seed=seed,buffer_size=10000)
    rows=[]
    for r in stream:
        if r.get('answerable',True): rows.append(musique_row(r))
        if len(rows)==count: break
    if len(rows)!=count: raise ValueError('Insufficient labeled MuSiQue rows')
    atomic(path,rows); return rows

def prepare_case(row,embedder,m):
    nodes,support=nodes_from_row(row)
    if len(nodes)<3 or not support: raise ValueError('Insufficient nodes/support: '+row['id'])
    emb=embedder.encode([row['question']]+[n.text for n in nodes],normalize_embeddings=True,
                        convert_to_numpy=True,show_progress_bar=False)
    sim=emb[1:]@emb[1:].T; qs=emb[1:]@emb[0]; primary=int(np.argmax(qs))
    docs=[n.document_id for n in nodes]; positions=[n.position for n in nodes]
    raw=edges(sim,docs,positions,m,True); sem=edges(sim,docs,positions,m,False)
    target=max(support-{primary} or support,key=lambda i:(qs[i],-i))
    adjacency,oldraw=build_document_graph(sim,nodes,top_m=m,fixed_weights=(.7,.3),max_degree=100)
    return SimpleNamespace(row=row,nodes=nodes,support_ids=support,similarity=sim,raw=raw,sem=sem,
                           primary=primary,target=target,oldadj=adjacency,oldraw=oldraw)

def prediction(c,name,k,weights,eta):
    p=Policy(name,c.sem if name=='semantic_only' else c.raw,c.similarity,weights,eta)
    return p,p.predict(c.primary,k)

def fitting(train,dev,k):
    grid=[]
    for a in np.linspace(0,1,21):
        recalls=[]
        for c in train:
            _,p=prediction(c,'offline',k,(a,1-a),.01)
            recalls.append(len(({c.primary}|set(p.ids))&c.support_ids)/len(c.support_ids))
        grid.append({'semantic':float(a),'support_recall':float(np.mean(recalls))})
    best=max(grid,key=lambda r:(r['support_recall'],-abs(r['semantic']-.7),-r['semantic']))
    weights=[best['semantic'],1-best['semantic']]; ets=[]
    for eta in (.01,.05,.1,.2,.5,1.):
        state=weights; recalls=[]
        for c in dev:
            p,pred=prediction(c,'online',k,state,eta)
            recalls.append(len(({c.primary}|set(pred.ids))&c.support_ids)/len(c.support_ids))
            p.observe(c.target,pred); state=p.weights.tolist()
        ets.append({'eta':eta,'support_recall':float(np.mean(recalls))})
    eta=max(ets,key=lambda r:(r['support_recall'],-abs(r['eta']-.1)))['eta']
    return {'weights':weights,'eta':eta,'train_grid':grid,'dev_grid':ets}

def generate(base,model,token_ids,max_tokens):
    tick=time.perf_counter_ns()
    r=requests.post(base.rstrip('/')+'/v1/completions',json={'model':model,'prompt':token_ids,
                    'temperature':0,'seed':43,'max_tokens':max_tokens,'stream':False},timeout=180)
    r.raise_for_status(); data=r.json()
    if 'error' in data: raise RuntimeError(str(data['error']))
    choice=data['choices'][0]
    return {'text':choice['text'],'finish_reason':choice['finish_reason'],
            'usage':data.get('usage'),'request_ms':(time.perf_counter_ns()-tick)/1e6}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['hotpot','2wiki','musique'],default='hotpot')
    p.add_argument('--model',default='Qwen/Qwen2.5-1.5B-Instruct')
    p.add_argument('--base-url',default='http://127.0.0.1:18000')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--questions',type=int,default=60); p.add_argument('--train',type=int,default=120)
    p.add_argument('--dev',type=int,default=40); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--top-m',type=int,default=5); p.add_argument('--k',type=int,default=10)
    p.add_argument('--budget',type=int,default=768); p.add_argument('--max-tokens',type=int,default=64)
    p.add_argument('--max-model-len',type=int,default=4096)
    p.add_argument('--backup-remote',default=os.getenv('GRAPHKV_BACKUP_REMOTE'))
    p.add_argument('--backup-every',type=int,default=5)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--evaluate-saved-contexts',action='store_true')
    p.add_argument('--include-legacy-fixed',action='store_true',help='Also generate answers with old fixed ranking to quantify F1 drift')
    a=p.parse_args()
    if min(a.questions,a.train,a.dev,a.top_m,a.k,a.budget,a.max_tokens,a.backup_every)<1: raise ValueError('Positive counts required')
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer
    tokenizer=AutoTokenizer.from_pretrained(a.model)
    spec={k:v for k,v in vars(a).items() if k not in ('backup_remote','output_dir','base_url','prepare_only','evaluate_saved_contexts')}
    spec['version']='quality-v2'; spec['tokenizer_revision']=tokenizer.init_kwargs.get('_commit_hash')
    from run_revision import source_hash
    spec['source']=source_hash(); fingerprint(a.output_dir,spec)
    backup=Backup(a.output_dir,a.backup_remote)
    prepared=a.output_dir/'prepared_quality.json'
    if not prepared.exists():
        if a.evaluate_saved_contexts: raise ValueError('Run --prepare-only first')
        cache=a.output_dir/'data'
        training=sample_rows(a.dataset,'train',a.train+a.dev,a.seed,cache)
        testing=sample_rows(a.dataset,'validation',a.questions,a.seed+1,cache)
        if set(r['id'] for r in training)&set(r['id'] for r in testing): raise ValueError('Question split overlap')
        if len(set(r['id'] for r in training+testing))!=len(training)+len(testing): raise ValueError('Duplicate question IDs')
        embedder=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
        cases=[prepare_case(r,embedder,a.top_m) for r in training+testing]
        fitted=fitting(cases[:a.train],cases[a.train:a.train+a.dev],a.k)
        tests=cases[a.train+a.dev:]; state=fitted['weights']; rows=[]; drift=[]
        names=[n for n in NAMES if n!='no_prefetch']
        if a.include_legacy_fixed: names.append('legacy_fixed')
        for qi,c in enumerate(tests):
            predictions={}; policies={}; selected={}
            for name in names:
                if name=='legacy_fixed':
                    pred=predict_fixed(c.oldadj,c.primary,a.k); policy=None
                else: policy,pred=prediction(c,name,a.k,state if name=='online' else fitted['weights'],fitted['eta'])
                predictions[name]=pred; policies[name]=policy
                selected[name]=list(dict.fromkeys([c.primary,*pred.ids]))
            contexts,budget,delivered=equal_reference_token_contexts(c,selected,tokenizer,a.budget)
            legacy=predict_fixed(c.oldadj,c.primary,a.k).ids
            drift.append({'question_id':c.row['id'],'old_fixed_ids':legacy,'new_fixed_ids':predictions['two_hop'].ids,
                          'changed':legacy!=predictions['two_hop'].ids})
            for name in names:
                text=PROMPT_TEMPLATE.format(context=contexts[name],question=c.row['question'])
                # Chat template is used identically across policies, before tokenization.
                ids=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True)
                if len(ids)+a.max_tokens>a.max_model_len: raise ValueError('Quality prompt exceeds context; reduce budget')
                rows.append({'question_id':c.row['id'],'question_index':qi,'policy':name,'k':a.k,
                    'answers':c.row['answers'],'selected_ids':selected[name],'delivered_ids':delivered[name],
                    'context_tokens':budget,'prompt_ids':ids,'prompt_hash':digest(ids),
                    'support_recall':len(set(delivered[name])&c.support_ids)/len(c.support_ids),
                    'target_hit':c.target in delivered[name],
                    'feedback_semantics':'gold support target observed only after current prediction; oracle-supervised replay'})
            online=policies['online']; online.observe(c.target,predictions['online']); state=online.weights.tolist()
        atomic(a.output_dir/'quality_policy_drift.json',{'rows':drift,'changed':sum(r['changed'] for r in drift),
            'note':'Prediction changes only. F1 changes require --include-legacy-fixed generation.'})
        atomic(prepared,{'rows':rows,'fit':fitted,'scope':'full-prompt answer quality; no independent-segment KV reuse'})
        backup.save('quality-prepared')
    if a.prepare_only: print('Quality inputs prepared. Start vLLM and use --evaluate-saved-contexts.'); return
    backup.save('quality-start')
    statepath=a.output_dir/'quality_progress.json'
    done=json.loads(statepath.read_text()) if statepath.exists() else {}
    rows=json.loads(prepared.read_text())['rows']
    for index,row in enumerate(rows):
        key=f"{row['question_id']}:{row['policy']}"
        if key in done: continue
        answer=generate(a.base_url,a.model,row['prompt_ids'],a.max_tokens)
        done[key]={**{k:v for k,v in row.items() if k!='prompt_ids'},**answer,
                   'em':exact_match(answer['text'],row['answers']),
                   'f1':max_token_f1(answer['text'],row['answers'])}
        atomic(statepath,done)
        print(f'{len(done)}/{len(rows)} quality answers saved',flush=True)
        if len(done)%a.backup_every==0: backup.save('quality-progress')
    import pandas as pd
    df=pd.DataFrame(done.values()); df.to_csv(a.output_dir/'quality_events.csv',index=False)
    table=df.groupby('policy')[['em','f1','support_recall','target_hit']].mean()
    table['N']=df.groupby('policy').size(); table.to_csv(a.output_dir/'quality_summary.csv')
    # Paired question bootstrap for fixed policies; online sequence dependence flagged explicitly.
    deltas=[]; rng=np.random.default_rng(43)
    for metric in ('em','f1','support_recall'):
        wide=df.pivot(index='question_id',columns='policy',values=metric)
        for name in wide:
            if name=='cosine': continue
            values=(wide[name]-wide.cosine).dropna().to_numpy()
            draws=rng.choice(values,(2000,len(values)),replace=True).mean(axis=1)
            lo,hi=np.quantile(draws,[.025,.975])
            deltas.append({'policy':name,'metric':metric,'mean_delta':float(values.mean()),
                'ci_low':None if name=='online' else float(lo),'ci_high':None if name=='online' else float(hi),
                'note':'online CI withheld: one dependent sequence' if name=='online' else 'paired question bootstrap; shared-document dependence not modeled'})
    atomic(a.output_dir/'quality_paired_deltas.json',deltas)
    atomic(a.output_dir/'COMPLETE.json',{'answers':len(done),'expected':len(rows),'fingerprint':digest(spec)})
    backup.save('quality-complete'); print(table.to_string())

if __name__=='__main__': main()
