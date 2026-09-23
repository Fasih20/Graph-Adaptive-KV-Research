#!/usr/bin/env python3
"""Audit completed archived arms without rerunning inference or modifying them."""
import argparse,json,sys,zipfile
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from revision_io import atomic

def audit(archive):
    groups=defaultdict(list); count=0; retrieval=0; semantics=set()
    with zipfile.ZipFile(archive) as z:
        for path in z.namelist():
            if not path.endswith('/COMPLETE.json') or '/arms/' not in path: continue
            marker=json.loads(z.read(path)); rel=marker['events_path']
            prefix=path.split('/arms/',1)[0]
            rows=[json.loads(line) for line in z.read(prefix+'/'+rel).splitlines() if line.strip()]
            if len(rows)!=marker['events']: raise ValueError('Incomplete archived arm')
            for r in rows:
                count+=1; semantics.add(r.get('cache_semantics'))
                retrieval+=int((r.get('lmcache_retrieved_tokens_delta') or 0)>0)
                groups[(r['repetition'],r['event_id'])].append(r)
    mismatches=[]; prompt_mismatches=[]
    for key,rows in groups.items():
        if len({r['prompt_token_hash'] for r in rows})!=1: prompt_mismatches.append(key)
        if len({r['output_text_hash'] for r in rows})!=1:
            mismatches.append({'key':key,'variants':[{k:r.get(k) for k in
                ('policy','k','output_text_hash','lmcache_retrieved_tokens_delta')} for r in rows]})
    return {'events':count,'paired_event_groups':len(groups),'groups_with_prompt_mismatch':len(prompt_mismatches),
            'groups_with_output_mismatch':len(mismatches),'requests_with_retrieval_evidence':retrieval,
            'cache_semantics':list(semantics),'mismatches':mismatches,
            'verdict_scope':'Saved one-token output hashes and prompt parity only; not new numerical GPU gate; not full RAG validation',
            'recall_note':'Prediction recall is computed from predicted IDs and targets, independent of cache numerical correctness.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('archive'); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); report=audit(a.archive); atomic(a.output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='mismatches'},indent=2))
