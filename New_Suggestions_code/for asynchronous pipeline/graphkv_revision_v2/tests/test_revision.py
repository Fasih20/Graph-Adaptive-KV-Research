import json,sys,tempfile,threading,time,unittest,zipfile,hashlib
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from adaptive_policies import OfflineAdaptivePolicy
from revision_policy import Policy,edges
from revision_execution import execute_event
from revision_gate import equivalent
from revision_io import atomic,fingerprint,restore,Backup
from unittest.mock import patch

class FakeClient:
    def __init__(self,delay=.003,started=None,release=None):
        self.delay=delay; self.started=started; self.release=release; self.calls=[]
    def completion(self,ids):
        self.calls.append(ids)
        if self.started: self.started.set()
        if self.release: self.release.wait(2)
        time.sleep(self.delay)
        return SimpleNamespace(request_e2e_ms=self.delay*1000,ttft_ms=1.,prompt_hash=str(ids),output_text_hash='x')

class RevisionTests(unittest.TestCase):
    def test_equal_weights_same_predictions(self):
        rng=np.random.default_rng(2); x=rng.normal(size=(30,8)); x/=np.linalg.norm(x,axis=1)[:,None]; sim=x@x.T
        raw=edges(sim,['a']*15+['b']*15,list(range(15))*2,5)
        for i in range(30):
            old=OfflineAdaptivePolicy(raw,(.7,.3)).predict(i,6).ids
            fixed=Policy('two_hop',raw,sim).predict(i,6).ids
            learned=Policy('offline',raw,sim,(.7,.3)).predict(i,6).ids
            self.assertEqual(old,fixed); self.assertEqual(fixed,learned)

    def test_hop_isolation(self):
        raw={0:[(1,.9,.5)],1:[(0,.9,.5),(2,.8,.5)],2:[(1,.8,.5)]}; sim=np.eye(3)
        self.assertNotIn(2,Policy('structure',raw,sim).predict(0,3).ids)
        self.assertIn(2,Policy('two_hop',raw,sim).predict(0,3).ids)

    def test_no_structural_cross_document(self):
        raw=edges(np.eye(4),['a','a','b','b'],[0,1,0,1],1)
        for i,neighbors in raw.items():
            for j,s,t in neighbors:
                if (i<2)!=(j<2): self.assertEqual(t,0)

    def test_async_demand_not_blocked_by_prefetch_drain(self):
        started=threading.Event(); release=threading.Event()
        bg=FakeClient(started=started,release=release)
        class Foreground(FakeClient):
            def completion(self,ids):
                if ids==[9]:
                    self.assertion=not release.is_set(); release.set()
                return super().completion(ids)
        fg=Foreground()
        result=execute_event(fg,bg,[0],[(1,[1]),(2,[2])],[9],'async',0)
        self.assertTrue(fg.assertion); self.assertEqual(len(bg.calls),1)
        self.assertGreaterEqual(result['post_target_drain_ms'],0)
        self.assertEqual(result['completed_before_arrival'],[])

    def test_sync_exposes_population_and_none_doesnt_run_it(self):
        fg=FakeClient(); bg=FakeClient(.02)
        sync=execute_event(fg,bg,[0],[(1,[1])],[2],'sync',0)
        self.assertGreater(sync['arrival_to_first_token_ms'],15)
        bg.calls=[]; execute_event(fg,bg,[0],[(1,[1])],[2],'none',0)
        self.assertEqual(bg.calls,[])

    def test_gate_rejects_numerical_and_missing_evidence(self):
        a={'tokens':['x'],'logprobs':[-.1]}; b={'tokens':['x'],'logprobs':[-.2]}
        self.assertFalse(equivalent(a,b,.001)); self.assertFalse(equivalent({}, {}, .001))
        self.assertTrue(equivalent(a,a,.001))

    def test_online_never_updates_on_hit_or_coverage_miss(self):
        raw={0:[(1,.8,.2)],1:[]}; p=Policy('online',raw,np.eye(2))
        pred=p.predict(0,1); self.assertFalse(p.observe(1,pred)['updated'])
        self.assertFalse(p.observe(7,pred)['updated'])

    def test_config_guard_preserves_previous(self):
        with tempfile.TemporaryDirectory() as td:
            fingerprint(td,{'x':1})
            with self.assertRaises(ValueError): fingerprint(td,{'x':2})
            self.assertEqual(json.loads((Path(td)/'revision_manifest.json').read_text())['config'],{'x':1})

    def test_snapshot_verification_and_restore(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); p=root/'s.zip'; data=b'{"events":5}'
            with zipfile.ZipFile(p,'w') as z:
                z.writestr('block/COMPLETE.json',data)
                z.writestr('BACKUP_HASHES.json',json.dumps({'block/COMPLETE.json':hashlib.sha256(data).hexdigest()}))
            restore(p,root/'out'); self.assertEqual((root/'out/block/COMPLETE.json').read_bytes(),data)
            with self.assertRaises(ValueError): restore(p,root/'out')

    def test_backup_failure_stops_without_deleting_progress(self):
        with tempfile.TemporaryDirectory() as td:
            atomic(Path(td)/'progress.json',{'events':[1]})
            with patch('revision_io.shutil.which',return_value='/rclone'),patch('revision_io.subprocess.run',return_value=SimpleNamespace(returncode=1)):
                with self.assertRaises(RuntimeError): Backup(td,'drive:test').save('test')
            self.assertTrue((Path(td)/'progress.json').exists())

    def test_musique_mapping(self):
        from run_quality_revision import musique_row
        r=musique_row({'id':'x','question':'q','answer':'a','paragraphs':[
            {'idx':3,'title':'A','paragraph_text':'t','is_supporting':True}]})
        self.assertEqual(r['support_pairs'],[['3: A',0]])

    def test_interrupted_block_restarts_and_completed_block_skips(self):
        import run_revision as runner
        from adaptive_policies import TraceEvent
        from isolated_runtime import RuntimeConfig
        import types
        class Runtime:
            starts=0
            def __init__(self,*args): pass
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def start(self,*args): Runtime.starts+=1
        class Tokenizer:
            bos_token_id=None
            def encode(self,text,**kw): return [int(text)]
        transformer=types.ModuleType('transformers')
        transformer.AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:Tokenizer())
        rc=RuntimeConfig('test',None,'0','127.0.0.1',18000,'127.0.0.1',18001,18002,18003,4096,16,16,.82,.5,900,43)
        events=[TraceEvent(i,'s',i,'d',0,1,'semantic') for i in range(2)]
        chunks=[{'text':str(i),'document_id':'d','position':i,'kv_bytes':10} for i in range(3)]
        calls=[]; fail=[True]
        original=runner.execute_event
        def event(*args,**kw):
            calls.append(1)
            if fail[0] and len(calls)==2: raise RuntimeError('simulated interruption')
            return original(*args,**kw)
        with tempfile.TemporaryDirectory() as td:
            a=runner.parser().parse_args(['run','--output-dir',td,'--policies','cosine','--block-events','2','--backup-every','1'])
            with patch.dict(sys.modules,{'transformers':transformer}),patch.object(runner,'require_gate',return_value='gate'),\
                 patch.object(runner,'IsolatedRuntime',Runtime),patch.object(runner,'client_for',side_effect=lambda *a:SimpleNamespace(completion=FakeClient().completion,close=lambda:None)),\
                 patch.object(runner,'summary'),patch.object(runner,'execute_event',side_effect=event):
                traces={'train':events,'dev':events,'test':events}
                with self.assertRaises(RuntimeError): runner.run(a,{},chunks,np.eye(3),traces,rc)
                self.assertFalse(list(Path(td).glob('blocks/*/COMPLETE.json')))
                self.assertTrue(list(Path(td).glob('blocks/*/attempt-*/progress.json')))
                fail[0]=False; runner.run(a,{},chunks,np.eye(3),traces,rc)
                self.assertEqual(len(calls),4) # both events rerun; not only the missing row
                before=Runtime.starts; runner.run(a,{},chunks,np.eye(3),traces,rc)
                self.assertEqual(Runtime.starts,before)

if __name__=='__main__': unittest.main()
