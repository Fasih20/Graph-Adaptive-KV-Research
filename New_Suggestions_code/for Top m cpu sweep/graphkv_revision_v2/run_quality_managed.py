#!/usr/bin/env python3
"""Prepare on CPU, launch a plain vLLM server, evaluate, then clean up.
Pass run_quality_revision.py arguments plus --gpu. This intentionally has NO
KV-transfer connector. Local answer quality is separate from cache latency.
"""
import argparse,os,signal,socket,subprocess,sys,time
from pathlib import Path
import requests

def main():
    p=argparse.ArgumentParser(add_help=False); p.add_argument('--gpu',default='0')
    control,args=p.parse_known_args()
    if '--prepare-only' in args or '--evaluate-saved-contexts' in args:
        raise SystemExit('Managed wrapper controls preparation/evaluation flags.')
    def value(flag,default=None):
        return args[args.index(flag)+1] if flag in args else default
    root=Path(__file__).resolve().parent
    subprocess.run([sys.executable,str(root/'run_quality_revision.py'),*args,'--prepare-only'],check=True)
    model=value('--model','Qwen/Qwen2.5-1.5B-Instruct'); base=value('--base-url','http://127.0.0.1:18000')
    from urllib.parse import urlparse
    parsed=urlparse(base)
    if parsed.hostname!='127.0.0.1': raise SystemExit('Managed server binds to 127.0.0.1 only')
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        probe.bind(('127.0.0.1',parsed.port or 18000))
    out=Path(value('--output-dir'))
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=control.gpu
    # Do not inherit a connector from external launch configuration.
    command=['vllm','serve',model,'--host','127.0.0.1','--port',str(parsed.port or 18000),
             '--tensor-parallel-size','1','--pipeline-parallel-size','1',
             '--max-model-len',value('--max-model-len','4096'),'--gpu-memory-utilization','0.82',
             '--no-enable-prefix-caching','--enforce-eager','--max-num-seqs','1']
    import json
    (out/'quality_launch.json').write_text(json.dumps({'command':command,'gpu':control.gpu,'kv_transfer':None},indent=2))
    with (out/'quality_server.log').open('w') as log:
        process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            deadline=time.monotonic()+900
            while True:
                if process.poll() is not None: raise RuntimeError('vLLM exited; inspect quality_server.log')
                try:
                    if requests.get(base+'/health',timeout=2).ok: break
                except requests.RequestException: pass
                if time.monotonic()>deadline: raise TimeoutError('vLLM startup timeout')
                time.sleep(1)
            subprocess.run([sys.executable,str(root/'run_quality_revision.py'),*args,'--evaluate-saved-contexts'],check=True)
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(30)
                except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL); process.wait()

if __name__=='__main__': main()
