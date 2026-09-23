"""Read-only follow-up to GraphKV's failed numerical gate.

Two vLLM launches: native prefix caching off, then on. No LMCache server or
connector. Uses the original prompts/model/runtime options and compares results
to the saved gate. Does not alter the gate or authorize benchmark execution.
"""
import argparse
import json
import math
import socket
import sys
from pathlib import Path


def difference(left, right):
    same = left.get('tokens') == right.get('tokens') and bool(left.get('tokens'))
    a, b = left.get('logprobs'), right.get('logprobs')
    comparable = same and a and b and len(a) == len(b) and all(
        x is not None and y is not None and math.isfinite(x) and math.isfinite(y)
        for x, y in zip(a, b))
    return {'same_tokens': bool(same), 'same_prompt': left.get('prompt_hash') == right.get('prompt_hash'),
            'max_abs_logprob_difference': max(abs(x-y) for x,y in zip(a,b)) if comparable else None}


def native_command(original, cache):
    command = list(original)
    index = command.index('--kv-transfer-config')
    del command[index:index+2]
    if cache:
        command[command.index('--no-enable-prefix-caching')] = '--enable-prefix-caching'
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path('/kaggle/working/graphkv_revision_v2'))
    parser.add_argument('--output-dir', type=Path, default=Path('/kaggle/working/graphkv_v2_hotpot'))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        sample = {'tokens':['a'], 'logprobs':[-.5], 'prompt_hash':'x'}
        assert difference(sample,sample)['max_abs_logprob_difference'] == 0
        assert difference(sample,{'tokens':['b']})['max_abs_logprob_difference'] is None
        command = ['vllm','serve','m','--kv-transfer-config','{}','--no-enable-prefix-caching']
        assert native_command(command,True) == ['vllm','serve','m','--enable-prefix-caching']
        assert command[3] == '--kv-transfer-config'
        print('Control logic checks passed. GPU execution still required.')
        return
    sys.path.insert(0, str(args.project/'src'))
    from isolated_runtime import IsolatedRuntime, RuntimeConfig, _wait_http
    from runtime_env import child_environment, collect_cli_help
    from revision_gate import environment_key, exact_ids, client_for, sample
    from revision_io import atomic, digest
    from real_cache_client import parse_prometheus, metric_delta
    import requests
    from transformers import AutoTokenizer

    gate_path = args.output_dir/'gate/reuse_gate.json'
    gate = json.loads(gate_path.read_text())
    rc = RuntimeConfig(**gate['environment']['runtime'])
    if environment_key(rc) != gate['environment']:
        raise RuntimeError('Environment/code differs from saved gate. Keep original files and package versions for this comparison.')
    conditions = gate['conditions']
    if len(conditions) != 3:
        raise RuntimeError('Expected three completed gate cases')
    root = args.output_dir/'native_cache_control'
    if (root/'control_report.json').exists():
        raise RuntimeError('Control report already exists. Download/review it before starting another control run.')
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(rc.model, revision=rc.model_revision)
    prompts = [
        'Aster station reports that the unique launch code is cobalt. '*24+'The launch code is',
        'Birch archive lists the inventor as Mira Chen and the year as 1982. '*24+'The inventor is',
        'Cedar observatory measures the altitude in meters using a calibrated instrument. '*24+'The unit is',
    ]
    ids = [exact_ids(tokenizer,text) for text in prompts]
    for tokens, case in zip(ids,conditions):
        if digest(tokens) != case['cold']['prompt_hash']:
            raise RuntimeError('Prompt mismatch; stop rather than compare different inputs')
    report = {'status':'running','environment':gate['environment'],'gate_sha256':digest(gate),
              'scope':'diagnostic only; gate remains unchanged; no performance claims',
              'note':'Retains original runtime options, including disabled chunked prefill, to isolate caching effects.',
              'runs':{}}
    atomic(root/'control_report.json',report)

    class NativeRuntime(IsolatedRuntime):
        def start_native(self, enabled):
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                probe.bind((self.config.vllm_host,self.config.vllm_port))
            help_text = collect_cli_help(['vllm','serve'], exhaustive=True)
            command = native_command(self._vllm_command(help_text),enabled)
            env = child_environment(self.config.gpu_id)
            process, log = self._spawn('vllm', command, env)
            atomic(self.run_dir/'launch_commands.json',{'vllm':command,'lmcache':None})
            _wait_http(f'http://{rc.vllm_host}:{rc.vllm_port}/health', process, log, rc.startup_timeout_s)

    def metrics():
        response = requests.get(f'http://{rc.vllm_host}:{rc.vllm_port}/metrics',timeout=10)
        response.raise_for_status()
        return parse_prometheus(response.text)

    try:
        for enabled in (False,True):
            name = 'native_cache_on' if enabled else 'native_cache_off'
            print(f'Starting {name} (one GPU, no LMCache)...',flush=True)
            rows=[]
            with NativeRuntime(rc,root/name) as server:
                server.start_native(enabled)
                client=client_for(rc,tokenizer)
                try:
                    for index,tokens in enumerate(ids):
                        first = sample(client,tokens)
                        repeats=[]
                        for repeat in range(3):
                            before=metrics(); observed=sample(client,tokens); after=metrics()
                            delta=metric_delta(before,after)
                            evidence={k:v for k,v in delta.items() if 'prefix_cache' in k.lower()}
                            hits=[v for k,v in delta.items() if k.split('{',1)[0] == 'vllm:prefix_cache_hits_total']
                            repeats.append({'sample':observed,'vs_first':difference(first,observed),
                                'vs_lmcache_warm':difference(conditions[index]['warm'],observed),
                                'native_prefix_hit_tokens':sum(hits) if hits else None,
                                'prefix_metrics_delta':evidence})
                        rows.append({'case':index,'first':first,'vs_original_cold':difference(conditions[index]['cold'],first),
                                     'repeats':repeats})
                        report['runs'][name]=rows; atomic(root/'control_report.json',report)
                        print(name,'case',index,':',json.dumps(repeats[0]['vs_first']),
                              'native_hit_tokens=',repeats[0]['native_prefix_hit_tokens'],flush=True)
                finally:
                    client.close()
        report['status']='complete_needs_review'
    except BaseException as exc:
        report['status']='error'; report['error']=str(exc)
        raise
    finally:
        atomic(root/'control_report.json',report)
    print('Saved:',root/'control_report.json',flush=True)
    print('The original gate is unchanged. Send this report for review before benchmarking.',flush=True)


if __name__ == '__main__':
    main()
