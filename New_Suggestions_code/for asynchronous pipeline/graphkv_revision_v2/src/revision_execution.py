"""Bounded real background execution, one preparation in flight per event.

Lead time is a controlled arrival schedule, NOT latency subtracted after the
fact. CPU submission is prioritized; GPU preemption is NOT promised.
"""
import threading, time
from concurrent.futures import ThreadPoolExecutor

def execute_event(foreground,background,current_ids,candidate_ids,target_ids,mode,lead_ms,
                  stop_on_demand=True):
    if mode not in ('none','sync','async'): raise ValueError(mode)
    if lead_ms<0: raise ValueError('lead_ms must be nonnegative')
    origin=time.perf_counter_ns(); stop=threading.Event(); records=[]
    demand_started=None
    def populate():
        for cid,ids in candidate_ids:
            if stop.is_set(): break
            start=time.perf_counter_ns()
            r=background.completion(ids)
            end=time.perf_counter_ns()
            records.append({'chunk_id':cid,'start_ns':start,'end_ns':end,
                            'request_ms':r.request_e2e_ms,'prompt_tokens':len(ids),
                            'prompt_hash':r.prompt_hash})
    # Independent clients/sessions avoid cross-thread Session use.
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(populate) if mode=='async' else None
        current=foreground.completion(current_ids)
        arrival=time.perf_counter_ns()+int(lead_ms*1e6)
        if mode=='sync': populate()
        remaining=(arrival-time.perf_counter_ns())/1e9
        if remaining>0: time.sleep(remaining)
        if stop_on_demand: stop.set()
        demand_started=time.perf_counter_ns()
        target=foreground.completion(target_ids)
        target_done=time.perf_counter_ns()
        stop.set()
        if future: future.result() # Drain AFTER timing, before next event and checkpoint.
    drained=time.perf_counter_ns()
    return {'current_request_ms':current.request_e2e_ms,
            'target_service_ttft_ms':target.ttft_ms,
            'arrival_to_first_token_ms':(demand_started-arrival)/1e6+target.ttft_ms,
            'arrival_to_complete_ms':(target_done-arrival)/1e6,
            'application_to_first_token_ms':(demand_started-origin)/1e6+target.ttft_ms,
            'application_request_ms':(target_done-origin)/1e6,
            'cycle_including_drain_ms':(drained-origin)/1e6,
            'post_target_drain_ms':(drained-target_done)/1e6,
            'current_start_ns':origin,'target_arrival_ns':arrival,
            'target_submit_ns':demand_started,'target_end_ns':target_done,
            'completed_before_arrival':[r['chunk_id'] for r in records if r['end_ns']<=arrival],
            'completed_before_submit':[r['chunk_id'] for r in records if r['end_ns']<=demand_started],
            'population':records,'target_prompt_hash':target.prompt_hash,
            'target_output_hash':target.output_text_hash}
