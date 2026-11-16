"""Finite two-model batch: wait for idle whole cards, run concurrently, stop own services."""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from run_generation import ROOT, read, dump, digest, run_lock, NoRedirect
from service_expansion08 import BASE, idle_gpus, alive


def ready(port, model):
    try:
        with urllib.request.build_opener(NoRedirect()).open('http://127.0.0.1:%d/v1/models' % port, timeout=2) as r:
            return model in {m['id'] for m in json.load(r)['data']}
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--allow-private-data', action='store_true')
    a = p.parse_args()
    if not a.allow_private_data or os.name != 'posix':
        raise ValueError('Explicit server private-source batch required')
    folder = ROOT / 'runs/workflow11_queue_01'
    protocol = read(ROOT / 'workflow11_protocol.json')
    with run_lock(folder / 'run.lock'):
        if (folder / 'state.json').exists():
            raise ValueError('Audit existing queue before restart; never duplicate a live job')
        plan = {'protocol_sha256': digest(ROOT / 'workflow11_protocol.json'), 'models': protocol['models'],
                'max_parallel_models': 2, 'dataset_integrity_sha256': protocol['dataset_integrity_sha256'],
                'code_sha256': {n: digest(ROOT / n) for n in ('queue_workflow11.py', 'service_workflow11.py', 'run_workflow11.py', 'workflow11_runtime.py')}}
        dump(folder / 'plan.json', plan)
        records, active, pending = [], {}, list(protocol['models'])
        log = (folder / 'workers.log').open('a')
        try:
            while pending or active:
                idle = [g for g in idle_gpus() if g not in {x['gpu'] for x in active.values()}]
                while pending and idle:
                    model, gpu = pending.pop(0), idle.pop(0)
                    idx = protocol['models'].index(model)
                    rec = {'model': model, 'gpu': gpu, 'port': 18500 + idx,
                           'tag': 'complete120_q9b_01' if idx == 0 else 'complete120_q27b_01',
                           'status': 'loading', 'started': time.time()}
                    cmd = [sys.executable, str(ROOT / 'service_workflow11.py'), '--action', 'start', '--model', model,
                           '--gpu', str(gpu), '--port', str(rec['port']), '--tag', rec['tag']]
                    start = subprocess.run(cmd, capture_output=True, text=True)
                    if start.returncode:
                        # GPU admission races are not medical results. Preserve
                        # an auditable wait and retry admission, never inference.
                        rec.update(status='admission_failed_no_inference', error=start.stderr[-1000:])
                        records.append(rec)
                        pending.insert(0, model)
                        break
                    rec['service_pid'] = read(BASE / ('workflow11_' + rec['tag'] + '.service.json'))['pid']
                    records.append(rec)
                    active[model] = rec
                    print(json.dumps({'event': 'workflow11_model_loading', 'model': model, 'gpu': gpu}), flush=True)
                for model, rec in list(active.items()):
                    if rec['status'] == 'loading':
                        if ready(rec['port'], model):
                            cmd = [sys.executable, '-u', str(ROOT / 'run_workflow11.py'), '--model', model,
                                   '--port', str(rec['port']), '--allow-private-data']
                            proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                                    stderr=subprocess.STDOUT, start_new_session=True)
                            rec.update(status='running', worker_pid=proc.pid)
                            rec['_process'] = proc
                        elif not alive(rec['service_pid']) or time.time() - rec['started'] > 1200:
                            rec['status'] = 'service_load_failed'
                    elif rec['status'] == 'running' and rec['_process'].poll() is not None:
                        code = rec['_process'].returncode
                        run = ROOT / 'runs' / ('workflow11_q9b_01' if model == 'Qwen3.5-9B' else 'workflow11_q27b_01')
                        rec.update(worker_exit_code=code, status='complete' if code == 0 and (run / 'complete.json').exists() else 'worker_failed')
                    if rec['status'] not in ('loading', 'running'):
                        stop = subprocess.run([sys.executable, str(ROOT / 'service_workflow11.py'), '--action', 'stop',
                            '--model', model, '--gpu', str(rec['gpu']), '--port', str(rec['port']), '--tag', rec['tag']], capture_output=True, text=True)
                        rec.update(stopped_at=time.time(), service_stop_exit_code=stop.returncode, service_stop_status=stop.stdout[-1000:])
                        rec.pop('_process', None)
                        del active[model]
                        print(json.dumps({'event': 'workflow11_model_accounted', 'model': model, 'status': rec['status']}), flush=True)
                state = {'jobs': [{k: v for k, v in r.items() if not k.startswith('_')} for r in records],
                         'pending_models': pending, 'idle_cards_now': idle_gpus(), 'updated': time.time(),
                         'waiting_for_whole_idle_gpu': bool(pending), 'all_accounted': not pending and not active}
                dump(folder / 'state.json', state)
                if pending or active:
                    time.sleep(15)
            dump(folder / 'complete.json', {'all_accounted': True, 'models_complete': sum(r['status'] == 'complete' for r in records),
                'models_planned': 2, 'failed_models': {r['model']: r['status'] for r in records if r['status'] != 'complete' and r['status'] != 'admission_failed_no_inference'},
                'owned_service_stop_attempted': True})
        finally:
            log.close()


if __name__ == '__main__':
    main()
