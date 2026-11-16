"""Same scientific run; revised user-authorized shared-resource admission only."""
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from run_generation import ROOT, read, dump, digest, run_lock
from service_expansion08 import BASE, alive
from service_workflow11_shared import gpu_snapshot, admitted
from queue_workflow11 import ready


def select_profile(model, reserved, amendment):
    snap = gpu_snapshot()
    candidates = [g for g in snap if g not in reserved and snap[g]['utilization_pct'] <= 10]
    candidates.sort(key=lambda g: (-snap[g]['free_mib'], g))
    options = amendment.get('q9b_options_in_order', [amendment['q9b']]) if model == 'Qwen3.5-9B' else amendment['q27b_options_in_order']
    for option in options:
        n = option['tensor_parallel'] * option['pipeline_parallel']
        for chosen in itertools.combinations(candidates, n):
            if admitted(chosen, option['gpu_memory_utilization'], [snap]):
                return list(chosen), option
    return None


def service_command(rec, action):
    return [sys.executable, str(ROOT / 'service_workflow11_shared.py'), '--action', action,
        '--model', rec['model'], '--gpus', ','.join(map(str, rec['gpus'])), '--tp', str(rec['tp']), '--pp', str(rec['pp']),
        '--fraction', str(rec['fraction']), '--port', str(rec['port']), '--tag', rec['tag']]


def main():
    if os.name != 'posix' or '--allow-private-data' not in sys.argv:
        raise ValueError('Explicit server execution required')
    folder = ROOT / 'runs/workflow11_queue_xml02'
    amendment = read(ROOT / 'workflow11_resource_amendment.json')
    protocol = read(ROOT / 'workflow11_protocol.json')
    with run_lock(folder / 'run.lock'):
        if (folder / 'state.json').exists() or (folder / 'resource_plan.json').exists():
            raise ValueError('Preserve existing corrected queue; audit before any resume')
        for name in ('workflow11_q9b_xml02', 'workflow11_q27b_xml02'):
            if (ROOT / 'runs' / name).exists():
                raise ValueError('Actor run already exists; audit before changing deployment')
        old_folder = ROOT / 'runs/workflow11_queue_01'
        dump(folder / 'broken_interface_queue_reference.json', read(old_folder / 'state.json'))
        plan = {'resource_amendment_sha256': digest(ROOT / 'workflow11_resource_amendment.json'),
            'protocol_sha256': digest(ROOT / 'workflow11_protocol.json'), 'original_queue_plan_sha256': digest(old_folder / 'plan.json'),
            'transport_erratum_sha256': digest(ROOT / 'workflow11_transport_erratum.json'),
            'code_sha256': {n: digest(ROOT / n) for n in ('queue_workflow11_shared.py', 'service_workflow11_shared.py')},
            'no_actor_requests_before_amendment': True, 'scientific_protocol_unchanged': True}
        dump(folder / 'resource_plan.json', plan)
        dump(folder / 'plan.json', plan)
        records, active, pending, attempts = [], {}, list(protocol['models']), {m: 0 for m in protocol['models']}
        with (folder / 'workers.log').open('a') as log:
            while pending or active:
                for model in list(pending):
                    reserved = {g for r in active.values() for g in r['gpus']}
                    choice = select_profile(model, reserved, amendment)
                    if choice is None:
                        continue
                    gpus, option = choice
                    idx = protocol['models'].index(model)
                    attempts[model] += 1
                    rec = {'model': model, 'gpu': gpus[0], 'gpus': gpus, 'tp': option['tensor_parallel'], 'pp': option['pipeline_parallel'],
                        'fraction': option['gpu_memory_utilization'], 'port': 18510 + idx,
                        'tag': ('xml02_q9b_' if idx == 0 else 'xml02_q27b_') + str(attempts[model]).zfill(2),
                        'status': 'loading', 'started': time.time()}
                    start = subprocess.run(service_command(rec, 'start'), capture_output=True, text=True)
                    if start.returncode:
                        rec.update(status='service_start_failed', error=start.stderr[-1800:])
                        records.append(rec)
                        pending.remove(model)
                        continue
                    result = json.loads(start.stdout)
                    if not result.get('admitted'):
                        continue
                    rec['service_pid'] = result['pid']
                    records.append(rec)
                    active[model] = rec
                    pending.remove(model)
                    print(json.dumps({'event': 'workflow11_shared_model_loading', 'model': model, 'gpus': gpus, 'tp': rec['tp'], 'pp': rec['pp']}), flush=True)
                for model, rec in list(active.items()):
                    if rec['status'] == 'loading':
                        if ready(rec['port'], model):
                            preflight = subprocess.run([sys.executable, str(ROOT / 'preflight_workflow11_xml.py'), '--model', model,
                                '--port', str(rec['port']), '--output', str((folder / ('preflight_' + str(rec['port']) + '.json')).relative_to(ROOT))],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                            rec['synthetic_preflight_exit_code'] = preflight.returncode
                            if preflight.returncode:
                                rec['status'] = 'synthetic_parser_preflight_failed'
                            else:
                                cmd = [sys.executable, '-u', str(ROOT / 'run_workflow11_xml02.py'), '--model', model, '--port', str(rec['port']), '--allow-private-data']
                                proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                                rec.update(status='running', worker_pid=proc.pid)
                                rec['_process'] = proc
                        elif not alive(rec['service_pid']) or time.time() - rec['started'] > 1200:
                            rec['status'] = 'service_load_failed'
                    elif rec['status'] == 'running' and rec['_process'].poll() is not None:
                        code = rec['_process'].returncode
                        run = ROOT / 'runs' / ('workflow11_q9b_xml02' if '9B' in model else 'workflow11_q27b_xml02')
                        rec.update(worker_exit_code=code, status='complete' if code == 0 and (run / 'complete.json').exists() else 'worker_failed')
                    if rec['status'] not in ('loading', 'running'):
                        stop = subprocess.run(service_command(rec, 'stop'), capture_output=True, text=True)
                        rec.update(stopped_at=time.time(), service_stop_exit_code=stop.returncode, service_stop_status=stop.stdout[-1200:])
                        rec.pop('_process', None)
                        del active[model]
                        print(json.dumps({'event': 'workflow11_shared_model_accounted', 'model': model, 'status': rec['status']}), flush=True)
                snap = gpu_snapshot()
                dump(folder / 'state.json', {'jobs': [{k: v for k, v in r.items() if not k.startswith('_')} for r in records],
                    'pending_models': pending, 'updated': time.time(), 'waiting_for_whole_idle_gpu': False,
                    'waiting_for_sufficient_low_load_memory': bool(pending), 'shared_resource_authorized': True,
                    'low_load_cards_now': [g for g, v in snap.items() if v['utilization_pct'] <= 10],
                    'all_accounted': not pending and not active})
                if pending or active:
                    time.sleep(10)
            dump(folder / 'complete.json', {'all_accounted': True, 'models_complete': sum(r['status'] == 'complete' for r in records),
                'models_planned': 2, 'failed_models': {r['model']: r['status'] for r in records if r['status'] != 'complete'},
                'owned_service_stop_attempted': True, 'shared_resource_authorized': True,
                'resource_amendment_sha256': plan['resource_amendment_sha256']})


if __name__ == '__main__':
    main()
