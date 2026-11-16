"""Gold-free completion of two missing factorial cells; no clinical evaluator."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import zipfile
import run_workflow11 as core
from run_generation import ROOT, read, dump, digest, run_lock
from pilot_runtime import fingerprint
from workflow11_runtime import World, fork, export_observations

RESTORE11 = core.restore
QUEUE = 'workflow12_queue_01'
ARMS = ('index_only', 'labels_guard')


class FactorialWorld(World):
    def initial_input(self):
        original = self.arm
        self.arm = 'evidence_constraint' if self.factorial_guard else 'labels_only'
        try:
            return super().initial_input()
        finally:
            self.arm = original

    def call(self, name, args):
        original = self.arm
        self.arm = 'evidence_constraint' if self.factorial_guard else 'labels_only'
        try:
            return super().call(name, args)
        finally:
            self.arm = original


def restore12(data):
    w = RESTORE11(data)
    if 'factorial_guard' in data:
        w.__class__ = FactorialWorld
    return w


def fork12(base, target, event, arm):
    if arm not in ARMS:
        raise ValueError('Undeclared supplementary arm')
    w = fork(base, target, event, 'evidence_constraint' if arm == 'index_only' else 'labels_only')
    w.__class__ = FactorialWorld
    w.arm, w.factorial_guard = arm, arm == 'labels_guard'
    assert fingerprint(w.state) == fingerprint(base.state)
    return w


def execute(actor, event, arm, base, base_hash, out, port, model, tokenizer, settings):
    sid, eid = actor['source_id'], event['event_id']
    rowpath = out / 'rows' / sid / (eid + '__' + arm + '.json')
    if rowpath.exists():
        old = read(rowpath)
        if old.get('initial_checkpoint_hash') != base_hash:
            raise ValueError('Existing row has another actual base')
        return old
    row = {'source_id': sid, 'event_id': eid, 'arm': arm, 'initial_checkpoint_hash': base_hash}
    if not base.done:
        row.update(submitted=False, status='blocked_actual_initial_not_submitted', requests=0, seconds=0)
    else:
        w = fork12(copy.deepcopy(base), event['controller'], eid, arm)
        start_hash = fingerprint(w.state)
        folder = out / 'phases' / sid / (eid + '__' + arm)
        w, status, calls, seconds = core.phase(w, folder, port, model, tokenizer, settings)
        row.update(submitted=w.done, status=status, requests=calls, seconds=seconds,
            pre_state_hash=start_hash, public_hash=fingerprint(w.public), environment=w.public['environment'],
            **export_observations(w))
    dump(rowpath, row)
    return row


def run(model, port):
    protocol = read(ROOT / 'workflow12_protocol.json')
    manifest = read(ROOT / 'workflow11_inputs_private/actor_manifest.json')
    if manifest['dataset_integrity_sha256'] != protocol['dataset_integrity_sha256']:
        raise ValueError('Frozen dataset differs')
    actors_file = ROOT / 'workflow11_inputs_private/actors.private.jsonl'
    if digest(actors_file) != manifest['files']['actors.private.jsonl']:
        raise ValueError('Actor input differs')
    parent = ROOT / 'runs' / protocol['parent_runs'][model]
    if not read(parent / 'complete.json')['all_phases_accounted']:
        raise ValueError('Parent run not terminal')
    prior = read(parent / 'results.json')
    actors = [json.loads(x) for x in actors_file.read_text(encoding='utf-8').splitlines()]
    if len(actors) != 120 or len(prior) != 1560:
        raise ValueError('Incomplete frozen matrix')
    out = ROOT / 'runs' / ('workflow12_q9b_01' if '9B' in model else 'workflow12_q27b_01')
    plan = {'model': model, 'port': port, 'attempt': 'factorial12_01', 'planned': 960,
        'protocol_sha256': digest(ROOT / 'workflow12_protocol.json'),
        'parent_results_sha256': digest(parent / 'results.json'),
        'parent_run': protocol['parent_runs'][model], 'parent_plan_sha256': digest(parent / 'plan.json'),
        'actor_bundle_sha256': digest(actors_file), 'code_sha256': digest(Path(__file__)),
        'gold_loaded_on_server': False, 'tool_call_parser': 'qwen3_coder'}
    with run_lock(out / 'run.lock'):
        if (out / 'plan.json').exists() and read(out / 'plan.json') != plan:
            raise ValueError('Never overwrite a changed run')
        dump(out / 'plan.json', plan)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('/data/zhanchen/models/' + model,
            local_files_only=True, trust_remote_code=False)
        # Core runner source remains sealed. Restore must preserve the two new
        # intervention flags when replaying checkpoints; fixed once before threads.
        core.restore = restore12
        jobs = []
        for actor in actors:
            sid = actor['source_id']
            ck = read(parent / 'phases' / sid / 'initial__initial/checkpoint.private.json')
            if not ck['terminal']:
                raise ValueError('Nonterminal parent initial')
            base = RESTORE11(ck['world'])
            h = fingerprint(core.snapshot(base))
            base_row = next(r for r in prior if r['source_id'] == sid and r['event_id'] == 'initial')
            if base.done != base_row['submitted'] or fingerprint(base.state) != fingerprint(base_row['state']):
                raise ValueError('Actual initial snapshot and exported row differ')
            if base.done and {r['initial_checkpoint_hash'] for r in prior if r['source_id'] == sid and r['event_id'] != 'initial'} != {h}:
                raise ValueError('Parent checkpoint identity differs')
            for event in actor['events']:
                for arm in ARMS:
                    jobs.append((actor, event, arm, base, h))
        jobs.sort(key=lambda j: fingerprint(j[0]['source_id'] + '|' + j[1]['event_id'] + '|' + j[2]))
        rows = []
        with ThreadPoolExecutor(max_workers=protocol['decoding']['workers_per_model']) as pool:
            futures = [pool.submit(execute, *job, out, port, model, tokenizer, protocol['decoding']) for job in jobs]
            for future in as_completed(futures):
                rows.append(future.result())
                dump(out / 'results.json', rows)
                counts = {s: sum(r['status'] == s for r in rows) for s in {r['status'] for r in rows}}
                dump(out / 'progress.json', {'model': model, 'planned': 960, 'recorded': len(rows),
                    'submitted': sum(r['submitted'] for r in rows), 'statuses': counts, 'updated': time.time()})
        if len(rows) != 960 or len({(r['source_id'], r['event_id'], r['arm']) for r in rows}) != 960:
            raise ValueError('Missing or duplicate new cells')
        dump(out / 'complete.json', {'model': model, 'recorded': 960,
            'submitted': sum(r['submitted'] for r in rows), 'all_phases_accounted': True,
            'attempt': 'factorial12_01', 'gold_loaded_on_server': False})


def service_command(rec, action):
    return [sys.executable, str(ROOT / 'service_workflow11_shared.py'), '--action', action,
        '--model', rec['model'], '--gpus', ','.join(map(str, rec['gpus'])), '--tp', str(rec['tp']),
        '--pp', str(rec['pp']), '--fraction', str(rec['fraction']), '--port', str(rec['port']), '--tag', rec['tag']]


def queue():
    from queue_workflow11_shared import select_profile
    from queue_workflow11 import ready
    from service_workflow11_shared import gpu_snapshot
    from service_expansion08 import alive
    p = read(ROOT / 'workflow12_protocol.json')
    folder = ROOT / 'runs' / QUEUE
    with run_lock(folder / 'run.lock'):
        if (folder / 'state.json').exists():
            raise ValueError('Existing queue: diagnose, never launch duplicate')
        for name in ('workflow12_q9b_01', 'workflow12_q27b_01'):
            if (ROOT / 'runs' / name).exists():
                raise ValueError('Existing new run: do not resample')
        dump(folder / 'plan.json', {'protocol_sha256': digest(ROOT / 'workflow12_protocol.json'),
             'code_sha256': digest(Path(__file__)), 'planned': 1920})
        pending, active, records = list(p['models']), {}, []
        amendment = read(ROOT / 'workflow11_resource_amendment.json')
        with (folder / 'workers.log').open('a') as log:
            while pending or active:
                for model in list(pending):
                    selected = select_profile(model, {g for r in active.values() for g in r['gpus']}, amendment)
                    if selected is None:
                        continue
                    gpus, option = selected
                    idx = p['models'].index(model)
                    rec = {'model': model, 'gpus': gpus, 'gpu': gpus[0], 'tp': option['tensor_parallel'],
                        'pp': option['pipeline_parallel'], 'fraction': option['gpu_memory_utilization'],
                        'port': 18610 + idx, 'tag': 'workflow12_' + ('q9b_01' if idx == 0 else 'q27b_01'),
                        'status': 'loading', 'started': time.time()}
                    result = subprocess.run(service_command(rec, 'start'), capture_output=True, text=True)
                    if result.returncode:
                        raise RuntimeError('Owned service admission/start failed: ' + result.stderr[-500:])
                    admission = json.loads(result.stdout)
                    if not admission.get('admitted'):
                        continue
                    rec['service_pid'] = admission['pid']
                    records.append(rec); active[model] = rec; pending.remove(model)
                for model, rec in list(active.items()):
                    if rec['status'] == 'loading':
                        if ready(rec['port'], model):
                            pre = subprocess.run([sys.executable, str(ROOT / 'preflight_workflow11_xml.py'),
                                '--model', model, '--port', str(rec['port']), '--output',
                                str((folder / ('preflight_' + str(rec['port']) + '.json')).relative_to(ROOT))],
                                stdout=log, stderr=subprocess.STDOUT)
                            rec['preflight_exit_code'] = pre.returncode
                            if pre.returncode:
                                rec['status'] = 'preflight_failed'
                            else:
                                proc = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__)),
                                    'run', '--model', model, '--port', str(rec['port']), '--allow-private-data'],
                                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                                rec.update(status='running', worker_pid=proc.pid, _proc=proc)
                        elif not alive(rec['service_pid']) or time.time() - rec['started'] > 1200:
                            rec['status'] = 'load_failed'
                    elif rec['status'] == 'running' and rec['_proc'].poll() is not None:
                        code = rec['_proc'].returncode
                        name = 'workflow12_q9b_01' if '9B' in model else 'workflow12_q27b_01'
                        rec.update(worker_exit_code=code, status='complete' if code == 0 and (ROOT / 'runs' / name / 'complete.json').exists() else 'worker_failed')
                    if rec['status'] not in ('loading', 'running'):
                        stop = subprocess.run(service_command(rec, 'stop'), capture_output=True, text=True)
                        rec.update(service_stop_exit_code=stop.returncode, service_stop_status=stop.stdout[-1200:])
                        rec.pop('_proc', None); del active[model]
                dump(folder / 'state.json', {'jobs': [{k: v for k, v in r.items() if not k.startswith('_')} for r in records],
                    'pending_models': pending, 'updated': time.time(), 'all_accounted': not pending and not active})
                if pending or active:
                    time.sleep(10)
        dump(folder / 'complete.json', {'models_complete': sum(r['status'] == 'complete' for r in records),
            'failed_models': {r['model']: r['status'] for r in records if r['status'] != 'complete'}, 'planned': 1920})


def cleanup_audit():
    records = read(ROOT / 'runs' / QUEUE / 'state.json')['jobs']
    checked = []
    for r in records:
        log = ROOT.parent / 'logs' / ('workflow11_' + r['tag'] + '.service.log')
        ids = {r['service_pid'], r['worker_pid']}
        if log.exists():
            ids.update(int(x) for x in re.findall(r'(?:EngineCore|Worker)[^\n]*?pid=(\d+)', log.read_text(errors='replace')))
        live = []
        for pid in ids:
            proc = Path('/proc') / str(pid)
            if proc.exists() and proc.stat().st_uid == os.getuid() and (proc / 'stat').read_text().split()[2] != 'Z':
                cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
                if 'vllm' in cmd or 'EngineCore' in cmd or 'workflow12.py run' in cmd:
                    live.append(pid)
        with socket.socket() as s:
            listening = s.connect_ex(('127.0.0.1', r['port'])) == 0
        checked.append({'model': r['model'], 'live_owned_pids': live, 'port_listening': listening})
    return {'services': checked, 'all_owned_services_stopped': all(not x['live_owned_pids'] and not x['port_listening'] for x in checked)}


def export(tag):
    if not tag.replace('_', '').isalnum():
        raise ValueError('Unsafe export name')
    folder = ROOT / 'runs' / QUEUE
    if read(folder / 'complete.json')['models_complete'] != 2:
        raise ValueError('Do not export partial batch')
    audit = cleanup_audit()
    if not audit['all_owned_services_stopped']:
        raise ValueError('Owned process cleanup incomplete')
    dump(folder / 'cleanup_audit.json', audit)
    names = ['workflow12_protocol.json']
    for runname in ('workflow12_q9b_01', 'workflow12_q27b_01'):
        names += ['runs/' + runname + '/' + x for x in ('plan.json', 'progress.json', 'results.json', 'complete.json')]
    names += ['runs/' + QUEUE + '/' + x for x in ('plan.json', 'state.json', 'complete.json', 'cleanup_audit.json', 'preflight_18610.json', 'preflight_18611.json')]
    dest = ROOT / 'exports' / ('workflow12_predictions_' + tag + '.zip')
    hashes = {n: digest(ROOT / n) for n in names}
    with zipfile.ZipFile(dest, 'x', zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(ROOT / n, n)
    dump(dest.with_suffix('.manifest.json'), {'archive': dest.name, 'archive_sha256': digest(dest), 'files': hashes,
        'source_or_gold_or_request_text_included': False})
    print(json.dumps({'archive': dest.name, 'bytes': dest.stat().st_size, 'no_gold_or_text': True}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('run', 'queue', 'export'))
    p.add_argument('--model', choices=('Qwen3.5-9B', 'Qwen3.8-27B'))
    p.add_argument('--port', type=int)
    p.add_argument('--tag')
    p.add_argument('--allow-private-data', action='store_true')
    a = p.parse_args()
    if a.action in ('run', 'queue') and (os.name != 'posix' or not a.allow_private_data):
        raise ValueError('Explicit authorized server execution only')
    if a.action == 'queue': queue()
    elif a.action == 'run': run(a.model, a.port)
    else: export(a.tag)
