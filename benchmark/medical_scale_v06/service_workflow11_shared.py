"""Explicitly user-authorized low-load sharing, BF16; stop only owned groups."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
from run_generation import read, dump, digest
from service_expansion08 import BASE, WORK, admission_lock, alive


def gpu_snapshot():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    return {int(a): {'free_mib': int(b), 'total_mib': int(c), 'utilization_pct': int(d)}
            for a, b, c, d in (line.split(',') for line in raw.splitlines())}


def admitted(gpus, fraction, samples):
    return all(g in s and s[g]['utilization_pct'] <= 10 and
               s[g]['free_mib'] >= fraction * s[g]['total_mib'] + 2048 for s in samples for g in gpus)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--action', choices=('start', 'stop', 'status'), required=True)
    p.add_argument('--model', choices=('Qwen3.5-9B', 'Qwen3.8-27B'), required=True)
    p.add_argument('--gpus', required=True)
    p.add_argument('--tp', type=int, default=1)
    p.add_argument('--pp', type=int, default=1)
    p.add_argument('--fraction', type=float, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--tag', required=True)
    a = p.parse_args()
    gpus = [int(n) for n in a.gpus.split(',')]
    if os.name != 'posix' or Path(__file__).resolve().parent != WORK or not a.tag.replace('_', '').isalnum() or len(set(gpus)) != a.tp * a.pp:
        raise ValueError('Fixed server, safe tag, matching distinct GPU count required')
    amendment = read(WORK / 'workflow11_resource_amendment.json')
    options = amendment.get('q9b_options_in_order', [amendment['q9b']]) if a.model == 'Qwen3.5-9B' else amendment['q27b_options_in_order']
    if not any((x['tensor_parallel'], x['pipeline_parallel'], x['gpu_memory_utilization']) == (a.tp, a.pp, a.fraction) for x in options):
        raise ValueError('Undeclared resource profile')
    owner = BASE / ('workflow11_' + a.tag + '.service.json')
    if a.action == 'start':
        with admission_lock():
            if owner.exists():
                raise FileExistsError('Preserve previous ownership record')
            samples = []
            for i in range(3):
                samples.append(gpu_snapshot())
                if i < 2:
                    time.sleep(1)
            if not admitted(gpus, a.fraction, samples):
                print(json.dumps({'admitted': False, 'reason': 'load_or_memory_changed', 'gpus': gpus}), flush=True)
                return
            with socket.socket() as s:
                if s.connect_ex(('127.0.0.1', a.port)) == 0:
                    raise ValueError('Port occupied')
            path = Path('/data/zhanchen/models') / a.model
            if not list(path.glob('*.safetensors')) or read(path / 'config.json')['model_type'] != 'qwen3_5':
                raise ValueError('Expected installed BF16 weights')
            tmp = Path(tempfile.mkdtemp(prefix='cpm_w11s_', dir='/tmp'))
            cmd = [str(Path(sys.executable).parent / 'vllm'), 'serve', str(path),
                '--served-model-name', a.model, '--host', '127.0.0.1', '--port', str(a.port),
                '--tensor-parallel-size', str(a.tp), '--pipeline-parallel-size', str(a.pp),
                '--distributed-executor-backend', 'mp', '--dtype', 'bfloat16', '--max-model-len', '65536',
                '--max-num-seqs', '8', '--max-num-batched-tokens', '4096', '--gpu-memory-utilization', str(a.fraction),
                '--enforce-eager', '--language-model-only', '--enable-auto-tool-choice',
                '--tool-call-parser', 'qwen3_coder', '--reasoning-parser', 'qwen3']
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': a.gpus, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'TMPDIR': str(tmp), 'VLLM_RPC_BASE_PATH': str(tmp), 'VLLM_CACHE_ROOT': str(tmp / 'vllm'),
                'PATH': str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')}
            log = BASE / 'logs' / ('workflow11_' + a.tag + '.service.log')
            with log.open('a') as h:
                proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=h, stderr=subprocess.STDOUT, start_new_session=True)
            metadata = {n: digest(path / n) for n in ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'chat_template.jinja', 'model.safetensors.index.json') if (path / n).exists()}
            rec = {'pid': proc.pid, 'uid': os.getuid(), 'gpu': gpus[0], 'gpus': gpus, 'port': a.port, 'model': a.model,
                'tp': a.tp, 'pp': a.pp, 'gpu_memory_utilization': a.fraction, 'command': cmd, 'temporary': str(tmp),
                'resource_amendment_sha256': digest(WORK / 'workflow11_resource_amendment.json'),
                'protocol_sha256': digest(WORK / 'workflow11_protocol.json'), 'checkpoint_metadata_sha256': metadata,
                'admission_samples': [{g: s[g] for g in gpus} for s in samples]}
            dump(owner, rec)
            print(json.dumps({'admitted': True, 'pid': proc.pid, 'gpus': gpus, 'model': a.model}), flush=True)
            return
    rec = read(owner)
    if rec['gpus'] != gpus or any(rec[k] != getattr(a, k) for k in ('model', 'port', 'tp', 'pp')):
        raise ValueError('Ownership arguments differ')
    pid, proc = rec['pid'], Path('/proc') / str(rec['pid'])
    if a.action == 'stop' and alive(pid):
        command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        if rec['uid'] != os.getuid() or proc.stat().st_uid != rec['uid'] or os.getpgid(pid) != pid or not all(t in command for t in ('vllm', a.model, str(a.port))):
            raise ValueError('Never stop a foreign or reused process')
        os.killpg(pid, signal.SIGTERM)
        for _ in range(60):
            if not alive(pid):
                break
            time.sleep(.5)
    with socket.socket() as s:
        listening = s.connect_ex(('127.0.0.1', a.port)) == 0
    print(json.dumps({'pid': pid, 'alive': alive(pid), 'listening': listening, 'gpus': gpus, 'model': a.model}), flush=True)


if __name__ == '__main__':
    main()
