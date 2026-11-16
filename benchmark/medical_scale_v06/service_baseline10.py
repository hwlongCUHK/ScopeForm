"""Owned whole-idle-GPU services for a fixed development evaluation only."""
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
from baseline10_common import context_for
from run_generation import read, dump, digest
from service_expansion08 import admission_lock, idle_gpus, alive, BASE, WORK


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--action', choices=('start', 'stop', 'status'), required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--tag', required=True)
    a = p.parse_args()
    if os.name != 'posix' or Path(__file__).resolve().parent != WORK or not a.tag.replace('_', '').isalnum():
        raise ValueError('Fixed server workspace and safe tag required')
    protocol = read(WORK / 'baseline10_protocol.json')
    allowed = set(sum([protocol[k] for k in ('generative_models', 'decision_models', 'retrieval_models')], []))
    if a.model not in allowed:
        raise ValueError('Not a predeclared existing local checkpoint')
    owner = BASE / ('baseline10_' + a.tag + '.service.json')
    if a.action == 'start':
        with admission_lock():
            if owner.exists():
                raise FileExistsError('Preserve prior ownership; use a new service tag')
            if a.gpu not in idle_gpus():
                raise ValueError('Whole GPU not idle: do not share or evict')
            with socket.socket() as s:
                if s.connect_ex(('127.0.0.1', a.port)) == 0:
                    raise ValueError('Port already occupied')
            model = Path('/data/zhanchen/models') / a.model
            if not list(model.glob('*.safetensors')):
                raise ValueError('No actual local checkpoint weights')
            config = read(model / 'config.json')
            ctx = context_for(a.model)
            cmd = [str(Path(sys.executable).parent / 'vllm'), 'serve', str(model),
                   '--served-model-name', a.model, '--host', '127.0.0.1', '--port', str(a.port),
                   '--tensor-parallel-size', '1', '--dtype', 'bfloat16',
                   '--max-model-len', str(ctx), '--max-num-seqs', '8',
                   '--gpu-memory-utilization', '0.82' if a.model == 'Qwen3.8-27B' else '0.65',
                   '--enforce-eager']
            if config['model_type'] == 'qwen3_5':
                cmd += ['--language-model-only']
            if a.model in protocol['retrieval_models']:
                cmd += ['--runner', 'pooling']
            if a.model.startswith('Tacit-') and (model / 'chat_template.jinja').exists():
                cmd += ['--chat-template', str(model / 'chat_template.jinja')]
            tmp = Path(tempfile.mkdtemp(prefix='cpm_b10_', dir='/tmp'))
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(a.gpu), 'HF_HUB_OFFLINE': '1',
                   'TRANSFORMERS_OFFLINE': '1', 'TMPDIR': str(tmp), 'VLLM_RPC_BASE_PATH': str(tmp),
                   'VLLM_CACHE_ROOT': str(tmp / 'vllm'),
                   'PATH': str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')}
            log = BASE / 'logs' / ('baseline10_' + a.tag + '.service.log')
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('a') as f:
                proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=f,
                                        stderr=subprocess.STDOUT, start_new_session=True)
            rec = {'pid': proc.pid, 'uid': os.getuid(), 'gpu': a.gpu, 'port': a.port,
                   'model': a.model, 'context': ctx, 'command': cmd, 'temporary': str(tmp),
                   'config_sha256': digest(model / 'config.json'),
                   'protocol_sha256': digest(WORK / 'baseline10_protocol.json')}
            dump(owner, rec)
            print(json.dumps({'event': 'baseline10_service_started', **rec}), flush=True)
            return
    rec = read(owner)
    if any(rec[k] != getattr(a, k) for k in ('model', 'gpu', 'port')):
        raise ValueError('Ownership arguments do not match')
    pid = rec['pid']
    proc = Path('/proc') / str(pid)
    if a.action == 'stop' and alive(pid):
        cmd = proc.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode()
        if rec['uid'] != os.getuid() or proc.stat().st_uid != rec['uid'] or os.getpgid(pid) != pid or not all(s in cmd for s in ('vllm', str(rec['port']), rec['model'])):
            raise ValueError('Refuse stopping a non-owned process')
        os.killpg(pid, signal.SIGTERM)
        for _ in range(40):
            if not alive(pid):
                break
            time.sleep(.5)
    with socket.socket() as s:
        listening = s.connect_ex(('127.0.0.1', rec['port'])) == 0
    print(json.dumps({'pid': pid, 'alive': alive(pid), 'listening': listening,
                      'gpu': rec['gpu'], 'model': rec['model']}), flush=True)


if __name__ == '__main__':
    main()
