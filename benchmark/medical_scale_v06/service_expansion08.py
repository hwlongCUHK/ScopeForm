"""Non-overwriting owned local services for the new fixed model coverage."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
from run_generation import read,dump,digest,run_lock

BASE=Path('/data/zhanchen/cpm_medical06_20261020')
WORK=BASE/'medical_scale_v06'


@contextmanager
def admission_lock():
    # Admission is a short shared critical section, not an experiment run lock.
    # Parallel starts must wait here, not become failed model experiments.
    import fcntl
    with (BASE/'expansion08_gpu_admission.lock').open('a') as handle:
        fcntl.flock(handle.fileno(),fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def idle_gpus():
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    gpu={int(i):{'uuid':u.strip(),'used':int(m)} for i,u,m in (x.split(',') for x in lines.splitlines())}
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    busy={x.split(',')[0].strip() for x in apps.splitlines() if ',' in x}
    return [i for i,d in gpu.items() if d['used']<=100 and d['uuid'] not in busy]


def alive(pid):
    proc=Path('/proc')/str(pid)
    return proc.exists() and (proc/'stat').read_text().split()[2]!='Z'


def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=('start','stop','status'),required=True)
    p.add_argument('--model',required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--port',type=int,required=True);p.add_argument('--tag',required=True)
    a=p.parse_args()
    if os.name!='posix' or Path(__file__).resolve().parent!=WORK or not a.tag.replace('_','').isalnum():
        raise ValueError('Isolated server path and safe new tag required')
    proto=read(WORK/'model_expansion08_protocol.json')
    if a.model not in sum([proto[k] for k in ('native_models','json_models','decision_models','retrieval_models')],[]):
        raise ValueError('Model not authorized by new coverage protocol')
    owner=BASE/('expansion08_'+a.tag+'.service.json')
    if a.action=='start':
        with admission_lock():
            if owner.exists():raise FileExistsError('Never overwrite service ownership')
            if a.gpu not in idle_gpus():raise ValueError('Whole GPU not idle; do not share')
            with socket.socket() as s:
                if s.connect_ex(('127.0.0.1',a.port))==0:raise ValueError('Port occupied')
            model=Path('/data/zhanchen/models')/a.model
            conf=read(model/'config.json');ctx=32768 if a.model.startswith(('Qwen3-','Tacit-4','Tacit-8')) else 65536
            ctx=min(ctx,conf.get('max_position_embeddings',conf.get('text_config',{}).get('max_position_embeddings',ctx)))
            cmd=[str(Path(sys.executable).parent/'vllm'),'serve',str(model),'--served-model-name',a.model,
                '--host','127.0.0.1','--port',str(a.port),'--tensor-parallel-size','1','--dtype','bfloat16',
                '--max-model-len',str(ctx),'--max-num-seqs','4','--gpu-memory-utilization','0.55','--enforce-eager']
            if a.model in proto['native_models']:
                cmd+=['--enable-auto-tool-choice','--tool-call-parser','hermes','--reasoning-parser','qwen3']
            if a.model.startswith(('Qwen3.5-','Tacit-2','Tacit-9')):
                cmd+=['--language-model-only']
            if a.model in proto['retrieval_models']:
                cmd+=['--runner','pooling']
            elif a.model not in proto['native_models']:
                cmd+=['--logprobs-mode','processed_logprobs']
            if a.model.startswith('Tacit-') and (model/'chat_template.jinja').exists():
                cmd+=['--chat-template',str(model/'chat_template.jinja')]
            # UNIX-domain socket names have a strict length limit; long task tags
            # must not become RPC socket directories.
            tmp=Path(tempfile.mkdtemp(prefix='cpm_e08_',dir='/tmp'))
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(a.gpu),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
                'TMPDIR':str(tmp),'VLLM_RPC_BASE_PATH':str(tmp),'VLLM_CACHE_ROOT':str(tmp/'vllm'),'PATH':str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')}
            log=BASE/'logs'/('expansion08_'+a.tag+'.service.log')
            with log.open('a') as f:proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            rec=dict(pid=proc.pid,uid=os.getuid(),gpu=a.gpu,port=a.port,model=a.model,context=ctx,command=cmd,temporary=str(tmp),
                config_sha256=digest(model/'config.json'),protocol_sha256=digest(WORK/'model_expansion08_protocol.json'))
            dump(owner,rec);print(json.dumps({'event':'service_started',**rec}));return
    who=read(owner);pid=who['pid'];proc=Path('/proc')/str(pid)
    if a.action=='stop' and alive(pid):
        command=proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
        if who['uid']!=os.getuid() or proc.stat().st_uid!=who['uid'] or os.getpgid(pid)!=pid or not all(x in command for x in ('vllm',str(who['port']),who['model'])):
            raise ValueError('Ownership mismatch; nothing stopped')
        os.killpg(pid,signal.SIGTERM)
        for _ in range(30):
            if not alive(pid):break
            time.sleep(.5)
    with socket.socket() as s:listening=s.connect_ex(('127.0.0.1',who['port']))==0
    print(json.dumps({'pid':pid,'alive':alive(pid),'listening':listening,'gpu':who['gpu'],'model':who['model']}))


if __name__=='__main__':main()
