"""Finite three-checkpoint batch, at most two owned whole-idle services.

Reuse tested local service management. No downloads, foreign task termination,
clinical scoring, retries or replacement of uncertain model requests.
"""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from run_generation import ROOT,read,dump,digest,run_lock
from service_expansion08 import idle_gpus,alive
from study15 import MODELS,SLUGS,STUDY

QUEUE=ROOT/'runs/confirmation15_queue_01'
SERVICE_ROOT=Path('/data/zhanchen/cpm_medical06_20261020/medical_scale_v06')


def service(rec,action):
    proc=subprocess.run([sys.executable,'-B',str(SERVICE_ROOT/'service_baseline10.py'),'--action',action,
        '--model',rec['model'],'--gpu',str(rec['gpu']),'--port',str(rec['port']),'--tag',rec['tag']],capture_output=True,text=True)
    if proc.returncode:raise RuntimeError('Owned service '+action+' failed')
    return json.loads(proc.stdout)


def ready(port,model):
    import urllib.request
    try:
        with urllib.request.urlopen('http://127.0.0.1:'+str(port)+'/v1/models',timeout=2) as f:return model in {v['id'] for v in json.load(f).get('data',[])}
    except Exception:return False


def group_audit(rec):
    live=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            fields=(path/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])==rec['service_pid'] and fields[0] not in ('Z','X'):live.append(int(path.name))
        except (OSError,ValueError,IndexError):pass
    with socket.socket() as s:s.settimeout(1);listening=s.connect_ex(('127.0.0.1',rec['port']))==0
    return {'model':rec['model'],'live_owned_service_group_pids':live,'port_listening':listening,
        'worker_alive':alive(rec['worker_pid']) if rec.get('worker_pid') else False}


def main():
    if os.name!='posix' or ROOT!=SERVICE_ROOT/'study15_runtime_v1':raise ValueError('Named isolated authorized server only')
    study=read(STUDY)
    for name in ('service_baseline10.py','baseline10_common.py','service_expansion08.py','baseline10_protocol.json'):
        if digest(SERVICE_ROOT/name)!=study['parent_service_sha256'][name]:raise ValueError('Existing service manager changed')
    with run_lock(QUEUE/'queue.lock'):
        if (QUEUE/'state.json').exists():raise ValueError('Existing batch: inspect, never duplicate launch')
        records=[];active={};pending=list(MODELS)
        dump(QUEUE/'plan.json',{'models':list(MODELS),'max_services':2,'study_protocol':str(STUDY),
            'web_controls_are_separate_pending':True,'admission':'whole idle only, no foreign process stopped'})
        log=(QUEUE/'workers.log').open('a')
        try:
            while pending or active:
                candidates=set(idle_gpus())
                for _ in range(2):
                    time.sleep(1);candidates.intersection_update(idle_gpus())
                available=[g for g in sorted(candidates) if g not in {r['gpu'] for r in active.values()}]
                while pending and available and len(active)<2:
                    model=pending.pop(0);idx=MODELS.index(model);gpu=available.pop(0)
                    rec={'model':model,'gpu':gpu,'port':18750+idx,'tag':'confirmation15_'+SLUGS[model]+'_01',
                        'status':'loading','started':time.time()}
                    admission=service(rec,'start');rec['service_pid']=admission['pid'];records.append(rec);active[model]=rec
                for model,rec in list(active.items()):
                    if rec['status']=='loading':
                        if ready(rec['port'],model):
                            proc=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'study15.py'),'--model',model,
                                '--port',str(rec['port']),'--allow-private-data'],cwd=ROOT,stdin=subprocess.DEVNULL,
                                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                            rec.update(status='running',worker_pid=proc.pid);rec['_proc']=proc
                        elif not alive(rec['service_pid']) or time.time()-rec['started']>1200:rec['status']='load_failed'
                    elif rec['status']=='running' and rec['_proc'].poll() is not None:
                        rec.update(worker_exit_code=rec['_proc'].returncode,
                            status='complete_current_batch' if rec['_proc'].returncode==0 and (ROOT/'runs'/('confirmation15_'+SLUGS[model]+'_01')/'complete.json').exists() else 'worker_failed')
                    if rec['status'] not in ('loading','running'):
                        rec['service_stop']=service(rec,'stop')
                        for _ in range(30):
                            audit=group_audit(rec)
                            if not audit['live_owned_service_group_pids'] and not audit['port_listening'] and not audit['worker_alive']:break
                            time.sleep(2)
                        rec['cleanup_audit']=audit;rec.pop('_proc',None);del active[model]
                dump(QUEUE/'state.json',{'jobs':[{k:v for k,v in r.items() if not k.startswith('_')} for r in records],
                    'pending_models':pending,'all_current_batch_accounted':not pending and not active,'updated':time.time()})
                if pending or active:time.sleep(10)
        finally:
            for rec in active.values():
                # A queue failure does not justify killing an in-progress
                # worker or a foreign process. Preserve its exact identity.
                if rec.get('_proc') and rec['_proc'].poll() is None:
                    dump(QUEUE/'needs_attention.json',{'reason':'queue_exited_with_owned_worker_running',
                        'model':rec['model'],'worker_pid':rec['worker_pid'],'service_pid':rec['service_pid']})
                else:
                    try:service(rec,'stop')
                    except Exception:pass
            log.close()
        clean=all(not r['cleanup_audit']['live_owned_service_group_pids'] and not r['cleanup_audit']['port_listening'] and not r['cleanup_audit']['worker_alive'] for r in records)
        dump(QUEUE/'complete.json',{'models_complete':sum(r['status']=='complete_current_batch' for r in records),
            'models_failed':{r['model']:r['status'] for r in records if r['status']!='complete_current_batch'},
            'all_owned_resources_clean':clean,'web_controls_complete':False,'full_study_complete':False})


if __name__=='__main__':main()
