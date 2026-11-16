#!/usr/bin/env python3
"""Portable launcher for the exact recorded inference snapshots (no scoring targets loaded)."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import getpass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
SETTINGS=['fresh','labels','index','guard','index_guard']
sys.dont_write_bytecode=True


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def load_five(backbone):
    os.environ['CPM_BASELINE_ROOT']=str(ROOT/'code/baseline')
    os.environ['CPM_PACKAGE_ROOT']=str(ROOT/'benchmark')
    path=ROOT/'code'/backbone/'run_five_settings.py'
    spec=importlib.util.spec_from_file_location('run_five_settings',path)
    five=importlib.util.module_from_spec(spec);spec.loader.exec_module(five)
    return five,five.load_frozen(ROOT/'code'/backbone)


def qwen_client(baseline,out,key,endpoint):
    class QwenClient(baseline.Client):
        def request(self,path,payload=None):
            headers={'Content-Type':'application/json'}
            if self.key:headers['Authorization']='Bearer '+self.key
            req=urllib.request.Request(self.endpoint+path,data=json.dumps(payload,ensure_ascii=False).encode() if payload is not None else None,headers=headers)
            try:
                with urllib.request.build_opener(baseline.NoRedirect()).open(req,timeout=1800) as response:return json.load(response)
            except urllib.error.HTTPError as error:
                raise RuntimeError('api_http_'+str(error.code)) from None
        def chat(self,payload):
            # Exact wire transformation used for the Qwen paper runs.
            sent=copy.deepcopy(payload);sent.pop('thinking',None)
            sent['chat_template_kwargs']={'enable_thinking':False};sent['seed']=20261022
            if sent.get('tools'):sent['tool_choice']='auto'
            stamp=str(time.time_ns())+'_'+str(threading.get_ident())
            dump(out/'transport_requests'/(stamp+'.json'),sent)
            return super().chat(sent)
    return QwenClient(key,endpoint)


def initial_checkpoints(backbone,source_ids,dest):
    with tarfile.open(ROOT/'artifacts'/backbone/'shared_initials.tar.gz','r:gz') as tar:
        for member in tar.getmembers():
            if not member.isfile():continue
            if Path(member.name).parts[1] not in source_ids:continue
            target=(dest/member.name).resolve();target.relative_to(dest.resolve())
            # Resuming a prepared run must not overwrite its current checkpoints.
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as src:target.write_bytes(src.read())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backbone',choices=('deepseek','qwen'),required=True)
    p.add_argument('--scope',choices=('pilot','full'),default='pilot',help='Pilot: 6 dialogue + 4 drug sources selected by identity hash.')
    p.add_argument('--settings',nargs='+',choices=SETTINGS,default=SETTINGS)
    p.add_argument('--source-ids',nargs='+',help='Explicit small subset; overrides scope selection.')
    p.add_argument('--initials',choices=('recorded','fresh'),default='recorded')
    p.add_argument('--endpoint',help='API base URL, including /v1 for local Qwen.')
    p.add_argument('--workers',type=int)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true',help='Load real inference code and prepare files; make zero model calls.')
    args=p.parse_args();out=args.out.resolve()
    original=read(ROOT/'provenance'/args.backbone/'config.original.json')
    for name,expected in original['code_sha256'].items():
        assert sha(ROOT/'code'/args.backbone/'frozen_code'/name)==expected,name
    for name,expected in original['runtime_code_sha256'].items():
        assert sha(ROOT/'benchmark/medical_scale_v06'/name)==expected,name
    for name,expected in original['original_baseline_sha256'].items():
        assert sha(ROOT/'code/baseline'/name)==expected,name
    actor_file=ROOT/'benchmark/medical_scale_v06/confirmation15_inputs_private/actors.private.jsonl'
    assert sha(actor_file)==original['actors_sha256']
    actors=[json.loads(line) for line in actor_file.read_text().splitlines() if line]
    if args.source_ids:
        known={a['source_id'] for a in actors};assert set(args.source_ids)<=known,'Unknown source ID'
        actors=[a for a in actors if a['source_id'] in args.source_ids]
    elif args.scope=='pilot':
        # Match the original outcome-independent selection exactly.
        def identity_hash(sid):return hashlib.sha256(json.dumps(sid,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        actors=[a for modality,n in [('dialogue',6),('drug',4)]
                for a in sorted((a for a in actors if a['initial']['modality']==modality),key=lambda a:identity_hash(a['source_id']))[:n]]
    actors.sort(key=lambda a:a['source_id']);ids=[a['source_id'] for a in actors]
    selected=[s for s in SETTINGS if s in args.settings]
    endpoint=args.endpoint or ('https://api.deepseek.com' if args.backbone=='deepseek' else 'http://127.0.0.1:18227/v1')
    settings={**original['settings']}
    if args.workers is not None:settings['workers_per_model']=args.workers
    assert settings['workers_per_model']>0
    config={
        'kind':'scopeform_portable_rerun','backbone':args.backbone,'model':original['model'],
        'endpoint':endpoint,'source_ids':ids,'events':original['events'],
        'selected_settings':selected,'settings':settings,'initials':args.initials,
        'scope':'explicit_subset' if args.source_ids else args.scope,
        'planned_phases':len(ids)*(1+4*2*len(selected)),
        'planned_new_initial_phases':len(ids) if args.initials=='fresh' else 0,
        'planned_update_phases':len(ids)*4*2*len(selected),
        'frozen_code':str(ROOT/'code'/args.backbone/'frozen_code'),
        'package_root':str(ROOT/'benchmark'),'baseline_root':str(ROOT/'code/baseline'),
        'code_sha256':original['code_sha256'],'actors_sha256':original['actors_sha256'],
        'native_tool_choice':'auto' if args.backbone=='qwen' else 'unchanged native API payload',
        'temperature':0,'thinking':'disabled','max_requests':40,'max_actions':40,
        'no_evaluator_targets_loaded':True,'notes_enabled':False,
        'replica_routing':'Single endpoint for both paired arms; historical replicas recorded in provenance.',
        'budget':'Native, formalization, diagnosis, and repair feedback calls share 40 total requests per phase; failures retain zero score.',
    }
    if (out/'config.json').exists():assert read(out/'config.json')==config,'Use a different output directory for a different configuration.'
    dump(out/'config.json',config)
    if args.initials=='recorded':initial_checkpoints(args.backbone,set(ids),out/'main/shared')
    five,run=load_five(args.backbone)
    five.ARMS=tuple(s+'__'+v for s in selected for v in ('baseline','formalization'))
    if args.prepare_only:
        print(json.dumps({'prepared':str(out),'source_ids':ids,'planned_phases':config['planned_phases'],'model_calls':0},ensure_ascii=False,indent=2));return
    if args.backbone=='deepseek':
        key=os.environ.get('DEEPSEEK_API_KEY') or getpass.getpass('DeepSeek API key (hidden): ')
        client=run.baseline.Client(key,endpoint)
    else:
        client=qwen_client(run.baseline,out,os.environ.get('QWEN_API_KEY',''),endpoint)
    runner=five.runner_class(run)(client,config['model'],settings,'repair')
    run.h.phase=runner.phase
    start=time.monotonic()
    dump(out/'run_status.json',{'status':'running','started_utc':datetime.now(timezone.utc).isoformat()})
    runner.workflow(actors,out/'main')
    dump(out/'provider_metadata.json',{'requested_model':config['model'],'returned_models':sorted(client.models),'system_fingerprints':sorted(client.fingerprints)})
    dump(out/'run_status.json',{'status':'complete','wall_seconds':time.monotonic()-start,'finished_utc':datetime.now(timezone.utc).isoformat()})
    print('Completed. Score with: python scripts/reproduce.py --run-dir '+str(out)+' --out '+str(out/'evaluation'))

if __name__=='__main__':main()
