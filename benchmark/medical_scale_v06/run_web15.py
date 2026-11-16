"""Finite paired-layout service study; no medical gold import."""
import argparse
from collections import Counter
import copy
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import json
import time

from run_generation import ROOT,read,dump,digest,run_lock
from pilot_runtime import fingerprint
import confirmation14_workflow as h
import study15 as parent
import web15_runtime as web

PROTOCOL=ROOT/'web15_protocol.json'
CODE=parent.CODE+('web15_runtime.py','run_web15.py','queue_web15.py','export_web15.py','validate_generated.py')


def one(actor,event,method,layout,out,port,tool_port,model,tokenizer,settings,base=None):
    sid=actor['source_id'];eid=event['event_id'] if event else 'initial'
    rowfile=out/'rows'/sid/(eid+'__'+method+'.json');folder=out/'phases'/sid/(eid+'__'+method)
    actual_hash=web.checkpoint_hash(base) if base is not None else None
    if rowfile.exists():
        row=read(rowfile)
        if row.get('initial_checkpoint_hash')!=actual_hash:raise ValueError('Actual Web start changed')
        return row,web.restore_proxy(read(folder/'checkpoint.private.json')['world']) if event is None else None
    row={'source_id':sid,'layout':layout,'event_id':eid,'arm':method,'initial_checkpoint_hash':actual_hash,'harness':'shared_json_action15_loopback_service'}
    if base is not None and not base.done:
        row.update(submitted=False,status='blocked_actual_initial_not_submitted',requests=0,seconds=0,
            usage={'input_tokens':0,'output_tokens':0});dump(rowfile,row);return row,None
    world=web.WebWorld(actor['initial'],layout,0,tool_port) if event is None else web.branch(copy.deepcopy(base),event['controller'],eid,method,tool_port)
    start=fingerprint(world.state)
    world,status,calls,seconds,usage=h.phase(world,folder,port,model,tokenizer,settings)
    row.update(submitted=world.done,status=status,requests=calls,seconds=seconds,usage=usage,
        pre_state_hash=start,public_hash=fingerprint(world.public),environment='E3',
        web_revision=world.web_revision,**h.export_observations(world))
    dump(rowfile,row);world.close();return row,world


def workflow(actors,layout,out,port,tool_port,model,tokenizer,settings):
    rows=[]
    with ThreadPoolExecutor(max_workers=settings['workers_per_model']) as pool:
        futures={pool.submit(one,a,None,'initial',layout,out,port,tool_port,model,tokenizer,settings):(a,True) for a in sorted(actors,key=lambda a:fingerprint(a['source_id']))}
        while futures:
            done,_=wait(futures,return_when=FIRST_COMPLETED)
            for future in done:
                actor,initial=futures.pop(future);row,world=future.result();rows.append(row)
                if initial:
                    for event,method in sorted([(e,m) for e in actor['events'] for m in parent.SOURCE_METHODS],key=lambda x:fingerprint([actor['source_id'],x[0]['event_id'],x[1]])):
                        futures[pool.submit(one,actor,event,method,layout,out,port,tool_port,model,tokenizer,settings,world)]=(actor,False)
                dump(out/'results.json',rows);dump(out/'progress.json',{'planned':480,'recorded':len(rows),
                    'submitted':sum(r['submitted'] for r in rows),'statuses':dict(Counter(r['status'] for r in rows)),'updated':time.time()})
    if len(rows)!=480 or len({(r['source_id'],r['event_id'],r['arm']) for r in rows})!=480:raise ValueError('Web matrix incomplete')
    dump(out/'complete.json',{'recorded':480,'submitted':sum(r['submitted'] for r in rows),'all_accounted':True})


def main(model,port):
    p=read(PROTOCOL)
    for n,sha in p['runtime_code_sha256'].items():
        if digest(ROOT/n)!=sha:raise ValueError('Frozen Web runtime changed')
    if digest(parent.STUDY)!=p['parent_study_sha256']:raise ValueError('Scientific design changed')
    manifest=read(parent.INPUTS/'actor_manifest.json')
    for n,sha in manifest['files'].items():
        if digest(parent.INPUTS/n)!=sha:raise ValueError('Task input changed')
    actors=[json.loads(x) for x in (parent.INPUTS/'actors.private.jsonl').read_text(encoding='utf8').splitlines()]
    chosen=set(p['source_ids']);actors=[a for a in actors if a['source_id'] in chosen]
    if len(actors)!=48:raise ValueError('Declared fixed subset required')
    out=ROOT/'runs'/('web15_'+parent.SLUGS[model]+'_01');settings={**p['decoding'],'context_limit':p['model_context_limits'][model]}
    with run_lock(out/'run.lock'):
        plan={'model':model,'port':port,'protocol_sha256':digest(PROTOCOL),'parent_study_sha256':p['parent_study_sha256'],
            'planned':960,'source_ids':p['source_ids'],'gold_loaded_on_server':False,'settings':settings,'http_service_port':port+30}
        if (out/'plan.json').exists() and read(out/'plan.json')!=plan:raise ValueError('Existing Web plan changed')
        dump(out/'plan.json',plan)
        from transformers import AutoTokenizer
        path=__import__('pathlib').Path('/data/zhanchen/models')/model
        tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        if not tokenizer.chat_template and (path/'chat_template.jinja').exists():tokenizer.chat_template=(path/'chat_template.jinja').read_text()
        dump(out/'checkpoint_metadata.json',{n:digest(path/n) for n in ('config.json','tokenizer_config.json','tokenizer.json','chat_template.jinja','model.safetensors.index.json') if (path/n).exists()})
        with web.ToolService(port+30) as service:
            original=h.World;h.World=lambda public:web.WebWorld(public,port=service.port)
            try:parent.preflight(port,model,tokenizer,settings,out)
            finally:h.World=original;service.sessions.clear()
            h.restore=web.restore_proxy
            for layout in web.LAYOUTS:workflow([web.actor(a,layout) for a in actors],layout,out/layout,port,service.port,model,tokenizer,settings)
            if service.sessions:raise ValueError('Owned HTTP sessions not all closed')
        dump(out/'complete.json',{'model':model,'phases':960,'all_accounted':True,'loopback_service_closed':True,'no_gold_loaded':True,'finished':time.time()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=parent.MODELS,required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--allow-private-data',action='store_true');a=p.parse_args()
    if not a.allow_private_data:raise ValueError('Explicit private-data authorization required')
    main(a.model,a.port)
