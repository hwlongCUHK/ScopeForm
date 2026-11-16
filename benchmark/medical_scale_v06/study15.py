"""Prospective common-harness main/static/robustness study, gold-free server.

No evaluator import or medical standard labels are needed on the server.
The Web-service control is a separate predeclared batch, not marked done here.
"""
import argparse
from collections import Counter
import copy
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED,as_completed
import json
from pathlib import Path
import time
import urllib.error

from run_generation import ROOT,read,dump,digest,run_lock
from pilot_runtime import fingerprint
from token_count import count_chat_tokens
import confirmation14_workflow as h

INPUTS=ROOT/'confirmation15_inputs_private'
STUDY=ROOT/'study15_protocol.json'
MODELS=('Qwen3.5-9B','Qwen3.8-27B','GLM-4-9B-0414')
SLUGS=dict(zip(MODELS,('q9b','q27b','glm9b')))
SOURCE_METHODS=('evidence_constraint','labels_only','fresh')
CODE=('study15.py','queue_study15.py','export_study15.py','confirmation14_workflow.py','workflow12.py',
    'run_workflow11.py','workflow11_runtime.py','medical_runtime07.py','paired_runtime06.py','pilot_runtime.py',
    'medical_runtime04.py','typed_client03.py','token_count.py','run_generation.py','service_baseline10.py',
    'baseline10_common.py','service_expansion08.py')


def static_jobs(records):
    jobs={}
    for record in records:
        controller=record['controller']
        for condition in ('full_context','source_withheld','stem_only'):
            docs=controller['documents'] if condition=='full_context' else [d for d in controller['documents'] if d.get('origin_kind')=='explicit_synthetic_chartlets'] if condition=='source_withheld' else []
            actor={'claims':controller['claims'],'scenario_contract':controller['scenario_contract'],'materials':docs}
            key=fingerprint({'condition':condition,'actor':actor})
            if key not in jobs:jobs[key]={'key':key,'condition':condition,'actor':actor,'members':[]}
            jobs[key]['members'].append({'source_id':record['source_id'],'packet_id':record['packet_id']})
    return sorted(jobs.values(),key=lambda j:j['key'])


def static_schema(ids):
    return {'type':'object','properties':{'items':{'type':'array','minItems':len(ids),'maxItems':len(ids),'items':{
        'type':'object','properties':{'item_id':{'type':'string','enum':ids},
            'label':{'type':'string','enum':['supported','refuted','not_determined']},
            'evidence_ids':{'type':'array','maxItems':0,'items':{'type':'string'}}},
        'required':['item_id','label','evidence_ids'],'additionalProperties':False}},
        'action_status':{'type':'string','enum':['not_requested']}},'required':['items','action_status'],'additionalProperties':False}


def static_one(job,out,port,model,tokenizer,settings):
    folder=out/'actor_requests'/job['key'];rowfile=folder/'row.json'
    if rowfile.exists():return read(rowfile)
    row={k:copy.deepcopy(job[k]) for k in ('key','condition','members')}
    row.update(submitted=False,status='not_started',requests=0,predictions={})
    ids=[c['item_id'] for c in job['actor']['claims']]
    messages=[{'role':'user','content':h.POLICY+'\n本次仅预测给定全部陈述在完整资料下的标签，不调用工具、不执行保存。即使资料被移除也预测相同目标，不自动改成未知。'
        +'只返回items和action_status。每项包含item_id,label,evidence_ids，evidence_ids为空数组，action_status=not_requested。\n'
        +json.dumps(job['actor'],ensure_ascii=False)}]
    response=folder/'response.private.json';dispatch=folder/'dispatch.json'
    try:
        ntokens=count_chat_tokens(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False))
        row['input_tokens']=ntokens
        if ntokens+settings['max_output_tokens']>settings['context_limit']:row['status']='context_limit_no_silent_truncation'
        else:
            body={'model':model,'messages':messages,'temperature':settings['temperature'],'seed':settings['seed'],
                'max_tokens':settings['max_output_tokens'],'stream':False,'chat_template_kwargs':{'enable_thinking':False},
                'response_format':{'type':'json_schema','json_schema':{'name':'medical_content_only','strict':True,'schema':static_schema(ids)}}}
            dump(folder/'request.private.json',body)
            if response.exists():raw=read(response);row['requests']=1
            elif dispatch.exists():raw=None;row.update(status='indeterminate_inflight_no_resample',requests=1)
            else:
                dump(dispatch,{'request_sha256':fingerprint(body),'input_tokens':ntokens,'time':time.time()})
                row['requests']=1;raw=h.post(port,body);dump(response,raw)
            if raw is not None:
                row['usage']=raw.get('usage');choice=raw['choices'][0]
                if choice['finish_reason']=='length':row['status']='output_limit_not_content_error'
                elif choice['finish_reason']!='stop':row['status']='response_not_completed'
                else:
                    value=json.loads(choice['message']['content']);items=value.get('items',[])
                    valid=(len(items)==len(ids) and len({i['item_id'] for i in items})==len(ids) and
                        {i['item_id'] for i in items}==set(ids) and value.get('action_status')=='not_requested' and
                        all(i['label'] in ('supported','refuted','not_determined') and i['evidence_ids']==[] for i in items))
                    if valid:row.update(submitted=True,status='submitted',predictions={i['item_id']:i['label'] for i in items})
                    else:row['status']='invalid_answer_schema'
    except urllib.error.HTTPError as error:
        dump(folder/'http_error.private.json',{'status':error.code,'body':error.read(12000).decode(errors='replace')})
        row['status']='http_'+str(error.code)
    except Exception as error:row['status']='runner_or_transport_'+type(error).__name__
    dump(rowfile,row);return row


def preflight(port,model,tokenizer,settings,out):
    if (out/'preflight.json').exists():
        if not read(out/'preflight.json')['passed']:raise ValueError('Preserve prior failed interface test')
        return
    public={'source_id':'SYNTHETIC_INTERFACE15','modality':'drug','environment':'E3','key_revokes_write':False,
        'task_id':'SYNTHETIC_INTERFACE15','task_version':'synthetic_only','documents':[{'document_id':'SYN-DOC',
        'title':'Synthetic counter record, not medical data','blocks':[{'anchor_id':'SYN-A','label':'counter','text':'Synthetic counter is2.'}]}],
        'claims':[{'item_id':'C1','claim_type':'synthetic_counter','statement':'The synthetic counter is2.'}],
        'scenario_contract':'Nonmedical interface test only. No real person or clinical record.',
        'visibility':{'initial':['SYN-A'],'updated':['SYN-A']}}
    scripted=h.World(public);scripted.call('read_source',{'document_id':'SYN-DOC'})
    state=scripted.call('read_state',{});items=[{'item_id':'C1','label':'supported','evidence_ids':['SYN-A']}]
    scripted.call('save_summary',{'items':items,'expected_version':state['version'],'idempotency_key':'synthetic_preflight15'})
    scripted.call('read_state',{});scripted.call('submit_answer',{'items':items,'action_status':scripted.action()})
    if not scripted.done:raise ValueError('Synthetic scripted harness failure')
    actual=h.World(public);requests=0
    for i,step in enumerate(scripted.trace):
        action={'name':step['tool'],'arguments':step['arguments']};folder=out/'nonmedical_preflight'/str(i)
        messages=[{'role':'user','content':'Nonmedical JSON interface test. Return exactly the following action object; do not change any field.\n'+json.dumps({'action':action},ensure_ascii=False)}]
        request,_=h.body(messages,actual.tools(),model,tokenizer,{**settings,'max_output_tokens':1024})
        response=folder/'response.private.json'
        if response.exists():raw=read(response)
        elif (folder/'dispatch.json').exists():raise ValueError('Ambiguous preflight dispatch, no resample')
        else:
            dump(folder/'dispatch.json',{'request_sha256':fingerprint(request)})
            raw=h.post(port,request);dump(response,raw);requests+=1
        decoded=h.decode(raw)
        if decoded!=action:
            dump(out/'preflight.json',{'passed':False,'contains_medical_data':False,'failure':'synthetic_action_mismatch'})
            raise ValueError('Model does not follow the common JSON action interface')
        actual.call(decoded['name'],decoded['arguments'])
    if not actual.done or actual.last_readback!=scripted.last_readback:raise ValueError('Synthetic nested write/readback failed')
    dump(out/'preflight.json',{'passed':True,'contains_medical_data':False,'serial_actions':len(scripted.trace),
        'nested_write_and_readback':True,'requests':requests,'clinical_accuracy_measured':False})


def workflow(actors,out,port,model,tokenizer,settings,methods):
    expected=len(actors)*(1+4*len(methods))
    if (out/'complete.json').exists():
        if read(out/'complete.json')['recorded']!=expected:raise ValueError('Existing workflow count differs')
        return
    rows=[]
    with ThreadPoolExecutor(max_workers=settings['workers_per_model']) as pool:
        futures={pool.submit(h.one,a,None,'initial',out,port,model,tokenizer,settings):(a,True) for a in sorted(actors,key=lambda a:fingerprint(a['source_id']))}
        while futures:
            ready,_=wait(futures,return_when=FIRST_COMPLETED)
            for future in ready:
                actor,initial=futures.pop(future);row,world=future.result();rows.append(row)
                if initial:
                    jobs=sorted([(e,m) for e in actor['events'] for m in methods],key=lambda x:fingerprint(actor['source_id']+x[0]['event_id']+x[1]))
                    for event,method in jobs:futures[pool.submit(h.one,actor,event,method,out,port,model,tokenizer,settings,world)]=(actor,False)
                dump(out/'results.json',rows)
                dump(out/'progress.json',{'planned':expected,'recorded':len(rows),'submitted':sum(r['submitted'] for r in rows),'statuses':dict(Counter(r['status'] for r in rows)),'updated':time.time()})
    if len(rows)!=expected or len({(r['source_id'],r['event_id'],r['arm']) for r in rows})!=expected:raise ValueError('Incomplete workflow matrix')
    dump(out/'complete.json',{'planned':expected,'recorded':len(rows),'submitted':sum(r['submitted'] for r in rows),'all_accounted':True})


def run(model,port):
    study=read(STUDY);manifest=read(INPUTS/'actor_manifest.json')
    for name,sha in manifest['files'].items():
        if name not in ('actors.private.jsonl','static_actors.private.jsonl') or digest(INPUTS/name)!=sha:raise ValueError('Actor bundle mismatch')
    if study['actor_manifest_sha256']!=digest(INPUTS/'actor_manifest.json'):raise ValueError('Study actor identity differs')
    for name,sha in study['runtime_code_sha256'].items():
        if digest(ROOT/name)!=sha:raise ValueError('Frozen runtime code changed')
    out=ROOT/'runs'/('confirmation15_'+SLUGS[model]+'_01')
    actors=[json.loads(x) for x in (INPUTS/'actors.private.jsonl').read_text(encoding='utf-8').splitlines()]
    statics=[json.loads(x) for x in (INPUTS/'static_actors.private.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(actors)!=120 or len(statics)!=360:raise ValueError('Full source matrix required')
    settings={**study['decoding'],'context_limit':study['model_context_limits'][model]}
    plan={'model':model,'port':port,'study_protocol_sha256':digest(STUDY),'actor_manifest_sha256':digest(INPUTS/'actor_manifest.json'),
        'harness':'shared_serial_json_action14','main_phases':2520,'robustness_phases':780,'static_logical_slots':1080,
        'web_substudy':'separate_pending_batch_not_complete_here','settings':settings,'gold_loaded_on_server':False}
    with run_lock(out/'run.lock'):
        if (out/'plan.json').exists() and read(out/'plan.json')!=plan:raise ValueError('Existing run plan changed')
        dump(out/'plan.json',plan)
        from transformers import AutoTokenizer
        path=Path('/data/zhanchen/models')/model
        tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        if not tokenizer.chat_template and (path/'chat_template.jinja').exists():tokenizer.chat_template=(path/'chat_template.jinja').read_text()
        dump(out/'checkpoint_metadata.json',{n:digest(path/n) for n in ('config.json','tokenizer_config.json','tokenizer.json','chat_template.jinja','model.safetensors.index.json') if (path/n).exists()})
        preflight(port,model,tokenizer,settings,out)
        workflow(actors,out/'main',port,model,tokenizer,settings,h.METHODS)
        jobs=static_jobs(statics);dump(out/'static/job_index.json',[{k:val for k,val in j.items() if k!='actor'} for j in jobs])
        if not (out/'static/complete.json').exists():
            rows=[]
            with ThreadPoolExecutor(max_workers=4) as pool:
                for f in as_completed([pool.submit(static_one,j,out/'static',port,model,tokenizer,settings) for j in jobs]):
                    rows.append(f.result());dump(out/'static/results.json',rows)
                    dump(out/'static/progress.json',{'planned_unique':len(jobs),'recorded':len(rows),'submitted':sum(r['submitted'] for r in rows),'updated':time.time()})
            if len(rows)!=len(jobs):raise ValueError('Missing static jobs')
            dump(out/'static/complete.json',{'recorded':len(rows),'logical_slots':sum(len(j['members']) for j in jobs),'all_accounted':True})
        chosen=set(study['robustness']['source_ids'])
        for replicate in (1,2):
            batch=[]
            for actor in actors:
                if actor['source_id'] not in chosen:continue
                a=copy.deepcopy(actor)
                for controller in [a['initial']]+[e['controller'] for e in a['events']]:
                    docs=controller['documents']
                    controller['documents']=list(reversed(docs)) if replicate==1 else sorted(docs,key=lambda d:fingerprint(['replicate2',actor['source_id'],d['document_id']]))
                batch.append(a)
            if len(batch)!=30:raise ValueError('Robustness metadata subset differs')
            workflow(batch,out/('robustness'+str(replicate)),port,model,tokenizer,settings,SOURCE_METHODS)
        dump(out/'complete.json',{'model':model,'main_phases':2520,'robustness_phases':780,'static_logical_slots':1080,
            'static_unique_requests':len(jobs),'all_current_batch_accounted':True,'web_controls_complete':False,
            'not_full_16020_study_completion':True,'gold_loaded_on_server':False,'finished':time.time()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--allow-private-data',action='store_true');a=p.parse_args()
    if not a.allow_private_data:raise ValueError('Explicit private-source authorization required')
    run(a.model,a.port)
