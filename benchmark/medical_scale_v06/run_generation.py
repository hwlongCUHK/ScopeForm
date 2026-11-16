"""Checkpointed local-model task construction, not benchmark evaluation.

Default preview is offline. Live mode requires an explicit loopback binding and
private-data flag; never falls back to a commercial endpoint or repeats bad labels.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import urllib.parse
import urllib.request

from validate_generated import validate
from token_count import count_chat_tokens

ROOT=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


@contextmanager
def run_lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as f:
        if f.tell()==0:f.write(b'1');f.flush()
        f.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as e:raise RuntimeError('This run is already active; no requests started') from e
        try:yield
        finally:
            f.seek(0)
            if os.name=='nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f,fcntl.LOCK_UN)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


def local_binding(config):
    c=config['construction_model'];base=c['base_url'].rstrip('/');url=urllib.parse.urlsplit(base)
    if url.scheme not in ('http','https') or url.hostname not in ('127.0.0.1','localhost','::1') or url.username or url.password or url.query or url.fragment:
        raise ValueError('Only explicit loopback local construction endpoints are allowed')
    model=c['served_model_name']
    if not isinstance(model,str) or not model:raise ValueError('Missing served model name')
    token=os.environ.get(c.get('api_key_env') or '',c.get('localhost_token','EMPTY'))
    return base,model,token


def load_requests(path):
    with path.open(encoding='utf-8') as f:rows=[json.loads(line) for line in f]
    assert len(rows)==120 and len({r['source_id'] for r in rows})==120
    assert all(r['split']=='development' and r['generation_only_not_solver_evaluation'] is True for r in rows)
    return rows


def small_batch(jobs,pool,count):
    """Score-blind, predeclared source order; balance modality and length first."""
    if count is None:return jobs
    if count<2 or count>len(jobs):raise ValueError('Small batch must contain 2..120 sources')
    catalog={s['source_id']:s for s in read(pool/'manifest.json')['source_catalog']}
    buckets={(m,b):[] for b in (0,1,2) for m in ('dialogue','drug')}
    for j in sorted(jobs,key=lambda j:j['source_id']):
        s=catalog[j['source_id']];buckets[(s['modality'],s['length_band'])].append(j)
    result=[]
    while len(result)<count:
        for key in buckets:
            if buckets[key] and len(result)<count:result.append(buckets[key].pop(0))
    return result


def token_preflight(config,jobs,max_output=16384):
    """Use the actual local checkpoint tokenizer; never estimate by characters."""
    from transformers import AutoTokenizer
    c=config['construction_model'];path=Path(c['model_path_on_user_server'])
    if not path.is_dir():raise ValueError('Actual local tokenizer checkpoint is required')
    tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True,trust_remote_code=False)
    limit=c.get('max_model_len',65536)
    if type(limit) is not int or limit<max_output:raise ValueError('Invalid context limit')
    rows=[]
    for j in jobs:
        ids=tokenizer.apply_chat_template(j['messages'],tokenize=True,add_generation_prompt=True)
        count=count_chat_tokens(ids)
        row={'source_id':j['source_id'],'input_tokens':count,'max_output_tokens':max_output,'max_model_len':limit}
        if count+max_output>limit:raise ValueError('Context capacity exceeded; no truncation: '+j['source_id'])
        rows.append(row)
    return rows


def generate_one(job,source,out,scope,base,token,retry_transport):
    folder=out/'sources'/job['source_id'];terminal=folder/'status.json'
    if terminal.exists():
        old=read(terminal)
        if not (retry_transport and old['status']=='transport_error'):return old
    attempt=1
    while (folder/f'attempt_{attempt:03}').exists():attempt+=1
    attempt_dir=folder/f'attempt_{attempt:03}';attempt_dir.mkdir(parents=True)
    body={'model':scope['model'],'messages':job['messages'],'temperature':1,'reasoning_effort':'medium',
        'max_tokens':16384,'stream':False}
    dump(attempt_dir/'request.private.json',body)
    request=urllib.request.Request(base+'/chat/completions',data=json.dumps(body,ensure_ascii=False).encode(),
        headers={'Content-Type':'application/json','Authorization':'Bearer '+token},method='POST')
    row={'source_id':job['source_id'],'source_cluster':job['source_cluster'],'split':'development','attempt':attempt,
        'kind':'task_construction_not_solver_evaluation','status':'pending','verified_silver_variants':0,'clinical_gold':False}
    try:
        with urllib.request.build_opener(NoRedirect()).open(request,timeout=1800) as response:
            raw=json.load(response)
        dump(attempt_dir/'response.private.json',raw)
    except Exception as e:
        row.update(status='transport_error',error_type=type(e).__name__)
    else:
        try:
            choice=raw['choices'][0]
            if choice.get('finish_reason')=='length':raise ValueError('Construction output reached completion cap')
            content=choice['message']['content']
            if not isinstance(content,str):raise ValueError('Missing construction content')
            content=re.sub(r'^```(?:json)?\s*|\s*```$','',content.strip())
            candidate=json.loads(content);report=validate(source,candidate)
            dump(attempt_dir/'candidate.private.json',candidate);dump(attempt_dir/'automatic_validation.json',report)
            row.update(status='silver_candidate_machine_verified',verified_silver_variants=len(report['variants']),
                model_response_identity=raw.get('model'),next_stage='compile_executable_tasks_and_oracle_check_before_evaluation')
        except (ValueError,KeyError,TypeError,IndexError) as e:
            row.update(status='construction_invalid_preserved_no_quality_retry',error_type=type(e).__name__,error=str(e)[:220])
    dump(terminal,row);return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('preview','live'),default='preview')
    p.add_argument('--requests',type=Path,default=ROOT/'construction/development_120_requests.private.jsonl')
    p.add_argument('--pool',type=Path,default=ROOT/'datasets/source_pool_v06_1000')
    p.add_argument('--config',type=Path,default=ROOT/'model_bindings.example.json');p.add_argument('--run',default='scale_v06_dev120_construction_01')
    p.add_argument('--workers',type=int,choices=(1,2,3,4),default=2);p.add_argument('--allow-private-data',action='store_true');p.add_argument('--retry-transport',action='store_true')
    p.add_argument('--limit',type=int,help='Fixed score-blind modality/length round-robin prefix; not best-of selection');a=p.parse_args()
    jobs=small_batch(load_requests(a.requests),a.pool,a.limit)
    if a.mode=='preview':
        print(json.dumps({'mode':'offline_preview','development_sources':len(jobs),'locked_test_sources':0,'model_calls':0,'completed_tasks':0}));return
    if not a.allow_private_data:raise ValueError('Explicit --allow-private-data required for local teacher inputs')
    if not re.fullmatch(r'scale_v06_[A-Za-z0-9_-]{1,60}',a.run):raise ValueError('Use an isolated scale_v06_ run name')
    base,model,token=local_binding(read(a.config));out=ROOT/'construction/runs'/a.run
    scope={'request_bundle_sha256':digest(a.requests),'source_pool_integrity_sha256':digest(a.pool/'build_integrity.json'),
        'code_sha256':{name:digest(ROOT/name) for name in ('run_generation.py','validate_generated.py','token_count.py')},
        'model':model,'base_url':base,'workers':a.workers,'temperature':1,'reasoning_effort':'medium','max_tokens':16384,
        'seed_not_forwarded':True,'quality_retry':False,'purpose':'silver_task_construction_not_model_evaluation','source_count':len(jobs),
        'source_ids':[j['source_id'] for j in jobs],'selection':'score_blind_modality_length_round_robin_before_inference'}
    with run_lock(out/'run.lock'):
        if (out/'plan.json').exists():
            if read(out/'plan.json')!=scope:raise ValueError('Existing construction scope differs; do not overwrite it')
        else:dump(out/'plan.json',scope)
        integrity=read(a.pool/'build_integrity.json')
        for relative in ('manifest.json','sources.private.jsonl'):
            if digest(a.pool/relative)!=integrity[relative]:raise ValueError('Frozen source pool changed')
        existing={j['source_id']:read(out/'sources'/j['source_id']/'status.json') for j in jobs if (out/'sources'/j['source_id']/'status.json').exists()}
        pending=[j for j in jobs if j['source_id'] not in existing or (a.retry_transport and existing[j['source_id']]['status']=='transport_error')]
        if not pending:
            rows=list(existing.values());dump(out/'summary.json',{'processed_sources':len(rows),'machine_verified_silver_variants':sum(r['verified_silver_variants'] for r in rows),
                'executable_benchmark_tasks':0,'solver_evaluation_conditions':0,'clinical_gold':False,'rows':rows})
            print(json.dumps({'reused_terminal_sources':len(rows),'new_model_calls':0,'not_solver_results':True}));return
        # No inference or data submission before service/model identity preflight.
        request=urllib.request.Request(base+'/models',headers={'Authorization':'Bearer '+token})
        with urllib.request.build_opener(NoRedirect()).open(request,timeout=15) as response:models=json.load(response)
        if model not in {m.get('id') for m in models.get('data',[])}:raise ValueError('Served model not listed; no construction requests started')
        dump(out/'tokenizer_preflight.json',token_preflight(read(a.config),pending))
        selected={j['source_id'] for j in pending};sources={}
        with (a.pool/'sources.private.jsonl').open(encoding='utf-8') as f:
            for line in f:
                s=json.loads(line)
                if s['source_id'] in selected:sources[s['source_id']]=s
        for job in pending:
            source=sources[job['source_id']]
            if source['split']!='development' or source['source_cluster']!=job['source_cluster']:raise ValueError('Development-source identity/split mismatch')
            expected={k:source[k] for k in ('source_id','modality','documents')}
            if json.loads(job['messages'][1]['content'])!=expected:raise ValueError('Prepared teacher material differs from frozen source')
        results=[r for sid,r in existing.items() if sid not in selected]
        with ThreadPoolExecutor(max_workers=a.workers) as executor:
            futures=[executor.submit(generate_one,j,sources[j['source_id']],out,scope,base,token,a.retry_transport) for j in pending]
            for future in as_completed(futures):
                row=future.result();results.append(row);dump(out/'construction_status.json',{'processed_sources':len(results),'planned_sources':len(jobs),'rows':results,'not_solver_results':True})
                print(json.dumps({'processed_sources':len(results),'planned_sources':len(jobs),'source_id':row['source_id'],'status':row['status']},ensure_ascii=False),flush=True)
        dump(out/'summary.json',{'processed_sources':len(results),'machine_verified_silver_variants':sum(r['verified_silver_variants'] for r in results),
            'executable_benchmark_tasks':0,'solver_evaluation_conditions':0,'clinical_gold':False,'rows':results})


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    main()
