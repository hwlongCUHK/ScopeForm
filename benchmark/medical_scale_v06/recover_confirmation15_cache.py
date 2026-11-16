"""Adopt qualified63 unchanged and recover existing author responses only.

No model endpoint, generation, source replacement, evaluator upload or rewrite
of any13/14 artifact. This does not itself qualify or seal replacement tasks.
"""
import argparse
import getpass
import hashlib
import json
from pathlib import Path
import re

import paramiko
import confirmation14 as parent
from author_gateway14 import job_id
from run_generation import ROOT, read, dump, digest

WORK=ROOT/'construction/confirmation15_replacement120_v1'
OUT=ROOT.parent/'output/medical15_replacement_20261022'
OLD=ROOT/'construction/confirmation14_120_v1'
JOURNAL='/data/zhanchen/cpm_medical06_20261020/medical_scale_v06/construction/confirmation14_author_journal_01'


def prepare():
    parent.c.verify_tree(parent.c.SOURCES)
    old_protocol=read(OLD/'protocol.json')
    for n,sha in old_protocol['code_sha256'].items():
        if digest(ROOT/n)!=sha:raise ValueError('Parent code changed')
    sources={s['source_id']:s for s in parent.c.load_reserved()}
    adopted=[];missing=[]
    for sid,source in sources.items():
        folder=OLD/'sources'/sid;row=read(folder/'result.json')
        if row['machine_valid']:
            spec=folder/'accepted_spec.private.json';valid,errors=parent.c.check_items(read(spec),source)
            if not valid:raise ValueError('Original qualified rule failed: '+sid)
            adopted.append({'source_id':sid,'source_cluster':source['source_cluster'],'modality':source['modality'],
                'length_band':source['length_band'],'parent_spec_sha256':digest(spec),'parent_result_sha256':digest(folder/'result.json')})
            parent.c.frozen(WORK/'adopted63'/sid/'accepted_spec.private.json',spec.read_bytes())
        elif row['construction_kind']=='execution_failure_not_source_quality':
            for path in sorted(folder.glob('candidate_repair_*/*/dispatch.json')):
                if (path.parent/'response.private.json').exists():continue
                dispatch=read(path);request=read(path.parent/'request.private.json')
                if job_id(request)!=dispatch['job_id']:raise ValueError('Recovery request hash mismatch')
                missing.append({'source_id':sid,'job_id':dispatch['job_id'],
                    'parent_dispatch':path.relative_to(ROOT).as_posix(),'stage':path.parent.name,
                    'parent_request_sha256':digest(path.parent/'request.private.json')})
    if len(adopted)!=63 or len(missing)!=5:raise ValueError('Unexpected parent closeout')
    plan={'version':'confirmation15-existing-cache-recovery-only-v1','user_authorized_route':2,
        'code_sha256':digest(Path(__file__)),'parent_closeout_sha256':digest(parent.OUT/'closeout_delivery_manifest.json'),
        'parent_protocol_sha256':digest(OLD/'protocol.json'),'adopted':adopted,'recovery_jobs':missing,
        'remote_journal':JOURNAL,'new_model_calls':0,'new_author_generations':0,
        'qualification':'Recover bytes, not a successful clinical task. Truncated responses remain truncated; no repair generation in this phase.',
        'source_selection':'Replacement eligibility/order will be separately frozen before new source construction; never use solver predictions.'}
    parent.c.frozen(WORK/'cache_recovery_plan.json',plan)
    parent.c.frozen(WORK/'code_snapshot/recover_confirmation15_cache.py',Path(__file__).read_bytes())
    dump(OUT/'RUN_STATUS.json',{'stage':'qualified63_adopted_cache_recovery_pending','adopted_sources':63,
        'replacement_source_inputs_frozen':False,'tasks_frozen':False,'solver_calls':0,'author_generation_calls':0})
    return plan


def recover():
    plan=read(WORK/'cache_recovery_plan.json')
    if digest(Path(__file__))!=plan['code_sha256']:raise ValueError('Recovery code changed')
    ssh=paramiko.SSHClient();ssh.load_system_host_keys();ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
    secret=getpass.getpass('SSH password (memory only): ')
    rows=[]
    try:
        ssh.connect('122.225.39.134',port=2222,username='zhanchen',password=secret,
            look_for_keys=False,allow_agent=False,timeout=20,banner_timeout=20,auth_timeout=20)
        secret=None
        with ssh.open_sftp() as sftp:
            for job in plan['recovery_jobs']:
                key=job['job_id']
                if not re.fullmatch('[0-9a-f]{64}',key):raise ValueError('Unsafe job ID')
                with sftp.open(JOURNAL+'/'+key+'/response.private.json','rb') as f:raw=f.read()
                value=json.loads(raw);reasons=[c.get('finish_reason') for c in value.get('choices',[])]
                parent.c.frozen(WORK/'recovered_jobs'/key/'response.private.json',raw)
                parent.c.frozen(WORK/'recovered_jobs'/key/'mapping.LOCAL_ONLY.json',job)
                rows.append({**job,'response_sha256':hashlib.sha256(raw).hexdigest(),'finish_reasons':reasons,
                    'complete_json_candidate':bool(reasons) and all(x=='stop' for x in reasons),
                    'not_automatically_qualified':True})
    finally:ssh.close();secret=None
    result={'jobs':rows,'responses_recovered':len(rows),'truncated':sum('length' in r['finish_reasons'] for r in rows),
        'complete_responses':sum(r['complete_json_candidate'] for r in rows),'new_model_calls':0,
        'parent_outputs_changed':False,'evaluator_targets_uploaded':False}
    parent.c.frozen(WORK/'cache_recovery_complete.json',result)
    dump(OUT/'RUN_STATUS.json',{'stage':'existing_cache_recovered_before_replacement_screen','adopted_sources':63,
        'cache_responses_recovered':len(rows),'complete_cached_responses':result['complete_responses'],
        'replacement_source_inputs_frozen':False,'tasks_frozen':False,'solver_calls':0,'author_generation_calls':0})
    print(json.dumps({k:v for k,v in result.items() if k!='jobs'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','recover'));args=p.parse_args()
    if args.action=='prepare':
        plan=prepare();print(json.dumps({'adopted':len(plan['adopted']),'existing_jobs_to_recover':len(plan['recovery_jobs']),'new_model_calls':0}))
    else:recover()
