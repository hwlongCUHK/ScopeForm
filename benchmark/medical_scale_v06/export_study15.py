"""Strict prediction-only finite-batch export; no clinical text or reasoning."""
import json
from pathlib import Path
import zipfile
from run_generation import ROOT,read,dump,digest
from pilot_runtime import fingerprint
from study15 import MODELS,SLUGS,STUDY,INPUTS


def main():
    dest=ROOT/'exports/study15_current_batch_v1';dest.mkdir(parents=True,exist_ok=True)
    queue=ROOT/'runs/confirmation15_queue_01';complete=read(queue/'complete.json')
    if complete['models_complete']!=3 or complete['models_failed'] or not complete['all_owned_resources_clean']:raise ValueError('Batch not complete and clean')
    actors=[json.loads(x) for x in (INPUTS/'actors.private.jsonl').read_text(encoding='utf-8').splitlines()]
    anchor_ids={b['anchor_id'] for a in actors for c in [a['initial']]+[e['controller'] for e in a['events']] for d in c['documents'] for b in d['blocks']}
    document_ids={d['document_id'] for a in actors for c in [a['initial']]+[e['controller'] for e in a['events']] for d in c['documents']}
    item_ids={i['item_id'] for a in actors for i in a['initial']['claims']}
    files={}
    def safe_refs(value):
        if isinstance(value,list):return [safe_refs(x) for x in value]
        if not isinstance(value,dict):return value
        result={k:safe_refs(v) for k,v in value.items()}
        if 'evidence_ids' in result:result['evidence_ids']=[r if r in anchor_ids else 'INVALID_REF_SHA256_'+fingerprint(r) for r in result['evidence_ids']]
        if 'item_id' in result and result['item_id'] not in item_ids:result['item_id']='INVALID_ITEM_SHA256_'+fingerprint(result['item_id'])
        return result
    def emit(name,value):
        raw=(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode();files[name]=raw
    emit('queue_complete.json',complete);emit('queue_state.json',read(queue/'state.json'))
    for model in MODELS:
        slug=SLUGS[model];out=ROOT/'runs'/('confirmation15_'+slug+'_01')
        for name in ('plan.json','preflight.json','checkpoint_metadata.json','complete.json'):emit(slug+'/'+name,read(out/name))
        for part,planned in (('main',2520),('robustness1',390),('robustness2',390)):
            rows=read(out/part/'results.json')
            if len(rows)!=planned:raise ValueError('Incomplete phase count')
            pairing={}
            for row in rows:
                if row['event_id']=='initial':
                    checkpoint=read(out/part/'phases'/row['source_id']/'initial__initial/checkpoint.private.json')
                    pairing[row['source_id']]={'initial_world_sha256':fingerprint(checkpoint['world']),
                        'initial_checkpoint_file_sha256':digest(out/part/'phases'/row['source_id']/'initial__initial/checkpoint.private.json'),
                        'actual_submitted':checkpoint['world']['done']}
                if not set(row.get('observed',[]))<=anchor_ids or not set(row.get('seen_documents',[]))<=document_ids:raise ValueError('Unexpected observation identifiers')
            emit(slug+'/'+part+'/results.json',safe_refs(rows));emit(slug+'/'+part+'/pairing_audit.json',pairing)
            emit(slug+'/'+part+'/complete.json',read(out/part/'complete.json'))
        for name in ('results.json','complete.json','job_index.json'):emit(slug+'/static/'+name,read(out/'static'/name))
    archive=dest/'predictions.zip'
    if archive.exists():
        old=read(dest/'manifest.json')
        if digest(archive)!=old['archive_sha256']:raise ValueError('Existing export differs')
        print(json.dumps({'export':str(dest),'manifest':old}));return
    import hashlib
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in sorted(files.items()):z.writestr(name,raw)
    manifest={'study_protocol_sha256':digest(STUDY),'archive_sha256':digest(archive),
        'files':{n:hashlib.sha256(b).hexdigest() for n,b in files.items()},
        'private_text_or_gold_or_thinking_included':False,'full_study_complete':False,'web_controls_pending':True}
    dump(dest/'manifest.json',manifest);print(json.dumps({'export':str(dest),'manifest':manifest}))


if __name__=='__main__':main()
