"""Local-only frozen main/static/robustness analysis. No network or model API.

Standards stay on this computer. Coverage-adjusted0 means no achieved correct
output, never an assertion that an interface failure is a wrong medical answer.
"""
import argparse
from collections import Counter,defaultdict
import itertools
import json
import math
from pathlib import Path
import statistics

from run_generation import ROOT,read,dump,digest
from pilot_runtime import fingerprint
from score_workflow11_local import score_row,mean_ci
from seal_confirmation15 import DATASET
import study15 as s

LABELS=('supported','refuted','not_determined')
EVENTS=('key_mixed','single_definite','single_uncertainty','neutral')


def jsonl(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
def avg(v):return statistics.fmean(v) if v else None


def content(row):
    return sum(i['content'] for i in row['score']['items'])/row['items_planned'] if row['submitted'] else 0.


def sign_test(values):
    pos=sum(v>1e-12 for v in values);neg=sum(v< -1e-12 for v in values);n=pos+neg
    return {'wins':pos,'losses':neg,'ties':len(values)-n,
        'paired_sign_p':min(1.,2*sum(math.comb(n,k) for k in range(min(pos,neg)+1))/2**n) if n else 1.}


def holm(tests):
    running=0.
    for k,t in enumerate(sorted(tests,key=lambda v:v['paired_sign_p'])):
        running=max(running,min(1.,(len(tests)-k)*t['paired_sign_p']));t['holm_p']=running


def enrich(rows,actors,targets,pairing):
    bases={r['source_id']:r for r in rows if r['event_id']=='initial'}
    if set(bases)!=set(actors):raise ValueError('Actual initial matrix incomplete')
    result=[]
    for original in rows:
        r=dict(original);sid=r['source_id'];t=targets[sid];a=actors[sid];base=bases[sid];event=r['event_id']
        expected=t['initial'] if event=='initial' else next(e['items'] for e in t['events'] if e['event_id']==event)
        controller=a['initial'] if event=='initial' else next(e['controller'] for e in a['events'] if e['event_id']==event)
        if r.get('public_hash') and r['public_hash']!=fingerprint(controller):raise ValueError('Public task mismatch')
        if event!='initial':
            if r['initial_checkpoint_hash']!=pairing[sid]['initial_world_sha256']:raise ValueError('Not the same actual initial checkpoint')
            if base['submitted']:
                if r['pre_state_hash']!=fingerprint(base['state']) or r['phase_start_state']!=base['state']:raise ValueError('Persistent real base not preserved')
            elif r['status']!='blocked_actual_initial_not_submitted':raise ValueError('Dependent branch silently replaced initial')
        r.update(construction_subset=t['construction_subset'],source_cluster=t['source_cluster'],modality=t['modality'],
            items_planned=len(expected),environment=controller['environment'],
            score=score_row(r,expected,controller,t['initial'],base))
        initial={g['item_id']:g['label'] for g in t['initial']}
        r['planned_changed']=sum(g['label']!=initial[g['item_id']] for g in expected)
        r['planned_stable']=len(expected)-r['planned_changed']
        r['gold_transition_local']=[{'item_id':g['item_id'],'from':initial[g['item_id']],'to':g['label']} for g in expected]
        result.append(r)
    return result


def summary(rows):
    done=[r for r in rows if r['submitted']];by=defaultdict(list)
    for r in rows:by[r['source_id']].append(content(r))
    out={'planned':len(rows),'submitted':len(done),'statuses':dict(Counter(r['status'] for r in rows)),
        'content_source_macro_coverage_adjusted':mean_ci({sid:avg(v) for sid,v in by.items()}),
        'content_judgments_correct':sum(sum(i['content'] for i in r['score']['items']) for r in done),
        'judgments_submitted':sum(r['items_planned'] for r in done),'judgments_planned':sum(r['items_planned'] for r in rows),
        'exact_content_packets':sum(r['score']['content'] for r in done),
        'evidence_packets':sum(r['score']['evidence'] for r in done),'strict_packets':sum(r['score']['strict'] for r in done),
        'actual_state_correct':sum(r['score']['actual_state'] is True for r in done),
        'readback_correct':sum(r['score']['readback'] is True for r in done),
        'readback_applicable_submitted':sum(r['score']['readback'] is not None for r in done),
        'requests':sum(r['requests'] for r in rows),'tools':sum(r.get('tool_calls',0) for r in rows),
        'guard_rejections':sum(r.get('guard_rejections',0) for r in rows),
        'unauthorized_attempts':sum(r['score']['unauthorized_attempts'] for r in done),
        'readback_consistent_but_saved_labels_wrong':sum(r['score']['readback'] is True and r['score']['actual_state'] is False for r in done),
        'scoring_unit':'source cluster; branches/claims are correlated'}
    for category in ('changed','stable','protected_initial_correct','initial_wrong_stable'):
        out[category]={'correct':sum(r['score'][category]['correct'] for r in done),'submitted_total':sum(r['score'][category]['total'] for r in done)}
        if category in ('changed','stable'):out[category]['planned_total']=sum(r['planned_'+category] for r in rows)
    transitions=[]
    for before,after in itertools.permutations(LABELS,2):
        source=defaultdict(list)
        for r in rows:
            scored={i['item_id']:i for i in r['score']['items']} if r['score'] else {}
            for g in r['gold_transition_local']:
                if (g['from'],g['to'])==(before,after):source[r['source_id']].append(int(scored.get(g['item_id'],{}).get('content',False)))
        transitions.append({'from':before,'to':after,'items_planned':sum(map(len,source.values())),
            **mean_ci({sid:avg(v) for sid,v in source.items()})})
    out['six_directions']=transitions
    return out


def static_summary(rows,targets):
    expanded=[]
    for row in rows:
        for member in row['members']:
            t=targets[member['source_id']];packet=next(p for p in t['packets'] if p['packet_id']==member['packet_id'])
            for g in packet['items']:
                pred=row['predictions'].get(g['item_id']) if row['submitted'] else None
                expanded.append({'source_id':t['source_id'],'packet_id':member['packet_id'],'item_id':g['item_id'],
                    'condition':row['condition'],'construction_subset':t['construction_subset'],
                    'gold':g['label'],'prediction':pred,'submitted':row['submitted'],'correct':pred==g['label'],
                    'stable_insufficiency':g.get('stable_insufficiency',False)})
    reports=[]
    for condition in ('full_context','source_withheld','stem_only'):
        for subset in ('all','retained63','replacement57'):
            chosen=[r for r in expanded if r['condition']==condition and (subset=='all' or (r['construction_subset']=='retained_qualified63')==(subset=='retained63'))]
            groups=defaultdict(list)
            for r in chosen:groups[r['source_id']].append(int(r['correct']))
            # Triplet correctness uses a family, not three independent patients.
            families=defaultdict(list)
            for r in chosen:families[(r['source_id'],r['item_id'])].append(r)
            changing=[v for v in families.values() if len({x['gold'] for x in v})==3]
            persistent=[r for r in chosen if r['gold']=='not_determined' and len({x['gold'] for x in families[r['source_id'],r['item_id']]})==1]
            reports.append({'condition':condition,'subset':subset,'judgments':len(chosen),
                'submitted_judgments':sum(r['submitted'] for r in chosen),'correct':sum(r['correct'] for r in chosen),
                'source_macro_coverage_adjusted':mean_ci({sid:avg(v) for sid,v in groups.items()}),
                'varying_three_state_families':len(changing),'all_three_correct':sum(all(x['correct'] for x in v) for v in changing),
                'varying_judgments_correct':sum(x['correct'] for v in changing for x in v),'varying_judgments_total':sum(len(v) for v in changing),
                'stable_insufficiency_correct':sum(r['correct'] for r in persistent),'stable_insufficiency_total':len(persistent),
                'fixed_always_unknown_accuracy':avg([int(r['gold']=='not_determined') for r in chosen])})
    return {'unique_jobs':len(rows),'logical_packets':sum(len(r['members']) for r in rows),
        'statuses':dict(Counter(r['status'] for r in rows)),'reports':reports,'expanded_local_only':expanded}


def main(downloaded,out):
    protocol=read(s.STUDY)
    for n,sha in protocol['local_scoring_code_sha256'].items():
        if digest(ROOT/n)!=sha:raise ValueError('Frozen scoring code changed')
    if digest(DATASET/'build_integrity.json')!=protocol['dataset_integrity_sha256']:raise ValueError('Dataset identity differs')
    for name,sha in read(DATASET/'build_integrity.json').items():
        if digest(DATASET/name)!=sha:raise ValueError('Frozen task changed')
    actors={a['source_id']:a for a in jsonl(DATASET/'actors.private.jsonl')};targets={t['source_id']:t for t in jsonl(DATASET/'evaluator_targets.LOCAL_ONLY.private.jsonl')}
    reports=[];paired=[];interactions=[];scored=[];robust=[];static=[]
    queue=read(downloaded/'queue_complete.json')
    if queue['models_complete']!=3 or queue['models_failed'] or not queue['all_owned_resources_clean']:raise ValueError('Wait for complete clean current batch')
    for model in s.MODELS:
        folder=downloaded/s.SLUGS[model];plan=read(folder/'plan.json')
        if plan['study_protocol_sha256']!=digest(s.STUDY) or plan['gold_loaded_on_server']:raise ValueError('Provenance mismatch')
        if not read(folder/'preflight.json')['passed']:raise ValueError('Interface not validated')
        rows=read(folder/'main/results.json');audits=read(folder/'main/pairing_audit.json')
        if len(rows)!=2520 or len({(r['source_id'],r['event_id'],r['arm']) for r in rows})!=2520:raise ValueError('Main matrix incomplete')
        rr=enrich(rows,actors,targets,audits);scored.extend({'model':model,**r} for r in rr)
        for event in ('initial',)+EVENTS:
            for method in (('initial',) if event=='initial' else s.h.METHODS):
                for subset in ('all','retained63','replacement57'):
                    chosen=[r for r in rr if r['event_id']==event and r['arm']==method and (subset=='all' or (r['construction_subset']=='retained_qualified63')==(subset=='retained63'))]
                    reports.append({'model':model,'event':event,'method':method,'subset':subset,**summary(chosen)})
        for event in EVENTS:
            arms={m:{r['source_id']:r for r in rr if r['event_id']==event and r['arm']==m} for m in s.h.METHODS}
            for left,right in (('evidence_constraint','labels_only'),('evidence_constraint','fresh'),('labels_only','fresh')):
                values={sid:content(arms[left][sid])-content(arms[right][sid]) for sid in actors}
                paired.append({'model':model,'event':event,'left':left,'right':right,**mean_ci(values),**sign_test(list(values.values()))})
            values={sid:content(arms['evidence_constraint'][sid])-content(arms['index_only'][sid])-content(arms['labels_guard'][sid])+content(arms['labels_only'][sid]) for sid in actors}
            interactions.append({'model':model,'event':event,'definition':'(index+guard-index_only)-(labels_guard-labels_only)',**mean_ci(values),**sign_test(list(values.values()))})
        static.append({'model':model,**static_summary(read(folder/'static/results.json'),targets)})
        chosen=set(protocol['robustness']['source_ids'])
        for rep in (1,2):
            batch={sid:__import__('copy').deepcopy(a) for sid,a in actors.items() if sid in chosen}
            for sid,a in batch.items():
                for controller in [a['initial']]+[e['controller'] for e in a['events']]:
                    docs=controller['documents'];controller['documents']=list(reversed(docs)) if rep==1 else sorted(docs,key=lambda d:fingerprint(['replicate2',sid,d['document_id']]))
            raw=read(folder/('robustness'+str(rep))/'results.json')
            if len(raw)!=390:raise ValueError('Robustness trajectory incomplete')
            extra=enrich(raw,batch,{sid:targets[sid] for sid in chosen},read(folder/('robustness'+str(rep))/'pairing_audit.json'))
            for event in EVENTS:
                for method in s.SOURCE_METHODS:
                    rs=[r for r in extra if r['event_id']==event and r['arm']==method]
                    baseline={r['source_id']:r for r in rr if r['event_id']==event and r['arm']==method and r['source_id'] in chosen}
                    diffs={r['source_id']:content(r)-content(baseline[r['source_id']]) for r in rs}
                    robust.append({'model':model,'replicate':rep,'event':event,'method':method,**summary(rs),'paired_to_original':mean_ci(diffs)})
    holm(paired);holm(interactions)
    dump(out/'scored_rows.LOCAL_ONLY.private.json',scored)
    dump(out/'results.json',{'protocol_sha256':digest(s.STUDY),'conditions':reports,'paired_primary36':paired,'factorial_interactions12':interactions,
        'static':static,'robustness':robust,'current_batch_main_static_robustness_finished':True,'web_controls_pending':True,
        'full_16020_complete':False,'clinical_expert_gold':False,'not_original120_all_passed':True})
    dump(out/'verification.json',{'main_phases':7560,'static_logical_slots':3240,'robustness_phases':2340,
        'all_actual_initial_pairs_verified':True,'full_study_complete':False,'web_controls_pending':True,'no_gold_uploaded':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--downloaded',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.downloaded,a.output)
