"""Prospective source-information curation, never solver-accuracy selection.

Retain accepted63 byte-for-byte. Recover/validate the one complete cached draft.
Screen unexposed locked sources with a frozen explicit-clause grammar, then
take the first qualifying candidates in a predeclared hash order/stratum.
All exclusions and construction rejects stay in the attrition ledger.
"""
from collections import Counter,defaultdict
import copy
import json
from pathlib import Path
from unittest.mock import patch

import confirmation14 as prior
import regimen15
from run_generation import ROOT,read,dump,digest
from recover_confirmation15_cache import WORK,OUT,OLD

SOURCES=ROOT/'datasets/medical15_confirmation120_sources_v1'
TARGETS={('dialogue',i):24 for i in range(3)}|{('drug',i):16 for i in range(3)}


def policy():
    return {'version':'confirmation15-source-information-replacement-policy-v1','user_authorized_route':2,
        'original_cohort_integrity_sha256':digest(prior.c.SOURCES/'build_integrity.json'),
        'original_closeout_sha256':digest(prior.OUT/'closeout_delivery_manifest.json'),
        'source_pool_integrity_sha256':digest(prior.c.POOL/'build_integrity.json'),
        'code_sha256':{n:digest(ROOT/n) for n in ('prepare_replacement15.py','regimen15.py','test_regimen15.py')},
        'retained':'Adopt qualified63 unchanged. The one stop-finished cached draft may additionally qualify under the unchanged gates; length-finished responses never count as complete.',
        'replacement_universe':'Only locked_test clusters outside all original120 and previously exposed development/construction sources; no model result or scored row is loaded.',
        'information_sufficiency':'Exact positive native-unit quantities in one unambiguous linked recorded regimen: per-administration+daily-frequency (either order/slash), explicitly divided daily total, or explicit medication course+frequency. Do not use prognosis as duration, planned patient questions as executed exposure, or mix subgroup/drug clauses.',
        'roles':'Doctor advice and named drug instructions, or explicit declarative patient exposure records. Reference role remains recorded; reconciliation is not clinical endorsement or prescribing.',
        'order':'SHA256 encoded [confirmation15-blind-source-order, source_id] within modality x existing length band. No shorter/easier/model-correct preference.',
        'stratum_targets':[{'modality':m,'length_band':b,'sources':n} for (m,b),n in sorted(TARGETS.items())],
        'construction_rejection':'Advance fixed source-blind queue only on unchanged machine provenance/type/three-state/primary-multistep qualification failure; preserve every attempted source and reason.',
        'near_duplicate_policy':'Exact normalized7gram checks against exposed and all original120 plus newly selected; block flags, do not use solver scores.',
        'qualification_sha256':digest(ROOT/'confirmation13.py'),'label_semantics_sha256':digest(ROOT/'scale09_rules_r4.py'),
        'new_author_or_solver_generations':0,'source_holdout_not_rule_family_holdout':True,'clinical_expert_gold':False,
        'curation_bias':'New replacements require explicit quantitative relationships; final cohort is construction-screened, not a random or representative clinical sample. Retained and replacement subsets must be reported separately.'}


def recover_cached_rule(sources):
    result=read(WORK/'cache_recovery_complete.json');qualified=[]
    for job in result['jobs']:
        if not job['complete_json_candidate']:continue
        sid=job['source_id'];folder=WORK/'cached_rule_replay'/'sources'/sid
        for path in sorted((OLD/'sources'/sid).glob('candidate_*/candidate.private.json')):
            prior.c.frozen(folder/path.parent.name/'candidate.private.json',path.read_bytes())
        response=read(WORK/'recovered_jobs'/job['job_id']/'response.private.json')
        proposed=json.loads(response['choices'][0]['message']['content'])
        trial=folder/'candidate_recovered_cache'
        changes=[]
        if job['stage']=='rules':
            parent_trial=(ROOT/job['parent_dispatch']).parent.parent
            facts=read(parent_trial/'checked_facts.private.json')
            known={f['key'] for f in facts}
            for family in proposed.get('families',[]):
                used={a[6:] for node in family['nodes'] for a in node['args'] if a.startswith('const:')}
                if used<=known and not used<=set(family['constant_ids']):
                    family['constant_ids']=sorted(used);changes.append('reconciled_redundant_declared_constant_ids_only')
            proposed=prior.native_candidate(proposed,facts,sources[sid])
        prior.c.frozen(trial/'candidate.private.json',proposed)
        prior.c.frozen(trial/'cache_mapping.json',{'job_id':job['job_id'],'changes':changes,'new_model_calls':0})
        with patch.object(prior,'WORK',WORK/'cached_rule_replay'):
            items,errors=prior.qualify(sources[sid])
        prior.c.frozen(folder/'cache_validation.json',{'machine_valid':items is not None,'errors':errors,'new_model_calls':0})
        if items:
            prior.c.frozen(folder/'accepted_spec.private.json',items);qualified.append(sid)
    return qualified


def main():
    prior.c.verify_tree(prior.c.SOURCES);prior.c.verify_tree(prior.c.POOL)
    p=policy();prior.c.frozen(WORK/'replacement_selection_policy.json',p)
    for name in p['code_sha256']:prior.c.frozen(WORK/'selection_code_snapshot'/name,(ROOT/name).read_bytes())
    original={s['source_id']:s for s in prior.c.load_reserved()}
    recovery=read(WORK/'cache_recovery_plan.json')
    selected={r['source_id']:original[r['source_id']] for r in recovery['adopted']}
    specs={sid:read(WORK/'adopted63'/sid/'accepted_spec.private.json') for sid in selected}
    kinds={sid:'retained_qualified63' for sid in selected}
    cache_qualified=recover_cached_rule(original)
    for sid in cache_qualified:
        selected[sid]=original[sid];specs[sid]=read(WORK/'cached_rule_replay/sources'/sid/'accepted_spec.private.json')
        kinds[sid]='recovered_existing_author_response_no_regeneration'
    exposed=prior.c.load_exposed_sources();excluded={s['source_cluster'] for s in exposed+list(original.values())}
    pool=[json.loads(x) for x in (prior.c.POOL/'sources.private.jsonl').read_text(encoding='utf-8').splitlines()]
    eligible=[s for s in pool if s['split']=='locked_test' and s['source_cluster'] not in excluded]
    groups=defaultdict(list);profiles=[]
    for source in eligible:
        parsed=regimen15.parse(source)
        row={k:source[k] for k in ('source_id','source_cluster','modality','length_band','characters')}
        row['information_eligible']=parsed is not None
        if parsed:
            row.update(relation_kind=parsed['kind'],record_kind=parsed['record_kind'],
                source_block_sha256=parsed['source_block_sha256'],reference_span=parsed['character_span_private'])
            groups[(source['modality'],source['length_band'])].append(source)
        profiles.append(row)
    prior.c.frozen(WORK/'candidate_source_profiles.json',profiles)
    queues={str(k):[s['source_id'] for s in sorted(vals,key=lambda s:prior.c.hash_value(['confirmation15-blind-source-order',s['source_id']]))]
        for k,vals in groups.items()}
    prior.c.frozen(WORK/'candidate_order.json',queues)
    counts=Counter((s['modality'],s['length_band']) for s in selected.values());ledger=[];extra=[]
    # No model score or task prediction is read by this process.
    for stratum,target in sorted(TARGETS.items()):
        for source in sorted(groups[stratum],key=lambda s:prior.c.hash_value(['confirmation15-blind-source-order',s['source_id']])):
            if counts[stratum]>=target:break
            sid=source['source_id'];items,report=regimen15.construct(source)
            okay,errors=prior.c.check_items(items,source) if items else (False,report.get('errors',[report.get('reason')]))
            row={'source_id':sid,'stratum':list(stratum),'machine_valid':okay,'errors':errors,'new_model_calls':0}
            prior.c.frozen(WORK/'replacement_candidates'/sid/'result.json',row);ledger.append(row)
            if not okay:continue
            prior.c.frozen(WORK/'replacement_candidates'/sid/'accepted_spec.private.json',items)
            prior.c.frozen(WORK/'replacement_candidates'/sid/'source_reference.private.json',report['parsed'])
            selected[sid]=source;specs[sid]=items;kinds[sid]='blind_information_screened_exact_regimen_replacement'
            counts[stratum]+=1;extra.append(sid)
    profile={'eligible_unexposed_locked_sources':len(eligible),'information_eligible_by_stratum':{str(k):len(vals) for k,vals in groups.items()},
        'selected_by_stratum':{str(k):n for k,n in counts.items()},'retained':63,'cached_source_qualified':len(cache_qualified),
        'new_replacement_sources':len(extra),'selected_total':len(selected),'target':120,'new_model_calls':0,
        'task_seal_complete':False,'new_source_ids':extra,'construction_attrition':ledger}
    prior.c.frozen(WORK/'source_screen_result.json',profile)
    if len(selected)!=120 or any(counts[k]!=n for k,n in TARGETS.items()):
        dump(OUT/'RUN_STATUS.json',{'stage':'information_screen_short_of_stratified_target',
            'selected_sources':len(selected),'retained_sources':63,'tasks_frozen':False,'solver_calls':0,'author_generation_calls':0})
        print(json.dumps({k:val for k,val in profile.items() if k not in ('new_source_ids','construction_attrition')}));return
    # Candidate duplicate flags are source-only. No solver run may begin before
    # this check. A flag blocks this proposal rather than silently changing it.
    audit=prior.c.near_duplicates(list(selected.values()),exposed+list(original.values()))
    prior.c.frozen(WORK/'selected_near_duplicate_audit.json',audit)
    if audit['exact_normalized_duplicate_pairs'] or audit['near_duplicate_flags']:
        raise ValueError('Proposed source-only cohort has duplicate flags; requires recorded selection amendment, no silent change')
    final=sorted(selected.values(),key=lambda s:s['source_id'])
    catalog=[{**{k:s[k] for k in ('source_id','source_cluster','modality','length_band','split','characters')},
        'construction_subset':kinds[s['source_id']]} for s in final]
    prior.c.frozen(SOURCES/'sources.private.jsonl',prior.c.jsonl(final))
    prior.c.frozen(SOURCES/'source_catalog.json',catalog)
    manifest={'kind':'confirmation15-replacement-source-cohort-not-yet-task-seal','source_clusters':120,
        'retained63':63,'cached_original_sources_additionally_qualified':cache_qualified,'new_replacements':extra,
        'original120_ids_not_retained':sorted(set(original)-set(selected)),
        'source_catalog':catalog,'original_source_records':sum(len(s['documents']) for s in final),
        'modalities':dict(Counter(s['modality'] for s in final)),
        'selection_policy_sha256':digest(WORK/'replacement_selection_policy.json'),
        'new_model_calls':0,'task_seal_complete':False,'clinical_expert_gold':False,
        'source_holdout_not_rule_family_holdout':True,'not_the_original120_all_qualified':True,
        'no_solver_score_used':True,'public_release_authorized':False}
    prior.c.frozen(SOURCES/'manifest.json',manifest)
    prior.c.frozen(SOURCES/'near_duplicate_audit.json',audit)
    prior.c.frozen(SOURCES/'build_integrity.json',{p.name:digest(p) for p in sorted(SOURCES.iterdir()) if p.is_file() and p.name!='build_integrity.json'})
    for source in final:
        sid=source['source_id'];prior.c.frozen(WORK/'sources'/sid/'accepted_spec.private.json',specs[sid])
        prior.c.frozen(WORK/'sources'/sid/'result.json',{'source_id':sid,'machine_valid':True,
            'construction_kind':kinds[sid],'family_count':len(specs[sid]),'solver_calls':0,'clinical_expert_gold':False})
    dump(OUT/'RUN_STATUS.json',{'stage':'replacement120_sources_selected_task_seal_pending','selected_sources':120,
        'retained_sources':63,'cache_qualified_sources':len(cache_qualified),'new_replacement_sources':len(extra),
        'replacement_source_inputs_frozen':True,'tasks_frozen':False,'solver_calls':0,'author_generation_calls':0})
    print(json.dumps({k:val for k,val in profile.items() if k not in ('new_source_ids','construction_attrition')}),flush=True)


if __name__=='__main__':main()
