"""Freeze a prospective finite study on the authorized replacement cohort.

Only the explicit actor/code bundle may be deployed; standards remain local.
No model scores are read here. No inference or network operation is performed.
"""
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import confirmation13 as c
from run_generation import ROOT,read,digest,dump
from pilot_runtime import fingerprint
from seal_confirmation15 import DATASET
from prepare_replacement15 import SOURCES
from recover_confirmation15_cache import OUT,WORK
import study15 as s

LOCAL_CODE=('prepare_study15.py','score_study15_local.py','test_study15.py','test_confirmation14_workflow.py')


def subset(catalog,n,tag):
    selected=[]
    for modality in ('dialogue','drug'):
        for band in range(3):
            per=n//6
            rows=sorted([r for r in catalog if r['modality']==modality and r['length_band']==band],
                key=lambda r:fingerprint([tag,r['source_cluster']]))
            if len(rows)<per:raise ValueError('Fixed stratum unavailable')
            selected.extend(r['source_id'] for r in rows[:per])
    if len(set(selected))!=n:raise ValueError('Duplicate source subset')
    return selected


def main():
    c.verify_tree(DATASET);c.verify_tree(SOURCES)
    for rel,sha in read(WORK/'preserved_old_hashes.json').items():
        if digest(ROOT.parent/rel)!=sha:raise ValueError('Historical artifact changed')
    verified=read(DATASET/'verification.json')
    if not verified['all_selected120_qualified_unchanged_gates'] or not verified['actor_target_separation']:raise ValueError('Dataset gate failed')
    if read(OUT/'delivery_manifest.json')['dataset_integrity_sha256']!=digest(DATASET/'build_integrity.json'):raise ValueError('Delivery differs')
    audit=read(OUT/'legacy_shape_audit_manifest.json')
    if audit['dataset_delivery_manifest_sha256']!=digest(OUT/'delivery_manifest.json'):raise ValueError('Auxiliary shape audit differs')
    for n in ('actors.private.jsonl','static_actors.private.jsonl','actor_manifest.json'):
        c.frozen(s.INPUTS/n,(DATASET/n).read_bytes())
    catalog=read(SOURCES/'manifest.json')['source_catalog']
    actors=[json.loads(x) for x in (DATASET/'actors.private.jsonl').read_text(encoding='utf-8').splitlines()]
    statics=[json.loads(x) for x in (DATASET/'static_actors.private.jsonl').read_text(encoding='utf-8').splitlines()]
    jobs=s.static_jobs(statics)
    if len(jobs)!=840 or sum(len(j['members']) for j in jobs)!=1080:raise ValueError('Unexpected public-payload dedup')
    p=copy.deepcopy(read(ROOT.parent/'output/medical14_confirmation_20261022/EXPERIMENT_PLAN.json'))
    p.update(version='confirmation15-replacement-finite-design-v1',
        authorization='User explicitly selected route2: preserve63, recover caches, blind information-screened replacements; complete finite experiments.',
        source_ids=[r['source_id'] for r in catalog],dataset_integrity_sha256=digest(DATASET/'build_integrity.json'),
        source_integrity_sha256=digest(SOURCES/'build_integrity.json'),construction_protocol_sha256=digest(DATASET/'protocol.json'),
        actor_manifest_sha256=digest(s.INPUTS/'actor_manifest.json'),solver_runs_started=False,
        source_catalog=catalog,not_original120_all_qualified=True,
        construction_subsets=dict(Counter(r['construction_subset'] for r in catalog)),
        runtime_code_sha256={n:digest(ROOT/n) for n in s.CODE+('validate_generated.py',)},
        local_scoring_code_sha256={n:digest(ROOT/n) for n in LOCAL_CODE+('score_workflow11_local.py',)},
        parent_service_sha256={n:digest(ROOT/n) for n in ('service_baseline10.py','baseline10_common.py','service_expansion08.py','baseline10_protocol.json')},
        model_context_limits={'Qwen3.5-9B':65536,'Qwen3.8-27B':65536,'GLM-4-9B-0414':32768},
        code_scope='Main/static/robustness code frozen now. Web layout/action transformer is a separate predeclared runtime freeze before its first solve; no result-dependent design changes.',
        start_gate=['selected replacement120 qualified','unchanged qualification rules','source/target separation','same-stem triplets and six direction quotas','independent exact evaluator and mechanical replay','task/source manifests','prior author resources cleaned'])
    p['analysis']['primary']='coverage-adjusted source-macro content; report retained63 and replacement57 separately; source is not a patient'
    p['analysis']['paired_family']='36 main primary contrasts:3models x4events x(evidence_constraint-labels_only,evidence_constraint-fresh,labels_only-fresh); Holm across36. Factorial2x2 interactions separately12 predeclared contrasts.'
    p['resource_scope']['admission']='Three fresh whole-idle samples plus atomic service admission; no foreign tasks stopped; stricter than authorized low-load sharing.'
    p['robustness']['source_ids']=subset(catalog,30,'confirmation15-robustness')
    p['web_service_controls']['source_ids']=subset(catalog,48,'confirmation15-web')
    p['medical_review_pending']['source_ids']=[]
    for mod,per in (('dialogue',12),('drug',8)):
        for band in range(3):
            rr=sorted([r for r in catalog if r['modality']==mod and r['length_band']==band],key=lambda r:fingerprint(['confirmation15-blind-review',r['source_cluster']]))
            p['medical_review_pending']['source_ids'] += [r['source_id'] for r in rr[:per]]
    p['static_shortcut_controls']['physical_unique_per_model']=840
    p['limitations'] += ['Replacement57 constructibility curation favors explicit quantitative regimen relations; not a random clinical sample or broad diagnosis benchmark.',
        'Retained63 reuse Qwen-authored proposals; replacement57 are exact-grammar compiled, not model authored.',
        'Auxiliary legacy shape audit excludes56 map-less proofs only from that comparison, without excluding old cases or scores.',
        'Document-order replicate2 may coincide with baseline or reverse for a source; disclose actual permutations, do not claim seed-only variability.']
    c.frozen(s.STUDY,p);c.frozen(OUT/'EXPERIMENT_PLAN.json',p)
    for n in p['runtime_code_sha256']|p['local_scoring_code_sha256']:
        c.frozen(WORK/'solver_code_snapshot'/n,(ROOT/n).read_bytes())
    c.frozen(OUT/'EXPERIMENT_PLAN.md','# 新120来源的封存实验计划\n\n'
        '原样保留63，盲筛候补57；不是原120全部合格。五方法四事件、静态资料移除、顺序扰动和双布局权限/版本控制，共16020逻辑槽，不是请求数。\n\n'
        'Qwen9B、27B和GLM9B使用同一JSON动作协议、温度0、最多40动作，目标和独立评分仅留本机。最多两项服务并行。每个分支复用模型真实初始提交，错误旧标签不替换，初始未提交则保留依赖阻塞。\n\n'
        '主要报告覆盖调整后的来源宏平均，以及内容、应改项、稳定项、当前证据、实际保存和读回；保留63与候补57分开。机械回放不是模型结果。来源隔离银标不是临床金标或规则族隔离。全部实验结束并核验后统一生成HTML。\n')
    dump(OUT/'RUN_STATUS.json',{'stage':'solver_protocol_frozen_not_launched','sources':120,'retained_sources':63,
        'new_replacement_sources':57,'tasks_frozen':True,'study_protocol_sha256':digest(s.STUDY),
        'planned_logical_slots_total':16020,'solver_calls':0,'local_scoring_finished':False,'html_final_updated':False,
        'web_controls_pending':True,'clinical_expert_gold':False})
    print(json.dumps({'tasks_frozen':True,'solver_protocol_frozen':True,'sources':120,'static_unique_per_model':len(jobs),'logical_slots':16020}))


if __name__=='__main__':main()
