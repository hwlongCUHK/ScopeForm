"""Append-only seal gate for the same reserved120 sources. Never calls a model."""
import argparse
from collections import Counter
import json

import confirmation14 as v
import verify_confirmation13_local as previous


def rule_shape(expression, fields, constants):
    def shape(node):
        if type(node) is bool:return ['bool']
        if 'field' in node:return ['field',fields[node['field']]['type']]
        if 'constant' in node:return ['source_constant',type(constants[node['constant']]['value']).__name__]
        return [node['op'],[shape(a) for a in node['args']]]
    return v.c.hash_value(shape(expression))


def seal():
    v.configure();v.c.verify_plan()
    report=previous.verify(oracle=True)
    v.c.frozen(v.OUT/'local_verification.json',report)
    cleanup=v.read(v.WORK/'resource_cleanup_verified.json')
    if not cleanup['verified_clean']:raise ValueError('Do not seal before owned processes and ports are clean')
    totals=v.summary()
    if totals['accounted']!=120 or totals['qualified']!=120:
        closeout={**totals,'tasks_frozen':False,'solver_calls':0,'verification':report,
            'resources_cleaned':True,'sources_unchanged':True,'next_solver_experiments_blocked':True}
        v.c.frozen(v.OUT/'construction_closeout.json',closeout)
        v.c.status(stage='construction_gate_not_passed_no_solver_experiment',tasks_frozen=False,**totals)
        return closeout
    if v.DATASET.exists():
        v.c.verify_tree(v.DATASET)
        return v.read(v.DATASET/'construction_audit.json')
    sources=v.c.load_reserved();directions=v.read(v.WORK/'direction_assignment.LOCAL_ONLY.json')
    actors=[];targets=[];statics=[];transitions=Counter();counts=Counter();kinds=Counter();shapes=Counter()
    old_shapes=set();old=v.ROOT/'datasets/medical09_dev120_silver_v1'
    for case in v.read(old/'manifest.json')['cases']:
        if case['configuration_order_private']==0:
            for proof in v.read(old/'cases'/case['task_id']/'proofs.private.json'):
                old_shapes.add(rule_shape(proof['expression'],proof['fields'],proof['constants']))
    for source in sources:
        sid=source['source_id'];folder=v.WORK/'sources'/sid
        row=v.read(folder/'result.json');items=v.read(folder/'accepted_spec.private.json')
        valid,errors=v.c.check_items(items,source)
        if not valid:raise ValueError('Qualified rule mutated: '+sid)
        bundle=v.compile_source(source,items,directions[sid]);v.c.verify_compiled(source,bundle)
        v.c.frozen(v.DATASET/'local_private'/(sid+'.private.json'),bundle)
        initial=bundle['initial']
        actors.append({'source_id':sid,'initial':initial['controller'],
            'events':[{'event_id':e['event_id'],'controller':e['controller']} for e in bundle['events']]})
        statics.extend({'source_id':sid,'packet_id':p['packet_id'],'controller':p['controller']} for p in bundle['packets'])
        targets.append({'source_id':sid,'source_cluster':source['source_cluster'],'adaptive_calibration_reused':False,
            'source_split':'confirmation_source_holdout','modality':source['modality'],
            'environment':initial['controller']['environment'],'construction_subset':row['construction_kind'],
            'initial':initial['items'],'events':[{'event_id':e['event_id'],'items':e['items']} for e in bundle['events']],
            'packets':[{'packet_id':p['packet_id'],'items':p['items']} for p in bundle['packets']]})
        base={g['item_id']:g['label'] for g in initial['items']}
        for event in bundle['events']:
            for g in event['items']:
                if base[g['item_id']]!=g['label']:transitions[(event['event_id'],base[g['item_id']],g['label'])]+=1
        counts['unique_authored_families']+=len(items)
        counts['claim_instances_per_static_configuration']+=bundle['claim_instances_per_packet']
        counts['primary_static_judgments']+=sum(len(p['items']) for p in bundle['packets'])
        counts['original_source_records']+=len(source['documents'])
        kinds[row['construction_kind']]+=1
        for item in items:shapes[rule_shape(item['expression'],item['field_map'],item['constant_map'])]+=1
    S,R,U=v.c.rules.S,v.c.rules.R,v.c.rules.U
    expected={('single_definite',S,R):60,('single_definite',R,S):60,
        ('single_uncertainty',S,U):30,('single_uncertainty',R,U):30,
        ('single_uncertainty',U,S):30,('single_uncertainty',U,R):30,
        ('key_mixed',S,R):60,('key_mixed',R,S):60,
        ('key_mixed',S,U):30,('key_mixed',R,U):30,('key_mixed',U,S):30,('key_mixed',U,R):30}
    if transitions!=Counter(expected):raise ValueError('Full prescribed transition quotas failed')
    v.c.frozen(v.DATASET/'actors.private.jsonl',v.c.jsonl(actors))
    v.c.frozen(v.DATASET/'static_actors.private.jsonl',v.c.jsonl(statics))
    v.c.frozen(v.DATASET/'evaluator_targets.LOCAL_ONLY.private.jsonl',v.c.jsonl(targets))
    v.c.frozen(v.DATASET/'protocol.json',v.read(v.WORK/'protocol.json'))
    v.c.frozen(v.DATASET/'source_snapshot_manifest.json',v.read(v.c.SOURCES/'manifest.json'))
    actor_manifest={'kind':'confirmation14-gold-free-actor-bundle','sources':120,'events':480,'static_packets':360,
        'files':{n:v.digest(v.DATASET/n) for n in ('actors.private.jsonl','static_actors.private.jsonl')},
        'protocol_sha256':v.digest(v.DATASET/'protocol.json'),'private_source_text':True,
        'no_evaluator_target_in_actor_manifest':True,'public_release_authorized':False}
    v.c.frozen(v.DATASET/'actor_manifest.json',actor_manifest)
    audit={'kind':'confirmation14-same-reserved120-mechanical-silver-seal','task_sources':120,'source_clusters':120,
        'primary_static_packets':360,'workflow_initial_states':120,'workflow_event_states':480,**counts,
        'construction_subsets':dict(kinds),'transition_counts':[{'event':e,'from':a,'to':b,'count':n} for (e,a,b),n in sorted(transitions.items())],
        'distinct_anonymous_rule_shapes':len(shapes),'rule_shapes_also_seen_in_development':len(set(shapes)&old_shapes),
        'authored_families_with_seen_rule_shape':sum(n for s,n in shapes.items() if s in old_shapes),
        'source_holdout_not_rule_family_holdout':True,'clinical_expert_gold':False,'solver_calls':0,'tasks_frozen':True,
        'protection_probe_is_reused_family_not_independent_case':True,'not_independent_patient_or_hospital_count':True}
    v.c.frozen(v.DATASET/'construction_audit.json',audit)
    v.c.frozen(v.DATASET/'rule_shape_overlap_audit.json',{'anonymous_shapes':dict(shapes),
        'previous_development_shapes':sorted(old_shapes),'shared_shapes':sorted(set(shapes)&old_shapes),
        'not_semantic_rule_novelty_or_training_contamination_certificate':True})
    v.c.frozen(v.DATASET/'verification.json',{'source_roundtrips':True,'same_stem_balanced_triplets':True,
        'independent_evaluator_agrees':True,'all_six_prescribed_transitions':True,'all120_reserved_no_replacement':True,
        'old_artifacts_unchanged':True,'known_answer_mechanical_phases':report['mechanical_oracle_phases_not_model_results'],
        'resources_cleaned':True,'no_solver_calls':True,'clinical_validation':False,'rule_family_disjoint':False})
    v.c.frozen(v.DATASET/'build_integrity.json',{p.relative_to(v.DATASET).as_posix():v.digest(p)
        for p in sorted(v.DATASET.rglob('*')) if p.is_file()})
    v.c.verify_tree(v.DATASET)
    v.c.frozen(v.OUT/'delivery_manifest.json',{'dataset_path':str(v.DATASET),
        'dataset_integrity_sha256':v.digest(v.DATASET/'build_integrity.json'),
        'source_integrity_sha256':v.digest(v.c.SOURCES/'build_integrity.json'),
        'protocol_sha256':v.digest(v.WORK/'protocol.json'),
        'local_verification_sha256':v.digest(v.OUT/'local_verification.json'),
        'cleanup_sha256':v.digest(v.WORK/'resource_cleanup_verified.json'),'tasks_frozen':True,'solver_calls':0})
    v.c.status(stage='complete_task_seal_ready_for_predeclared_experiments',tasks_frozen=True,**totals)
    v.c.frozen(v.OUT/'DATASET_SEAL.md','# 同一预留120来源：修订构建封存\n\n'
        '原清单、原文、旧版本及旧结果保持不变。全部120来源通过原有程序门槛，含360份同问法三态资料包、120初始状态和480事件定义。\n\n'
        '仅来源隔离；部分规则形状见于开发集。这是模拟状态银标，不是独立患者数、临床金标或规则族隔离。构建模型与部分求解模型同系列，需披露作者偏差。\n\n'
        '完整题目封存与已知答案机械回放不等于模型评测完成。答题实验须另封存方案，从模型真实初始提交分叉；不得上传本机标准标签或私有规则。\n')
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seal',action='store_true');args=parser.parse_args()
    v.configure()
    print(json.dumps(seal() if args.seal else previous.verify(oracle=True),ensure_ascii=True))
