"""Seal the explicitly authorized replacement cohort, not the old failed120.

All clinical targets/proofs stay local. Mechanical known-answer runs are
software tests only and can never become model initial checkpoints.
"""
from collections import Counter
import copy
import json
from pathlib import Path

import confirmation13 as c
from run_generation import ROOT,read,digest,dump
from recover_confirmation15_cache import WORK,OUT,OLD
from prepare_replacement15 import SOURCES
from verify_confirmation14_local import rule_shape

DATASET=ROOT/'datasets/medical15_confirmation120_silver_v1'
BASE_COMPILE=c.compile_source
CODE=('seal_confirmation15.py','test_confirmation15.py','regimen15.py','test_regimen15.py',
    'recover_confirmation15_cache.py','prepare_replacement15.py','extend_replacement15.py')+c.CODE


def compile_source(source,items,direction):
    bundle=BASE_COMPILE(source,items,direction)
    for packet in bundle['packets']+[bundle['initial']]+bundle['events']:
        packet['controller']['task_id']=source['source_id']+'__confirmation15_clinical_state'
        packet['controller']['task_version']='confirmation15'
        packet['packet_id']=packet['packet_id'].replace('__c13_','__c15_').replace('__c14_','__c15_')
    return bundle


def oracle_replay(bundle):
    from confirmation14_workflow import World,branch
    from validate_workflow11_offline import complete
    from score_workflow11_local import score_row
    from workflow11_runtime import export_observations
    from pilot_runtime import fingerprint
    initial=bundle['initial'];base=complete(World(initial['controller']),initial['items'])
    base_row={'submitted':True,'event_id':'initial',**export_observations(base)}
    if not score_row(base_row,initial['items'],initial['controller'],initial['items'],base_row)['strict']:
        raise ValueError('Known-answer initial replay failed')
    n=1
    for event in bundle['events']:
        for method in ('labels_only','index_only','labels_guard','evidence_constraint','fresh'):
            world=branch(copy.deepcopy(base),event['controller'],event['event_id'],method)
            if fingerprint(world.state)!=fingerprint(base.state):raise ValueError('Mechanical test did not preserve base')
            complete(world,event['items'])
            row={'submitted':True,'event_id':event['event_id'],**export_observations(world)}
            if not score_row(row,event['items'],event['controller'],initial['items'],base_row)['strict']:
                raise ValueError('Known-answer branch replay failed')
            n+=1
    return n


def seal():
    c.verify_tree(SOURCES)
    source_manifest=read(SOURCES/'manifest.json')
    for name,sha in read(OLD/'protocol.json')['code_sha256'].items():
        if digest(ROOT/name)!=sha:raise ValueError('Parent construction code changed')
    for rel,sha in read(OLD/'preserved_old_hashes.json').items():
        if digest(ROOT.parent/rel)!=sha:raise ValueError('Old artifact changed')
    sources=[json.loads(x) for x in (SOURCES/'sources.private.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(sources)!=120 or len({s['source_cluster'] for s in sources})!=120:raise ValueError('Need full selected120 distinct clusters')
    protocol={'version':'confirmation15-source-information-replacement-silver-v1','sources':120,
        'retained63_unchanged':True,'new_replacement_sources':source_manifest['new_replacement_sources'],
        'not_the_original120_all_qualified':True,'source_integrity_sha256':digest(SOURCES/'build_integrity.json'),
        'parent_construction_protocol_sha256':digest(OLD/'protocol.json'),
        'selection_policy_sha256':digest(WORK/'replacement_selection_policy.json'),
        'source_supply_amendment_sha256':digest(WORK/'source_supply_amendment01.json'),
        'code_sha256':{n:digest(ROOT/n) for n in CODE},
        'qualification_unchanged':read(OLD/'protocol.json')['primary_family_requirement'],
        'label_semantics_sha256':digest(ROOT/'scale09_rules_r4.py'),
        'direction_assignment':'Same original source-hash/stratum algorithm on the NEW full cohort, before solver access. This creates new states, not edits to old files.',
        'source_selection_bias':'Replacement sources need explicit quantitative relationships. Report retained63 and replacement57 separately; not a random clinical sample.',
        'rule_family_holdout':False,'clinical_expert_gold':False,'new_author_generations':0,'solver_calls':0}
    c.frozen(WORK/'protocol.json',protocol)
    for name in CODE:c.frozen(WORK/'code_snapshot'/name,(ROOT/name).read_bytes())
    directions=c.assignment(source_manifest['source_catalog']);c.frozen(WORK/'direction_assignment.LOCAL_ONLY.json',directions)
    preserved=read(OLD/'preserved_old_hashes.json')
    for name in ('protocol.json','replacement_selection_policy.json','source_supply_amendment01.json'):
        if name!='protocol.json':preserved[(WORK/name).relative_to(ROOT.parent).as_posix()]=digest(WORK/name)
    c.frozen(WORK/'preserved_old_hashes.json',preserved)
    if (DATASET/'build_integrity.json').exists():c.verify_tree(DATASET);return read(DATASET/'construction_audit.json')
    actors=[];statics=[];targets=[];transitions=Counter();totals=Counter();shapes=Counter();subsets=Counter();relation_kinds=Counter()
    old_shapes=set();old_dataset=ROOT/'datasets/medical09_dev120_silver_v1'
    for case in read(old_dataset/'manifest.json')['cases']:
        if case['configuration_order_private']==0:
            for proof in read(old_dataset/'cases'/case['task_id']/'proofs.private.json'):
                old_shapes.add(rule_shape(proof['expression'],proof['fields'],proof['constants']))
    for source in sources:
        sid=source['source_id'];folder=WORK/'sources'/sid;row=read(folder/'result.json');items=read(folder/'accepted_spec.private.json')
        valid,errors=c.check_items(items,source)
        if not valid:raise ValueError('Selected rule failed unchanged qualification: '+sid)
        if row['construction_kind']=='retained_qualified63' and digest(folder/'accepted_spec.private.json')!=digest(OLD/'sources'/sid/'accepted_spec.private.json'):
            raise ValueError('Mandatory retained63 changed')
        blocks={b['anchor_id']:b['text'] for d in source['documents'] for b in d['blocks']}
        for item in items:
            for reference in item['references']:
                if reference['quote'] not in blocks[reference['anchor_id']]:raise ValueError('Quote roundtrip failed')
                totals['original_quote_roundtrips']+=1
            for state in item['state_values']:
                if c.independent_label(item,state)!=c.rules.outcome(item['expression'],state,item['constant_map']):raise ValueError('Independent evaluator disagreement')
                totals['independent_rule_state_checks']+=1
            shapes[rule_shape(item['expression'],item['field_map'],item['constant_map'])]+=1
        bundle=compile_source(source,items,directions[sid]);c.verify_compiled(source,bundle)
        totals['mechanical_oracle_phases_not_model_results']+=oracle_replay(bundle)
        c.frozen(DATASET/'local_private'/(sid+'.private.json'),bundle)
        initial=bundle['initial'];actors.append({'source_id':sid,'initial':initial['controller'],
            'events':[{'event_id':e['event_id'],'controller':e['controller']} for e in bundle['events']]})
        statics.extend({'source_id':sid,'packet_id':p['packet_id'],'controller':p['controller']} for p in bundle['packets'])
        targets.append({'source_id':sid,'source_cluster':source['source_cluster'],'modality':source['modality'],
            'environment':initial['controller']['environment'],'construction_subset':row['construction_kind'],
            'source_split':'confirmation_source_holdout_replacement','initial':initial['items'],
            'events':[{'event_id':e['event_id'],'items':e['items']} for e in bundle['events']],
            'packets':[{'packet_id':p['packet_id'],'items':p['items']} for p in bundle['packets']]})
        base={g['item_id']:g['label'] for g in initial['items']}
        for event in bundle['events']:
            for g in event['items']:
                if base[g['item_id']]!=g['label']:transitions[(event['event_id'],base[g['item_id']],g['label'])]+=1
        totals['unique_authored_families']+=len(items);totals['claim_instances_per_configuration']+=len(initial['items'])
        totals['static_judgments']+=sum(len(p['items']) for p in bundle['packets'])
        totals['original_source_records']+=len(source['documents']);subsets[row['construction_kind']]+=1
        if row['construction_kind']!='retained_qualified63':relation_kinds[items[0]['parsed_regimen_private']['kind']]+=1
    S,R,U=c.rules.S,c.rules.R,c.rules.U
    for event in ('single_definite','key_mixed'):
        if transitions[event,S,R]!=60 or transitions[event,R,S]!=60:raise ValueError('Definite direction quota mismatch')
    for event in ('single_uncertainty','key_mixed'):
        if any(transitions[event,a,b]!=30 for a,b in ((S,U),(R,U),(U,S),(U,R))):raise ValueError('Uncertainty direction quota mismatch')
    if totals['mechanical_oracle_phases_not_model_results']!=2520:raise ValueError('Full five-method software replay incomplete')
    c.frozen(DATASET/'actors.private.jsonl',c.jsonl(actors));c.frozen(DATASET/'static_actors.private.jsonl',c.jsonl(statics))
    c.frozen(DATASET/'evaluator_targets.LOCAL_ONLY.private.jsonl',c.jsonl(targets))
    c.frozen(DATASET/'protocol.json',protocol)
    c.frozen(DATASET/'actor_manifest.json',{'kind':'confirmation15-gold-free-actor-bundle','sources':120,'events':480,'static_packets':360,
        'files':{n:digest(DATASET/n) for n in ('actors.private.jsonl','static_actors.private.jsonl')},
        'construction_protocol_sha256':digest(WORK/'protocol.json'),'private_source_text':True,'public_release_authorized':False})
    audit={'task_sources':120,'retained63':63,'new_replacements':57,'static_packets':360,'initial_states':120,'event_states':480,
        **totals,'construction_subsets':dict(subsets),'replacement_relation_kinds':dict(relation_kinds),
        'directions':[{'event':e,'from':a,'to':b,'count':n} for (e,a,b),n in sorted(transitions.items())],
        'anonymous_rule_shapes':len(shapes),'shapes_also_seen_in_development':len(set(shapes)&old_shapes),
        'source_holdout_not_rule_family_holdout':True,'not_original120_all_passed':True,'clinical_expert_gold':False,
        'new_author_generations':0,'solver_calls':0,'tasks_frozen':True,'protection_probe_is_reused_rule_not_independent_case':True}
    c.frozen(DATASET/'construction_audit.json',audit)
    c.frozen(DATASET/'verification.json',{'all_selected120_qualified_unchanged_gates':True,'retained63_specs_byte_identical':True,
        'source_roundtrips':True,'same_stem_triplets':True,'prescribed_six_direction_quotas':True,
        'independent_exact_evaluator':True,'five_method_mechanical_replays':2520,'old_files_unchanged':True,
        'actor_target_separation':True,'solver_calls':0,'new_construction_gpu_services':0,'clinical_validation':False})
    c.frozen(DATASET/'build_integrity.json',{path.relative_to(DATASET).as_posix():digest(path) for path in sorted(DATASET.rglob('*')) if path.is_file() and path.name!='build_integrity.json'})
    c.verify_tree(DATASET)
    c.frozen(OUT/'delivery_manifest.json',{'dataset_path':str(DATASET),'dataset_integrity_sha256':digest(DATASET/'build_integrity.json'),
        'source_integrity_sha256':digest(SOURCES/'build_integrity.json'),'protocol_sha256':digest(WORK/'protocol.json'),
        'tasks_frozen':True,'solver_calls':0,'new_author_generations':0,'new_model_services_started':0})
    dump(OUT/'RUN_STATUS.json',{'stage':'replacement120_tasks_frozen_experiments_pending','sources':120,'retained_sources':63,
        'new_replacement_sources':57,'tasks_frozen':True,'solver_calls':0,'author_generation_calls':0,'clinical_expert_gold':False})
    c.frozen(OUT/'DATASET_SEAL.md','# 候补补齐后的120来源：任务封存\n\n'
        '保留原样合格63来源，按预先定义的原文信息规则补入57来源。不是原120来源全通过。旧清单、失败记录、旧分数与旧HTML不变。\n\n'
        '新120来源均通过原资格门槛、同问法三态、六向变化、来源回查与独立规则计算；五种方法的2520个已知答案软件阶段回放通过。这些不是模型答题结果。\n\n'
        '候补需要明确数量关系，存在构建筛选偏差；必须分开报告原保留部分与候补部分。来源隔离不等于规则族隔离、独立患者或临床金标。\n\n'
        '本批没有新增作者或求解生成，没有启动GPU模型服务。后续实验必须另封存求解运行与评分代码，从模型真实提交的初始检查点分叉，不得使用机械回放起点。\n')
    return audit


if __name__=='__main__':print(json.dumps(seal(),ensure_ascii=True))
