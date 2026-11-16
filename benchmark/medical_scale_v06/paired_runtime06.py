"""Gold-free, same-question clinical chartlet sandbox for paired06."""
import copy
from pilot_runtime import World as BaseWorld, fingerprint, ledger as base_ledger
from medical_runtime04 import POLICY as OLD_POLICY

POLICY = OLD_POLICY + '''
本轮为明确标记的独立模拟临床交接场景。每个字母场景是不同的独立情况，不能相互补参数。
原始资料保持原样，供核对引用规则/历史参照；本轮的患者状态只在相应模拟场景中，不得用原始患者旧值替代。
请核对陈述本身的有界含义：满足一项条件不等于整体安全；违反所核对的必要条件才构成反驳。
缺实际检查/暴露/关键参数或缺相应人群规则，应为not_determined，不要因“不能证明”就反驳真实病因命题。
空值表示未知，绝不默认正常、阴性或从未发生。临床缺口须保留，不通过外部常识补齐。
直接完成七项核对并提交；证据引用实际观察的原文和模拟状态锚点，不需要叙述思考过程。
'''

PRIOR_POLICY = '''这是题干标签先验探测，不是临床诊疗建议，也不是对当前隐藏原文的可回答性评分。
来源原文和模拟状态均隐藏。请仅根据七项题干，预测看过完整资料的评测者最可能给出的最终标签：
supported、refuted、not_determined。不要因为本次隐藏资料就一律选not_determined；请尽力预测标签。
你不知道各类数量、任何状态取值或真实答案。陈述本身不是观察事实，不要据其肯定语气直接认定为真。
直接调用submit_answer，保留七个item_id，evidence_ids为空列表，action_status为not_requested。
'''


class World(BaseWorld):
    def initial_input(self):
        result=super().initial_input()
        result['claims']=[{k:c[k] for k in ('item_id','claim_type','statement')} for c in self.public['claims']]
        for claim, public_claim in zip(result['claims'], self.public['claims']):
            if 'reference_locator' in public_claim:
                claim['reference_locator'] = copy.deepcopy(public_claim['reference_locator'])
        result['workflow'].pop('expected_action_status',None)
        result['simulated_environment_notice']='原始医疗正文未改。临床场景是明确的模拟交接任务，不是真实随访；不同字母场景相互独立。'
        result['request']='根据各独立模拟场景的当前状态及所附原始资料，核对七项有界医疗陈述，不评价整体处方适宜性。'
        result['scenario_contract']=self.public['scenario_contract']
        result['source_notice']='本轮所有原始来源均开放；只以相应模拟场景作为本轮状态，不能用原始旧值或其他场景补齐。状态变更后须重新读取当前版本。'
        return result

    def fork_state(self, public, fresh=False):
        w=super().fork('key',fresh=fresh)
        old={d['document_id']:d for d in self.documents()}
        w.public=copy.deepcopy(public)
        current={d['document_id']:d for d in w.documents()}
        changed={k for k in current if fingerprint(current[k])!=fingerprint(old.get(k))}
        w.observed-={b['anchor_id'] for k in changed for b in current[k]['blocks']}
        w.seen_documents-=changed
        return w


def ledger(base, with_evidence=True):
    value=base_ledger(base,with_evidence)
    value['method']='actual_observed_evidence_with_scope_review' if with_evidence else 'actual_prior_labels_only'
    if with_evidence:
        text={b['anchor_id']:b['text'] for d in base.documents() for b in d['blocks'] if b['anchor_id'] in base.observed}
        for row in value['entries']:
            row['prior_observed_evidence']=[{'anchor_id':a,'previous_text':text[a]} for a in row['evidence_ids'] if a in text]
        value['rule']='旧答案不是真值。逐项查当前场景版本，核对主体/标本/实测或计划/真实暴露/剂量分母/条件及来源规则。旧引用可能已过期；更新受影响结论，也修复旧错误。不能从没有记录推成阴性。保存前核权限，保存后读回。'
    return value


def score(w, gold, base=None):
    expected=gold[w.stage]; pred=(w.prediction or {}).get('items',[]); by={x['item_id']:x for x in pred}
    active={b['anchor_id'] for d in w.documents() for b in d['blocks']}
    rows=[]
    for g in expected:
        p=by.get(g['item_id'],{});refs=p.get('evidence_ids',[])
        observed=len(refs)==len(set(refs)) and set(refs)<=w.observed and set(refs)<=active
        enough=any(set(group)<=set(refs) for group in g['sufficient_evidence_groups'])
        if g['label']=='not_determined': enough={d['document_id'] for d in w.documents()}<=w.seen_documents
        rows.append({'item_id':g['item_id'],'gold':g['label'],'prediction':p.get('label'),
                     'content':p.get('label')==g['label'],'evidence':bool(observed and enough)})
    shape=w.shape(pred);action=(w.prediction or {}).get('action_status')==w.action()
    state=readback=None
    if not w.qa:
        expected_values={x['item_id']:x['label'] for x in expected}
        if w.action()=='saved': state=w.state['summary'] is not None and {x['item_id']:x['label'] for x in w.state['summary']}==expected_values and w.state['protected']==w.phase_start_state['protected']
        else: state=w.state==w.phase_start_state
        if w.public['environment']=='E3': readback=w.last_readback==w.state
    return {'content':bool(shape and all(x['content'] for x in rows)), 'structure':shape,
            'item_correct':sum(x['content'] for x in rows),'item_total':7,'item_results':rows,
            'evidence':bool(shape and all(x['evidence'] for x in rows)), 'actual_state':state,'readback':readback,
            'action_report':action,'tool_calls':len(w.trace),
            'unauthorized_attempts':sum(t['result'].get('error')=='not_authorized' for t in w.trace),
            'evidence_scope':'author-specified sufficient anchor groups, not independent clinical entailment validation'}
