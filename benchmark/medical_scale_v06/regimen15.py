"""Source-information screen and exact bounded regimen reconciliation.

No model, diagnostic thresholds or inferred treatment eligibility. Extensions
cover either clause order, per-administration notation, explicitly divided
daily totals and explicit medication-course duration. All native quantities
must occur in the same original block; original text is never rewritten.
"""
import copy
import json
import re
import threading
import quantitative_target09 as q
import scale09_rules_r4 as rules
from finalize_medical09 import operator_count

BASE_PARSE=q.parse
LOCK=threading.RLock()
NUM=q.NUM;UNIT=r'(?:毫克|毫升|微克|mg|mL|ml|μg|ug|克|g|片|粒|袋|支|丸|滴|喷|揿)'
DOSE=re.compile(r'(?:每次|一次|单次)\s*('+NUM+r')\s*(?:[-—~～至到]\s*('+NUM+r'))?\s*('+UNIT+r')',re.I)
DOSE_SLASH=re.compile(r'('+NUM+r')\s*(?:[-—~～至到]\s*('+NUM+r'))?\s*('+UNIT+r')\s*[/／]\s*次',re.I)
FREQ=re.compile(r'(?:每日|每天|一日|一天|1天)\s*('+NUM+r')\s*(?:[-—~～至到]\s*('+NUM+r'))?\s*次')
FREQ_SLASH=re.compile(r'('+NUM+r')\s*(?:[-—~～至到]\s*('+NUM+r'))?\s*次\s*[/／]\s*(?:日|天)')
TOTAL=re.compile(r'(?:每日(?:总量|剂量)?|每天(?:总量|剂量)?|一日(?:总量|剂量)?|日总量|日剂量)\s*(?:为|是|[:：])?\s*('+NUM+r')\s*(?:[-—~～至到]\s*('+NUM+r'))?\s*('+UNIT+r')',re.I)
DIVIDED=re.compile(r'分\s*('+NUM+r')\s*(?:[-—~～至到]\s*('+NUM+r'))?\s*次')
COURSE=re.compile(r'(?:疗程\s*(?:为|是)?|连(?:续)?(?:服用|使用|用药|服药|口服|用|服|涂|滴|喷)\s*(?:为)?|(?:口服|服用|用药)\s*(?:共计|共))\s*('+NUM+r')\s*(?:[-—~～至到]\s*('+NUM+r'))?\s*(天|日)')
PATIENT_FACT=re.compile(r'(?:我|孩子|宝宝|患者|目前|现在|之前|一直|已经|已|正在).{0,18}(?:服|吃|用药|口服|喷|涂|滴)')
PATIENT_UNCERTAIN=re.compile(r'[?？]|能否|是否|可不可以|可以吗|能不能|建议|打算|准备|想要|想吃|想用|计划|担心')
MEDICATION=re.compile(r'服|用药|给药|口服|注射|静脉|滴|喷|涂|片|粒|胶囊|药')


def valid_role(source,block):
    if source['modality']=='drug':return 'drug_instruction'
    if block.get('role')=='医生一方':return 'clinician_recorded_regimen_or_advice'
    text=block['text']
    if PATIENT_FACT.search(text) and not PATIENT_UNCERTAIN.search(text):return 'explicit_patient_reported_exposure_reference'
    return None


def match_values(match):
    a=q.number(match[1]);z=q.number(match[2] or match[1])
    if not 0<a<=z<100000:raise ValueError('Invalid native interval')
    return a,z


def parse(source):
    for document in source['documents']:
        for block in document['blocks']:
            role=valid_role(source,block)
            if role is None:continue
            text=block['text'];freqs=list(FREQ.finditer(text))+list(FREQ_SLASH.finditer(text))
            doses=list(DOSE.finditer(text))+list(DOSE_SLASH.finditer(text))
            totals=list(TOTAL.finditer(text));divided=list(DIVIDED.finditer(text));courses=list(COURSE.finditer(text))
            options=[]
            for kind,lefts,rights in [('single_frequency',doses,freqs),('daily_total_divided',totals,divided),('duration_frequency',courses,freqs)]:
                for left in lefts:
                    nearby=[]
                    for right in rights:
                        if max(left.start(),right.start())<min(left.end(),right.end()):continue
                        start=min(left.start(),right.start());end=max(left.end(),right.end())
                        between=text[min(left.end(),right.end()):max(left.start(),right.start())]
                        if len(between)>90 or re.search(r'[。；;!?！？\n]',between):continue
                        if q.BAD.search(text[start:end]):continue
                        # Other dose/frequency instructions between the pair
                        # would leave drug/subgroup pairing ambiguous.
                        if any(m is not left and m is not right and start<m.start()<end for m in lefts+rights):continue
                        try:
                            a,z=match_values(left);b,y=match_values(right)
                            if y>24:continue
                        except ValueError:continue
                        if not MEDICATION.search(text[max(0,start-30):min(len(text),end+30)]):continue
                        if re.match(r'\s*(?:[/／]\s*[周月]|[?？吗么])',text[end:end+8]):continue
                        nearby.append((start,end,left,right,(a,z,b,y)))
                    # Do not resolve two competing subgroup regimens by guess.
                    if len(nearby)==1:
                        start,end,l,r,values=nearby[0]
                        options.append((start,end,kind,l,r,values))
            if not options:continue
            start,end,kind,left,right,values=sorted(options,key=lambda x:(x[0],x[1],x[2]))[0]
            return {'kind':kind,'anchor_id':block['anchor_id'],'document_id':document['document_id'],
                'first_unit':left[3],'single_min':values[0],'single_max':values[1],
                'daily_min':values[2],'daily_max':values[3],'unit':left[3],
                'original_contiguous_quote':text[start:end],'character_span_private':[start,end],
                'source_block_sha256':q.digest_text(text),'record_kind':role,
                'doctor_or_instruction_only':role!='explicit_patient_reported_exposure_reference',
                'medical_scope':'Bounded quantitative reconciliation with this named original record, not prescribing, diagnosis, safety or proof of actual adherence.'}
    return None


def construct(source):
    parsed=parse(source)
    if parsed is None:return None,{'reason':'no_unambiguous_linked_native_quantities_under_frozen_source_screen'}
    if parsed['kind']=='single_frequency':
        # The old rule compiler is preserved; only its already-parsed exact
        # clause is injected in memory. No original text/role/order is changed.
        with LOCK:
            old=q.parse;q.parse=lambda _:parsed
            try:items,report=q.construct(source)
            finally:q.parse=old
        if items is None:return None,report
    else:
        aid=parsed['anchor_id'];block=next(b for d in source['documents'] for b in d['blocks'] if b['anchor_id']==aid)
        a,z,b,y=[parsed[k] for k in ('single_min','single_max','daily_min','daily_max')]
        duration=parsed['kind']=='duration_frequency'
        key='actual_course_days' if duration else 'actual_daily_total'
        label='本轮记录的用药疗程时长' if duration else '本轮记录的日合计使用量'
        fields=[{'key':key,'label':label,'type':'number','unit':parsed['first_unit']},
            {'key':'actual_daily_frequency','label':'本轮记录的每日使用次数','type':'number','unit':'次/日'}]
        constants=[{'key':k,'value_json':json.dumps(val),'unit':unit,'anchor_id':aid,'quote':block['text']}
            for k,val,unit in [('a_min',a,parsed['first_unit']),('a_max',z,parsed['first_unit']),('b_min',b,'次/日'),('b_max',y,'次/日')]]
        f=lambda k:{'field':k};c=lambda k:{'constant':k};op=lambda o,*xs:{'op':o,'args':list(xs)}
        arithmetic='mul' if duration else 'div'
        actual=op(arithmetic,f(key),f('actual_daily_frequency'))
        low=op(arithmetic,c('a_min'),c('b_min' if duration else 'b_max'))
        high=op(arithmetic,c('a_max'),c('b_max' if duration else 'b_min'))
        exprs=[op('and',op('ge',actual,low),op('le',actual,high)),
            op('and',op('ge',f(key),c('a_min')),op('le',f(key),c('a_max')),
                op('ge',f('actual_daily_frequency'),c('b_min')),op('le',f('actual_daily_frequency'),c('b_max')))]
        scope='仅相对指定原文锚点“'+aid+'”中reference_locator定位的明确记载：'
        stems=[scope+('本轮疗程时长与每日次数相乘的名义总使用次数，位于该原文记载推导的区间内。' if duration
            else '本轮日合计使用量除以每日次数得到的名义平均单次量，位于该原文记载推导的区间内。'),
            scope+'本轮记录的'+('疗程时长' if duration else '日合计使用量')+'与每日次数，分别位于该参照记载的两个区间内。']
        base={key:a,'actual_daily_frequency':b};delta=max(.01,z*.1)
        bad=(z*y/b+delta) if duration else z+delta
        states=[[base,{key:bad,'actual_daily_frequency':b},{key:a,'actual_daily_frequency':None}],
            [base,{key:z+delta,'actual_daily_frequency':b},{key:None,'actual_daily_frequency':b}]]
        native=[]
        for stem,expr,values in zip(stems,exprs,states):
            used={n['constant'] for n in rules.walk(expr) if isinstance(n,dict) and 'constant' in n}
            native.append({'statement':stem,'family_type':'multi_step','references':[{'anchor_id':aid,'quote':block['text']}],
                'fields':fields,'constants':[c for c in constants if c['key'] in used],
                'expression_json':json.dumps(expr),'states':[[{'key':k,'value_json':json.dumps(val)} for k,val in s.items()] for s in values],
                'clinical_scope':'仅核对指定来源的明确记录及名义计算。不能据此证明治疗适宜、安全、疗效、分次实际均等或已按医嘱执行。',
                'reasoning_steps':['回查同一原段落中的原生数量、角色与两项关联记录。','按明确单位计算乘积或比率，再区别名义合计关系与逐项记录约束。'],
                'missing_information':''})
        native.append({'statement':scope+'本轮记录的'+('用药疗程时长' if duration else '日合计使用量')+'达到该参照记载的下界。',
            'family_type':'stable_insufficiency','references':[{'anchor_id':aid,'quote':block['text']}],
            'fields':[fields[0]],'constants':[constants[0]],'expression_json':json.dumps(op('ge',f(key),c('a_min'))),
            'states':[[{'key':key,'value_json':'null'}] for _ in range(3)],
            'clinical_scope':'原文完整也不能替代当前独立模拟患者缺失的实际临床记录，不将建议或历史参照当成当前事实。',
            'reasoning_steps':['定位原始参照及其角色、时间和数量。','当前模拟患者该项记录缺失，不能从历史参照填补。'],
            'missing_information':'缺少本轮独立模拟患者该项实际临床记录。'})
        items,errors=rules.validate({'source_id':source['source_id'],'items':native},source)
        if items is None:return None,{'reason':'typed_rule_validation','errors':errors}
        report={'parsed':parsed,'new_model_calls':0}
    for item in items:
        gap=item['family_type']=='stable_insufficiency'
        item['counterfactual_design_private']={'stratum':'persistent_insufficiency' if gap else 'hidden_reference_clinical_parameter',
            'primary_content_challenge':not gap,'reference_hidden_from_stem':not gap,'execution_flags_held_fixed':True,
            'same_medical_rule_and_statement':True,'chosen_using_solver_results':False}
        item['parsed_regimen_private']=copy.deepcopy(parsed)
        item['construction_provenance_private']={'kind':'source_exact_regimen15','source_id':source['source_id'],'relation_kind':parsed['kind']}
        item['nonredundant_operator_count_private']=operator_count(item['expression'])
        item['reference_locator_public']={'anchor_id':parsed['anchor_id'],'start_character':parsed['character_span_private'][0],
            'end_character':parsed['character_span_private'][1],'indexing':'zero-based Unicode codepoint, half-open',
            'purpose':'Locate the exact original clause and role, without copying reference values or truth labels.'}
        item['task_scope_private']='bounded_regimen_information_reconciliation_not_clinical_diagnosis_or_treatment_safety'
    return items,{**report,'parsed':parsed,'clinical_expert_gold':False,'new_model_calls':0}
