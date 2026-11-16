"""Execute the constructed Figure 1 example with the frozen ScopeForm code.

This is a didactic example, not a benchmark observation or a model-generated
diagnosis. The edit is supplied explicitly to check the illustrated interface.
"""
from pathlib import Path
import copy
import json
import sys

FROZEN = Path('/Users/darrenlong/Desktop/长江/Drug Agent Project/drug-agent-pi/experiments/cpm_methods/versions/frame_repair_id_fields_20261114/frozen_code')
sys.path.insert(0, str(FROZEN))
from methods import frame_library, frame_protocol, frame_repair

OUT = Path(__file__).parent / 'illustration'
OUT.mkdir(exist_ok=True)
props = [{'name': 'object_id', 'type': 'String', 'semantic_type': 'Identity'}]
props += [{'name': n, 'type': 'Real', 'semantic_type': 'Dose', 'unit': 'mL'}
          for n in ('dose', 'dose_lower', 'dose_upper')]
props += [{'name': n, 'type': 'Real', 'semantic_type': 'Frequency', 'unit': '1/d'}
          for n in ('frequency', 'frequency_lower', 'frequency_upper')]

def record(identifier, time, modality, fields, quote):
    return dict(id=identifier, kind='regimen', subject='P', time=time,
                modality=modality, scenario='visit', document_id=identifier,
                anchor_id=identifier + '_1',
                fields={'object_id': 'Drug X', **fields},
                sources=[dict(document_id=identifier, location=identifier+'_1', quote=quote)])

graph = dict(properties=props, records=[
    record('r_old', 'past', 'actual', dict(dose=5, frequency=2),
           '此前实际用药：药品X，每次5mL，每日2次。'),
    record('r_order', 'current', 'order', dict(dose_lower=5, dose_upper=5,
           frequency_lower=2, frequency_upper=2), '本次医嘱：药品X，每次5mL，每日2次。'),
    record('r_now', 'current', 'actual', dict(dose=2.5, frequency=4),
           '本次实际用药：药品X，每次2.5mL，每日4次。'),
], references=[
    dict(id='obs', text='患者P本次实际用药', where=dict(subject='P', time='current', scenario='visit'), slots={}, scores={}),
    dict(id='ref', text='本次医嘱', where=dict(document_id='r_order', anchor_id='r_order_1'), slots={}, scores={}),
], rules=[], queries=[
    dict(item_id='Q1', text='本次实际名义日总量符合本次医嘱', expr=dict(op='daily_total_conforms', args=['obs', 'ref'])),
    dict(item_id='Q2', text='本次实际单次量和频次分别符合本次医嘱', expr=dict(op='dose_frequency_conforms', args=['obs', 'ref'])),
])

def compact(g):
    result = frame_library.solve(frame_protocol.decode(g))
    return {k: result[k] for k in ('base_status', 'items', 'selection', 'compiled')}

correct = compact(graph)
missing = copy.deepcopy(graph)
missing['records'][2]['fields']['frequency'] = None
missing['records'][2]['sources'][0]['quote'] = '本次实际用药：药品X，每次2.5mL，每日次数未记录。'
unknown = compact(missing)
wrong = copy.deepcopy(graph)
wrong['references'][0]['where']['time'] = 'past'
before = compact(wrong)
diagnosis = {'issues': [{
    'category': 'context_fidelity',
    'reason': 'The query concerns current administration; obs incorrectly selects past administration.',
    'evidence': [{'document_id': 'r_now', 'anchor_id': 'r_now_1', 'quote': '本次实际用药'}],
    'edits': [{'collection': 'references', 'id': 'obs', 'path': ['where', 'time'], 'value': 'current'}],
}]}
categories = frame_repair.load_taxonomy()['categories']
context = next(c['id'] for c in categories if 'context' in c['id'])
diagnosis['issues'][0]['category'] = context
fixed = frame_repair.apply_repairs(wrong, diagnosis)
after = compact(fixed)
labels = lambda r: {x['item_id']: x['label'] for x in r['items']}
assert labels(correct) == {'Q1': 'supported', 'Q2': 'refuted'}
assert labels(unknown) == {'Q1': 'not_determined', 'Q2': 'refuted'}
assert labels(before) == {'Q1': 'supported', 'Q2': 'supported'}
assert labels(after) == labels(correct)
assert graph['records'] == fixed['records']
for name, data in [('graph', graph), ('correct', correct), ('wrong_binding', before),
                   ('missing_frequency', unknown), ('illustrative_diagnosis', diagnosis), ('after_id_patch', after)]:
    (OUT / (name + '.json')).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'illustrative_example': True, 'model_calls': 0,
                  'correct': labels(correct), 'wrong_binding': labels(before),
                  'missing_frequency': labels(unknown), 'after_id_patch': labels(after),
                  'records_unchanged': True}, indent=2))
