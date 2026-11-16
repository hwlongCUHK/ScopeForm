#!/usr/bin/env python3
"""Execute the paper's constructed medication example and ID-addressed repair, offline."""
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--backbone',choices=['deepseek','qwen'],default='deepseek')
p.add_argument('--out',type=Path,default=ROOT/'output/demo')
a=p.parse_args()
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'code'/a.backbone/'frozen_code'))
from methods import frame_library,frame_protocol,frame_repair
read=lambda n:json.loads((ROOT/'examples/illustration'/n).read_text())
graph=read('graph.json')
def solve(g):
 r=frame_library.solve(frame_protocol.decode(g))
 return {x['item_id']:x['label'] for x in r['items']}
wrong=copy.deepcopy(graph);wrong['references'][0]['where']['time']='past'
fixed=frame_repair.apply_repairs(wrong,read('illustrative_diagnosis.json'))
missing=copy.deepcopy(graph);missing['records'][2]['fields']['frequency']=None
missing['records'][2]['sources'][0]['quote']='本次实际用药：药品X，每次2.5mL，每日次数未记录。'
result={'illustrative_example':True,'model_calls':0,'backbone':a.backbone,
        'wrong_binding':solve(wrong),'after_id_repair':solve(fixed),'missing_frequency':solve(missing),
        'source_records_unchanged':graph['records']==fixed['records']}
assert result['wrong_binding']=={'Q1':'supported','Q2':'supported'}
assert result['after_id_repair']=={'Q1':'supported','Q2':'refuted'}
assert result['missing_frequency']=={'Q1':'not_determined','Q2':'refuted'}
assert result['source_records_unchanged']
a.out.mkdir(parents=True,exist_ok=True)
(a.out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
