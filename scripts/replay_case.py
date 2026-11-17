#!/usr/bin/env python3
"""Replay an externally supplied paired episode through the runtime; no model API."""
import argparse
import copy
import json
from pathlib import Path
import socket
import tarfile
import tempfile
from run_experiment import ROOT,load_five,read,dump

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--case',type=Path,required=True,help='External recorded case JSON with model request/response transcripts.')
p.add_argument('--initials-archive',type=Path,required=True,help='External shared-initial checkpoint archive containing this source.')
p.add_argument('--out',type=Path,default=ROOT/'output/recorded_replay')
a=p.parse_args()
def no_network(*args,**kw):raise RuntimeError('Recorded replay does not call a model API')
socket.create_connection=no_network;socket.socket.connect=no_network
case=read(a.case)
five,run=load_five('deepseek')
actor=next(json.loads(line) for line in (ROOT/'benchmark/medical_scale_v06/confirmation15_inputs_private/actors.private.jsonl').read_text().splitlines() if json.loads(line)['source_id']==case['source_id'])
event=next(e for e in actor['events'] if e['event_id']==case['event_id'])
with tarfile.open(a.initials_archive,'r:gz') as tar:
    cp=json.load(tar.extractfile('phases/'+case['source_id']+'/initial__initial/checkpoint.private.json'))

class ReplayClient:
    def __init__(self,transcript):self.transcript=transcript;self.index=0
    def chat(self,payload):
        item=self.transcript[self.index]
        assert payload==item['request'], 'Generated request differs from recorded request '+str(self.index+1)
        self.index+=1
        return copy.deepcopy(item['response'])

config=read(ROOT/'provenance/deepseek/config.original.json')
report={'source_id':case['source_id'],'event_id':case['event_id'],'setting':case['setting'],
        'mode':'replay_actual_model_responses_execute_runtime_and_solver','new_model_calls':0,'variants':{}}
for variant,data in case['variants'].items():
    client=ReplayClient(data['transcript'])
    runner=five.runner_class(run)(client,config['model'],config['settings'],'repair')
    run.h.phase=runner.phase
    with tempfile.TemporaryDirectory(prefix='scopeform_replay_') as temp:
        row,_=runner.one(actor,event,case['setting']+'__'+variant,Path(temp),base=run.h.restore(copy.deepcopy(cp['world'])))
        assert client.index==len(client.transcript), (variant,client.index,row['status'])
        for key,value in data['expected'].items():assert row[key]==value,(variant,key)
        report['variants'][variant]={'recorded_responses_replayed':client.index,
            'all_requests_match':True,'result_matches':True,'requests_charged':row['requests'],
            'status':row['status'],'prediction':row['prediction']}
report['passed']=True
dump(a.out/'verification.json',report)
print(json.dumps({k:v for k,v in report.items() if k!='variants'},ensure_ascii=False,indent=2))
