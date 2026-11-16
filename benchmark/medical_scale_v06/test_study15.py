"""Offline synthetic interface, de-duplication and exact checkpoint tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import study15 as s
import confirmation14_workflow as h
from test_confirmation14_workflow import bundle,Tokenizer,SETTINGS
from run_generation import read
from pilot_runtime import fingerprint
from validate_workflow11_offline import complete
import score_study15_local as score


class Study15Tests(unittest.TestCase):
    def test_synthetic_model_interface_nested_write_readback(self):
        def model(port,body):
            value=json.loads(body['messages'][0]['content'].split('\n',1)[1])
            return {'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':json.dumps(value)}}]}
        with tempfile.TemporaryDirectory() as d,patch.object(h,'post',side_effect=model) as mocked:
            s.preflight(1,'synthetic',Tokenizer(),SETTINGS,Path(d))
            report=read(Path(d)/'preflight.json')
            self.assertTrue(report['passed']);self.assertTrue(report['nested_write_and_readback'])
            self.assertFalse(report['contains_medical_data']);self.assertEqual(mocked.call_count,5)
            s.preflight(1,'synthetic',Tokenizer(),SETTINGS,Path(d));self.assertEqual(mocked.call_count,5)

    def test_identical_stem_only_states_are_physically_deduplicated(self):
        case=bundle()
        records=[{'source_id':'SYN','packet_id':p['packet_id'],'controller':p['controller']} for p in case['packets']]
        jobs=s.static_jobs(records)
        self.assertEqual(sum(len(j['members']) for j in jobs),9)
        self.assertEqual(len([j for j in jobs if j['condition']=='stem_only']),1)
        self.assertEqual(len(jobs),7)
        for j in jobs:
            self.assertNotIn('packet_id',j['actor']);self.assertNotIn('items',j['actor'])
            if j['condition']=='stem_only':self.assertFalse(j['actor']['materials'])
            if j['condition']=='source_withheld':
                self.assertTrue(all(x['origin_kind']=='explicit_synthetic_chartlets' for x in j['actor']['materials']))

    def test_unsubmitted_base_keeps_twenty_blocked_conditions(self):
        b=bundle();actor={'source_id':'SYN','initial':b['initial']['controller'],
            'events':[{'event_id':e['event_id'],'controller':e['controller']} for e in b['events']]}
        def fake_phase(world,*args):return world,'action_limit',0,0,{'input_tokens':0,'output_tokens':0}
        with tempfile.TemporaryDirectory() as d,patch.object(h,'phase',side_effect=fake_phase):
            s.workflow([actor],Path(d),1,'synthetic',Tokenizer(),{**SETTINGS,'workers_per_model':2},h.METHODS)
            rows=read(Path(d)/'results.json')
            self.assertEqual(len(rows),21)
            self.assertEqual(sum(r['status']=='blocked_actual_initial_not_submitted' for r in rows),20)
            self.assertTrue(all(not r['submitted'] for r in rows))

    def test_static_failures_not_resampled_or_called_content_zero(self):
        b=bundle();p=b['packets'][0]
        job=s.static_jobs([{'source_id':'SYN','packet_id':p['packet_id'],'controller':p['controller']}])[0]
        with tempfile.TemporaryDirectory() as d,patch.object(h,'post',side_effect=TimeoutError) as mocked:
            one=s.static_one(job,Path(d),1,'synthetic',Tokenizer(),SETTINGS)
            two=s.static_one(job,Path(d),1,'synthetic',Tokenizer(),SETTINGS)
            self.assertFalse(one['submitted']);self.assertEqual(one,two);self.assertEqual(mocked.call_count,1)
            self.assertNotIn('correct',one)

    def test_local_scorer_requires_true_base_and_accepts_exact_replay(self):
        b=bundle();actor={'source_id':'SYN','initial':b['initial']['controller'],
            'events':[{'event_id':e['event_id'],'controller':e['controller']} for e in b['events']]}
        target={'source_id':'SYN','source_cluster':'synthetic','modality':'drug','construction_subset':'synthetic',
            'initial':b['initial']['items'],'events':[{'event_id':e['event_id'],'items':e['items']} for e in b['events']]}
        def phase(world,*args):
            items=b['initial']['items'] if world.event_kind=='initial' else next(e['items'] for e in b['events'] if e['event_id']==world.event_kind)
            complete(world,items);return world,'submitted',0,0,{'input_tokens':0,'output_tokens':0}
        with tempfile.TemporaryDirectory() as d,patch.object(h,'phase',side_effect=phase):
            base,w=h.one(actor,None,'initial',Path(d),1,'synthetic',Tokenizer(),SETTINGS)
            row,_=h.one(actor,actor['events'][0],'evidence_constraint',Path(d),1,'synthetic',Tokenizer(),SETTINGS,w)
            audits={'SYN':{'initial_world_sha256':fingerprint(h.prior.snapshot(w))}}
            rows=score.enrich([base,row],{'SYN':actor},{'SYN':target},audits)
            self.assertTrue(all(r['score']['strict'] for r in rows))
            broken=copy.deepcopy(row);broken['initial_checkpoint_hash']='fabricated'
            with self.assertRaises(ValueError):score.enrich([base,broken],{'SYN':actor},{'SYN':target},audits)


if __name__=='__main__':unittest.main()
