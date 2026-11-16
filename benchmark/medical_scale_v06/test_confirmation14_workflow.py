"""Synthetic known-answer harness tests, not model accuracy experiments."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import confirmation14 as v
import confirmation14_workflow as h
from test_confirmation13 import synthetic_source
import quantitative_target09 as q
from validate_workflow11_offline import complete
from run_generation import dump
from pilot_runtime import fingerprint


class Tokenizer:
    def apply_chat_template(self,*args,**kwargs):return [1,2,3]


def bundle():
    source=synthetic_source('ACTION14');items,_=q.construct(source)
    return v.compile_source(source,items,{'definite':[v.c.rules.S,v.c.rules.R],'uncertainty':[v.c.rules.U,v.c.rules.S]})


SETTINGS={'temperature':0,'seed':20261022,'max_output_tokens':128,'context_limit':65536,'max_actions':40}


class SharedActionTests(unittest.TestCase):
    def test_known_answer_serial_tool_replay_really_submits_and_saves(self):
        case=bundle();oracle=complete(h.World(case['initial']['controller']),case['initial']['items'])
        actions=[{'name':t['tool'],'arguments':t['arguments']} for t in oracle.trace]
        responses=[{'choices':[{'finish_reason':'stop','message':{'role':'assistant',
            'content':json.dumps({'action':a},ensure_ascii=False)}}],'usage':{'prompt_tokens':3,'completion_tokens':4}} for a in actions]
        with tempfile.TemporaryDirectory() as d,patch.object(h,'post',side_effect=responses) as model:
            world,status,calls,seconds,usage=h.phase(h.World(case['initial']['controller']),Path(d),1,'synthetic',Tokenizer(),SETTINGS)
            self.assertTrue(world.done);self.assertEqual(status,'submitted');self.assertEqual(calls,len(actions))
            self.assertEqual(world.state,oracle.state);self.assertEqual(usage['input_tokens'],3*len(actions))
            # A terminal checkpoint reuses the actual state without a new request.
            again=h.phase(h.World(case['initial']['controller']),Path(d),1,'synthetic',Tokenizer(),SETTINGS)
            self.assertTrue(again[0].done);self.assertEqual(model.call_count,len(actions))

    def test_all_five_methods_fork_identical_actual_state(self):
        case=bundle();base=complete(h.World(case['initial']['controller']),case['initial']['items'])
        for method in h.METHODS:
            world=h.branch(base,case['events'][0]['controller'],case['events'][0]['event_id'],method)
            self.assertEqual(fingerprint(world.state),fingerprint(base.state))
            self.assertFalse(world.observed);self.assertFalse(world.seen_documents)
            self.assertEqual(h.restore(h.prior.snapshot(world)).initial_input(),world.initial_input())

    def test_lost_response_is_not_resampled_or_called_content_wrong(self):
        case=bundle()
        with tempfile.TemporaryDirectory() as d,patch.object(h,'post') as model:
            folder=Path(d);dump(folder/'dispatch_001.json',{'request_sha256':'unacknowledged'})
            world,status,calls,_,_=h.phase(h.World(case['initial']['controller']),folder,1,'synthetic',Tokenizer(),SETTINGS)
            self.assertFalse(world.done);self.assertEqual(status,'indeterminate_inflight_no_resample');self.assertEqual(calls,1);model.assert_not_called()

    def test_blocked_base_retained_and_context_is_never_truncated(self):
        case=bundle();base=h.World(case['initial']['controller'])
        with tempfile.TemporaryDirectory() as d,patch.object(h,'post') as model:
            actor={'source_id':'ACTION14','initial':case['initial']['controller']}
            row,_=h.one(actor,case['events'][0],'fresh',Path(d),1,'synthetic',Tokenizer(),SETTINGS,base)
            self.assertEqual(row['status'],'blocked_actual_initial_not_submitted');self.assertEqual(row['requests'],0)
            with self.assertRaises(ValueError):h.body(h.messages_for(base),base.tools(),'synthetic',Tokenizer(),{**SETTINGS,'context_limit':2})
            model.assert_not_called()

    def test_schema_serial_tool_identity_is_tied_to_arguments(self):
        world=h.World(bundle()['initial']['controller']);schema=h.action_schema(world.tools())
        options=schema['properties']['action']['anyOf']
        self.assertEqual(len(options),len(world.tools()))
        for option,tool in zip(options,world.tools()):
            self.assertEqual(option['properties']['name']['enum'],[tool['function']['name']])
            self.assertEqual(option['properties']['arguments'],tool['function']['parameters'])
        with self.assertRaises(ValueError):h.decode({'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':'{"action":[]}'}}]})


if __name__=='__main__':unittest.main()
