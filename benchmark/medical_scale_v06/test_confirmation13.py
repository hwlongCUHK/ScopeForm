"""Synthetic-only tests; no network, model calls, private output or old writes."""
import copy
from pathlib import Path
import tempfile
import unittest
from collections import Counter

import confirmation13 as c
import quantitative_target09 as q


def synthetic_source(sid='UNIT13'):
    text = '本段只作程序单元测试。原文给定方案：每次1片，每日2次。不得把此测试用于真实患者。'
    return {'source_id': sid, 'source_cluster': sid + '-cluster', 'modality': 'drug',
            'split': 'locked_test', 'length_band': 0, 'characters': len(text),
            'documents': [{'document_id': sid + '-DOC', 'title': '合成说明书', 'kind': 'drug',
                           'origin_kind': 'synthetic_unit_test', 'blocks': [{'anchor_id': sid + '-A1',
                           'label': '用法用量', 'role': '说明书', 'origin_kind': 'synthetic_unit_test', 'text': text}]}]}


class Confirmation13Tests(unittest.TestCase):
    def test_append_only_frozen_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'a.json'
            c.frozen(path, {'v': 1})
            c.frozen(path, {'v': 1})
            with self.assertRaises(ValueError):
                c.frozen(path, {'v': 2})

    def test_plain_markdown_not_json_string(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'README.md'
            c.frozen(path, '# Title\n')
            self.assertEqual(path.read_text(encoding='utf-8'), '# Title\n')

    def test_direction_balance_by_source_before_labels(self):
        catalog = []
        for modality, n in [('dialogue', 24), ('drug', 16)]:
            for band in range(3):
                catalog.extend({'source_id': f'{modality}{band}_{k}', 'modality': modality, 'length_band': band} for k in range(n))
        rows = c.assignment(catalog)
        self.assertEqual(Counter(tuple(r['definite']) for r in rows.values()),
                         {(c.rules.S,c.rules.R):60,(c.rules.R,c.rules.S):60})
        self.assertEqual(sorted(Counter(tuple(r['uncertainty']) for r in rows.values()).values()), [30]*4)

    def test_duplicate_audit_catches_normalized_exact(self):
        a = synthetic_source('a')
        b = synthetic_source('b')
        audit = c.near_duplicates([a], [b])
        self.assertEqual(len(audit['exact_normalized_duplicate_pairs']), 1)

    def test_quantitative_source_rules_and_independent_evaluator(self):
        source = synthetic_source()
        items, report = q.construct(source)
        self.assertIsNotNone(items)
        self.assertEqual(c.check_items(items, source), (True, []))
        for item in items:
            for state in item['state_values']:
                self.assertEqual(c.independent_label(item, state), c.rules.outcome(item['expression'], state, item['constant_map']))

    def test_unknown_is_not_automatically_refuted_or_supported(self):
        node = {'op': 'and', 'args': [{'field': 'missing'}, False]}
        self.assertIs(c.exact_evaluate(node, {'missing': None}, {}), False)
        node['op'] = 'or'
        self.assertIsNone(c.exact_evaluate(node, {'missing': None}, {}))
        node['args'][1] = True
        self.assertIs(c.exact_evaluate(node, {'missing': None}, {}), True)

    def test_same_stem_triplets_and_all_directions(self):
        source = synthetic_source()
        items, _ = q.construct(source)
        for definite in [(c.rules.S,c.rules.R),(c.rules.R,c.rules.S)]:
            for uncertainty in [(c.rules.S,c.rules.U),(c.rules.R,c.rules.U),(c.rules.U,c.rules.S),(c.rules.U,c.rules.R)]:
                bundle = c.compile_source(source, items, {'definite': definite, 'uncertainty': uncertainty})
                self.assertTrue(c.verify_compiled(source, bundle))
                self.assertEqual(bundle['unique_authored_families'], 3)
                self.assertEqual(bundle['claim_instances_per_packet'], 4)

    def test_actor_contains_no_target_or_program(self):
        source = synthetic_source()
        items, _ = q.construct(source)
        bundle = c.compile_source(source, items, {'definite': [c.rules.S,c.rules.R], 'uncertainty': [c.rules.U,c.rules.R]})
        for packet in bundle['packets']+[bundle['initial']]+bundle['events']:
            self.assertFalse(set(packet['controller']) & {'gold','items','expression','constants','proofs_private','labels'})

    def test_source_mutation_does_not_pass_roundtrip(self):
        source = synthetic_source()
        items, _ = q.construct(source)
        bundle = c.compile_source(source, items, {'definite': [c.rules.S,c.rules.R], 'uncertainty': [c.rules.U,c.rules.R]})
        bundle['initial']['controller']['documents'][0]['blocks'][0]['text'] += 'changed'
        with self.assertRaises(ValueError):
            c.verify_compiled(source, bundle)

    def test_source_parser_does_not_invent_rule(self):
        source = synthetic_source()
        source['documents'][0]['blocks'][0]['text'] = '本原文没有明确剂量和每日使用次数，不能据此编造用药规则。'
        items, report = q.construct(source)
        self.assertIsNone(items)


if __name__ == '__main__':
    unittest.main()
