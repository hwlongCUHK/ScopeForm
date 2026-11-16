"""Read-only audit of recorded first-formalization repair trajectories.

No model or solver is invoked. Gold is read only from an existing offline scored
file; it is never written to run inputs or sent to the actor. Guard rereading is
excluded from the primary comparison by selecting only attempt 001 per phase.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

RUN = Path('/Users/darrenlong/Desktop/长江/Drug Agent Project/drug-agent-pi/experiments/cpm_methods/runs/frame_five_settings_repair_idfix_full_20261114')
OUT = Path(__file__).parent / 'repair_trace_audit'
SCORED = RUN / 'evaluation/scored_rows.LOCAL_ONLY.private.json'
CATEGORIES = ('representation_validity', 'concept_role_fidelity', 'context_fidelity', 'logic_knowledge_fidelity')
GOLD_KEYS = {'gold', 'gold_label', 'gold_labels', 'target_label', 'target_labels', 'correct_answer', 'evaluator_targets', 'private_target'}


def read(path):
    return json.loads(path.read_text())


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def forbidden_keys(value, prefix=''):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in GOLD_KEYS:
                found.append(prefix + '/' + key)
            found += forbidden_keys(child, prefix + '/' + key)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            found += forbidden_keys(child, prefix + '/' + str(idx))
    return found


def result_stats(result, gold):
    labels = {x['item_id']: x['label'] for x in result.get('items', [])}
    executable = result['status'] == 'completed' and set(labels) == set(gold)
    correct = sum(labels[k] == value for k, value in gold.items()) if executable else 0
    return dict(status=result['status'], executable=executable, correct=correct,
                exact=executable and correct == len(gold), score=correct / len(gold),
                labels=labels, error_type=result.get('error_type'), error=result.get('error'))


def method_dirs(row):
    phase = RUN / 'main' / row['phase_relative_path']
    guarded = sorted((phase / 'guard_rounds').glob('[0-9][0-9][0-9]/method'))
    return guarded or [phase / 'method']


def collect(row, method, attempt):
    before_path = method / 'repair_before_result.json'
    if not before_path.exists():
        return None
    assert (method / 'repair_after_result.json').exists(), str(method)
    gold = {x['item_id']: x['to'] for x in row['gold_transition_local']}
    if row.get('score'):
        assert gold == {x['item_id']: x['gold'] for x in row['score']['items']}
    before_raw, after_raw = read(before_path), read(method / 'repair_after_result.json')
    before, after = result_stats(before_raw, gold), result_stats(after_raw, gold)
    public = read(method / 'method_input.json')
    repair = read(method / 'repair_input.json')
    expected_public = {k: public.get(k, []) for k in ('task_input', 'observations')}
    assert repair['problem'] == expected_public, str(method)
    assert not forbidden_keys(repair['problem']), str(method)
    b_hash = before_raw.get('diagnostics', {}).get('evidence_sha256')
    a_hash = after_raw.get('diagnostics', {}).get('evidence_sha256')
    # This runtime hash covers compiled scopes/properties/facts, NOT retrieved
    # observations. A source-grounded edit may legitimately change this hash.
    original, final = read(method / 'repair_original_graph.json'), read(method / 'repair_final_graph.json')
    rounds = [read(p) for p in sorted(method.glob('repair_round_[0-9][0-9][0-9].json'))]
    assert rounds and rounds[-1]['result_after'] == after_raw
    issues = [issue for rr in rounds for issue in (rr.get('diagnosis') or {}).get('issues', [])]
    first_issues = (rounds[0].get('diagnosis') or {}).get('issues', [])
    reported = Counter(i.get('category', 'missing') for i in issues)
    initial_reported = Counter(i.get('category', 'missing') for i in first_issues)
    for category in reported:
        assert category in CATEGORIES, (category, method)
    transitions = Counter()
    fixed = harmed = 0
    if before['executable'] and after['executable']:
        for k, truth in gold.items():
            b, a = before['labels'][k], after['labels'][k]
            if b != a:
                transitions[f'{b} -> {a}; gold={truth}'] += 1
            fixed += b != truth and a == truth
            harmed += b == truth and a != truth
    return dict(source_id=row['source_id'], event_id=row['event_id'], setting=row['paper_setting'],
                attempt=attempt, method_path=str(method.relative_to(RUN)), claims=len(gold),
                before=before, after=after, submitted_final_phase=row['submitted'],
                phase_status=row['status'], graph_changed=original != final,
                before_graph_sha256=sha(original), after_graph_sha256=sha(final),
                public_input_sha256=sha(expected_public), diagnostic_rounds=len(rounds),
                compiled_facts_hash_comparable=bool(b_hash and a_hash),
                compiled_facts_changed=bool(b_hash and a_hash and b_hash != a_hash),
                first_issue_count=len(first_issues), first_issue_categories=dict(initial_reported),
                all_reported_issue_categories=dict(reported),
                claim_fixed=fixed, claim_harmed=harmed, changed_label_transitions=dict(transitions))


def aggregate(rows):
    n = len(rows)
    c = Counter()
    first_categories, all_categories, category_phases = Counter(), Counter(), Counter()
    per_source = defaultdict(list)
    transitions = Counter()
    for r in rows:
        b, a = r['before'], r['after']
        bv, av = b['executable'], a['executable']
        c['pairs'] += 1
        c['claims'] += r['claims']
        c['before_executable'] += bv
        c['after_executable'] += av
        c['before_exact_packets'] += b['exact']
        c['after_exact_packets'] += a['exact']
        c['before_correct_labels'] += b['correct']
        c['after_correct_labels'] += a['correct']
        c['before_invalid_after_valid'] += not bv and av
        c['before_valid_after_invalid'] += bv and not av
        c['neither_valid'] += not bv and not av
        if not bv and av:
            c['rescued_execution_correct_labels'] += a['correct']
            c['rescued_execution_claims'] += r['claims']
            c['rescued_execution_exact_packets'] += a['exact']
        if bv and not av:
            c['broken_execution_lost_correct_labels'] += b['correct']
            c['broken_execution_claims'] += r['claims']
            c['broken_execution_lost_exact_packets'] += b['exact']
        if bv and av:
            c['both_executable_pairs'] += 1
            c['both_executable_claims'] += r['claims']
            c['both_executable_before_correct'] += b['correct']
            c['both_executable_after_correct'] += a['correct']
            c['both_executable_before_exact'] += b['exact']
            c['both_executable_after_exact'] += a['exact']
            c['both_executable_exact_gained'] += not b['exact'] and a['exact']
            c['both_executable_exact_lost'] += b['exact'] and not a['exact']
            c['both_executable_better_packets'] += a['correct'] > b['correct']
            c['both_executable_worse_packets'] += a['correct'] < b['correct']
            c['both_executable_equal_packets'] += a['correct'] == b['correct']
            c['wrong_to_right_labels'] += r['claim_fixed']
            c['right_to_wrong_labels'] += r['claim_harmed']
        c['first_diagnosis_with_issues'] += r['first_issue_count'] > 0
        c['first_diagnosis_with_issues_initially_executable'] += r['first_issue_count'] > 0 and bv
        c['graph_changed'] += r['graph_changed']
        c['graph_changed_initially_executable'] += r['graph_changed'] and bv
        c['compiled_facts_hash_comparable'] += r['compiled_facts_hash_comparable']
        c['compiled_facts_changed'] += r['compiled_facts_changed']
        c['diagnostic_rounds'] += r['diagnostic_rounds']
        c['feedback_rounds_after_initial'] += r['diagnostic_rounds'] - 1
        first_categories.update(r['first_issue_categories'])
        all_categories.update(r['all_reported_issue_categories'])
        category_phases.update(r['first_issue_categories'].keys())
        per_source[r['source_id']].append(r)
        transitions.update(r['changed_label_transitions'])
    before_phase = statistics.mean(r['before']['score'] for r in rows) * 100 if n else None
    after_phase = statistics.mean(r['after']['score'] for r in rows) * 100 if n else None
    before_source = statistics.mean(statistics.mean(r['before']['score'] for r in rs) for rs in per_source.values()) * 100 if n else None
    after_source = statistics.mean(statistics.mean(r['after']['score'] for r in rs) for rs in per_source.values()) * 100 if n else None
    c['net_correct_labels'] = c['after_correct_labels'] - c['before_correct_labels']
    assert c['net_correct_labels'] == (c['rescued_execution_correct_labels'] -
          c['broken_execution_lost_correct_labels'] + c['wrong_to_right_labels'] - c['right_to_wrong_labels'])
    return dict(counts=dict(c), sources=len(per_source),
                first_diagnosis_issue_counts=dict(first_categories),
                first_diagnosis_category_phase_counts=dict(category_phases),
                all_round_issue_counts=dict(all_categories),
                changed_label_transitions=dict(transitions),
                before_executable_percent=100*c['before_executable']/n if n else None,
                after_executable_percent=100*c['after_executable']/n if n else None,
                phase_macro_before=before_phase, phase_macro_after=after_phase,
                phase_macro_delta_pp=after_phase-before_phase if n else None,
                source_macro_before=before_source, source_macro_after=after_source,
                source_macro_delta_pp=after_source-before_source if n else None)


def main():
    rows = [r for r in read(SCORED) if r['variant'] == 'formalization' and r['event_id'] != 'initial']
    first, later, missing = [], [], []
    for row in rows:
        methods = method_dirs(row)
        item = collect(row, methods[0], 1)
        if item is None:
            missing.append({k: row[k] for k in ('source_id', 'event_id', 'paper_setting', 'status')})
        else:
            first.append(item)
        for attempt, method in enumerate(methods[1:], 2):
            item = collect(row, method, attempt)
            if item is not None:
                later.append(item)
    summary = dict(kind='recorded_same_evidence_repair_trajectory_audit_not_ablation',
                   run=str(RUN), input_scored_sha256=hashlib.sha256(SCORED.read_bytes()).hexdigest(),
                   planned_phases=len(rows), observed_first_pairs=len(first),
                   unavailable_first_pairs=len(missing), unavailable_reasons=dict(Counter(r['status'] for r in missing)),
                   first_attempt=aggregate(first),
                   first_attempt_by_setting={s: aggregate([r for r in first if r['setting'] == s])
                                             for s in ('fresh', 'labels', 'index', 'guard', 'index_guard')},
                   later_guard_attempts_separate=aggregate(later),
                   audit=dict(no_new_model_or_solver_calls=True, original_run_read_only=True,
                              first_attempt_only_in_primary_analysis=True,
                              same_public_input_during_repair=True, no_gold_keys_in_repair_public_input=True,
                              public_input_equality_verified=True,
                              compiled_facts_hash_may_change_with_edits=True,
                              gold_only_read_from_existing_offline_scored_artifact=True,
                              counts_are_model_diagnosed_categories_not_manual_gold=True),
                   definitions=dict(executable='status=completed and label IDs equal the fixed claim IDs',
                                    score='Correct labels / planned labels in each observed pair; non-executable side scores zero.',
                                    phase_macro='Mean phase score over the fixed 2,372 observed first-attempt pairs.',
                                    source_macro='Mean phase score within each observed source, then mean across sources; excludes unavailable pairs on BOTH sides.',
                                    exact='Executable and every planned claim label correct.',
                                    repair_after='Last recorded repair result in the SAME first formalization attempt; no Guard reread between before and after.',
                                    later_guard='Separate descriptive counts; may contain repeated phases and additional evidence. Not pooled into primary analysis.'),
                   caveat='Trace audit of the recorded integrated method, not a randomized or separately executed no-repair ablation; before results are intermediate outputs, not actual submissions.')
    OUT.mkdir(exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    (OUT/'paired_traces.LOCAL_ONLY.private.json').write_text(json.dumps(dict(first=first, later_guard=later, missing=missing), ensure_ascii=False, indent=2)+'\n')
    a = summary['first_attempt']; c=a['counts']
    lines = ['# First-formalization diagnosis and repair: recorded-trajectory audit', '',
             f"The fixed primary slice contains {len(first)} first-attempt pairs from {a['sources']} sources across five settings. Of 2,400 planned phases, 20 were blocked by the shared initial state and 8 had no first graph because of native/transport/action failures. No unobserved before/after output is imputed.", '',
             'Before and after use the same public evidence within one formalization attempt. Later Guard rereads are excluded. Categories below are model-reported issue types, not manually verified error labels. This is a trace audit, not a component ablation.', '',
             '| Quantity | Before diagnosis | After repair |', '|---|---:|---:|',
             f"| Executable packets / {len(first)} | {c['before_executable']} ({a['before_executable_percent']:.2f}%) | {c['after_executable']} ({a['after_executable_percent']:.2f}%) |",
             f"| Exact packets / {len(first)} | {c['before_exact_packets']} | {c['after_exact_packets']} |",
             f"| Correct labels / {c['claims']} | {c['before_correct_labels']} | {c['after_correct_labels']} |",
             f"| Observed-slice phase-macro score | {a['phase_macro_before']:.4f}% | {a['phase_macro_after']:.4f}% |",
             f"| Observed-slice source-macro score | {a['source_macro_before']:.4f}% | {a['source_macro_after']:.4f}% |", '',
             'These scores use the same observed pair denominator and assign zero to non-executable outputs; they do not replace the full benchmark M, E, or J.', '',
             '## Decomposition', '',
             f"- Execution rescue: {c['before_invalid_after_valid']} packets, adding {c['rescued_execution_correct_labels']} correct labels and {c['rescued_execution_exact_packets']} exact packets.",
             f"- Execution breakage: {c['before_valid_after_invalid']} packets, losing {c['broken_execution_lost_correct_labels']} correct labels and {c['broken_execution_lost_exact_packets']} exact packets.",
             f"- Both executable: {c['both_executable_pairs']} pairs; {c['wrong_to_right_labels']} labels become correct and {c['right_to_wrong_labels']} become incorrect, a net {c['wrong_to_right_labels']-c['right_to_wrong_labels']:+d}. Exact packets change {c['both_executable_before_exact']} → {c['both_executable_after_exact']} ({c['both_executable_exact_gained']} gained; {c['both_executable_exact_lost']} lost).",
             f"- Total correct-label change: {c['rescued_execution_correct_labels']} − {c['broken_execution_lost_correct_labels']} + {c['wrong_to_right_labels']} − {c['right_to_wrong_labels']} = {c['net_correct_labels']:+d}.",
             f"- {c['neither_valid']} pairs remain non-executable. {c['diagnostic_rounds']} diagnostic calls are logged, including {c['feedback_rounds_after_initial']} feedback calls after the initial diagnosis.", '',
             'The observed net benefit comes from recovering executable coverage, while semantic edits to already executable graphs have a net negative label effect.', '',
             '## First diagnosis categories', '', '| Obligation | Reported issues | Phases reporting category |', '|---|---:|---:|']
    for category in CATEGORIES:
        lines.append(f"| {category} | {a['first_diagnosis_issue_counts'].get(category,0)} | {a['first_diagnosis_category_phase_counts'].get(category,0)} |")
    lines += ['', '## Fixed-pair breakdown by setting', '',
              '| Setting | Pairs | Executable before → after | Exact before → after | Phase score before → after | Source score before → after |',
              '|---|---:|---:|---:|---:|---:|']
    for setting, b in summary['first_attempt_by_setting'].items():
        z=b['counts']; lines.append(f"| {setting} | {z['pairs']} | {z['before_executable']} → {z['after_executable']} | {z['before_exact_packets']} → {z['after_exact_packets']} | {b['phase_macro_before']:.3f} → {b['phase_macro_after']:.3f} | {b['source_macro_before']:.3f} → {b['source_macro_after']:.3f} |")
    lines += ['', '## Audit', '',
              'All model/solver results are pre-existing files. The analysis reads private scoring labels only offline. It verifies that every repair payload contains exactly the public task and the same observations as its formalization input, scans that payload for private answer-field names, and checks the final round against the saved after-result. The runtime evidence_sha256 hashes compiled facts, not retrieved observations, and can change when a record is corrected. No run files are written and no model/solver calls are made.', '',
              f"There are {len(later)} additional Guard-attempt pairs, reported separately in summary.json. The root method/ copy in guarded phases is deliberately ignored because it is an overwritten convenience copy of later attempts."]
    (OUT/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:summary[k] for k in ('observed_first_pairs','unavailable_first_pairs','first_attempt','audit')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
