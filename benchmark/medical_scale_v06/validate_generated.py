"""Automatic construction checks; passing is provenance-verified SILVER, not clinical gold."""
from collections import Counter

LABELS={'supported','refuted','not_determined'}


def validate(source,generated):
    if not isinstance(generated,dict) or generated.get('source_id')!=source['source_id']:
        raise ValueError('Source identity mismatch')
    variants=generated.get('variants')
    if not isinstance(variants,list) or not 1<=len(variants)<=3:
        raise ValueError('Expected one to three independently checked candidate variants')
    anchors={b['anchor_id']:b for d in source['documents'] for b in d['blocks']}
    known=set(anchors);ids=set();report=[]
    for variant in variants:
        vid=variant.get('variant_id')
        if not isinstance(vid,str) or not vid or vid in ids:raise ValueError('Variant id missing or duplicated')
        ids.add(vid)
        visibility={}
        for stage in ('initial','updated'):
            available=variant.get(stage+'_visible_anchor_ids')
            if not isinstance(available,list) or not available or any(not isinstance(x,str) for x in available) or len(available)!=len(set(available)) or not set(available)<=known:
                raise ValueError('Invalid stage visibility')
            visibility[stage]=set(available)
        if not visibility['initial']<visibility['updated'] or visibility['updated']!=known:
            raise ValueError('Reveal must add original material without hiding arbitrary evidence')
        if source['modality']=='dialogue':
            ordered=[b['anchor_id'] for d in source['documents'] for b in d['blocks']]
            cut=len(visibility['initial'])
            if cut<8 or len(ordered)-cut<8 or visibility['initial']!=set(ordered[:cut]):
                raise ValueError('Dialogue chronology must be a full prefix with at least eight turns on each side')
        claims=variant.get('claims')
        if not isinstance(claims,list) or len(claims)!=7 or len({c.get('item_id') for c in claims})!=7:
            raise ValueError('Exactly seven unique claims required')
        changed=stable=0;quotes=0;stage_counts={s:Counter() for s in ('initial','updated')}
        for claim in claims:
            if not isinstance(claim.get('item_id'),str) or not isinstance(claim.get('statement'),str) or not claim['statement'].strip():
                raise ValueError('Missing atomic record-level claim')
            for stage in ('initial','updated'):
                label=claim.get(stage+'_label');support=claim.get(stage+'_support')
                if label not in LABELS or not isinstance(support,list):raise ValueError('Invalid label or support schema')
                stage_counts[stage][label]+=1
                if label!='not_determined' and not support:raise ValueError('Definite label without a quoted witness')
                support_ids=[]
                for witness in support:
                    aid,quote=witness.get('anchor_id'),witness.get('exact_quote')
                    if aid not in visibility[stage]:raise ValueError('Future, unknown or inactive source citation')
                    if not isinstance(quote,str) or not quote or quote not in anchors[aid]['text']:
                        raise ValueError('Quote not found in original anchor')
                    support_ids.append(aid);quotes+=1
                if len(support_ids)!=len(set(support_ids)):raise ValueError('Use one quoted witness per support anchor')
            changed+=claim['initial_label']!=claim['updated_label']
            stable+=claim['initial_label']==claim['updated_label']
        if changed<2 or stable<2 or any(len(c)<2 for c in stage_counts.values()):
            raise ValueError('Insufficient change/preservation or label-class diversity')
        event=variant.get('simulated_event')
        if not isinstance(event,str) or not event.strip():raise ValueError('Missing explicitly simulated availability event')
        report.append({'variant_id':vid,'changed_claims':changed,'stable_claims':stable,'exact_quotes_checked':quotes,
            'status':'provenance_and_structure_verified_silver','semantic_label_correctness_established':False})
    return {'source_id':source['source_id'],'variants':report,'clinical_gold':False,
        'limits':'Exact quotes and source visibility do not prove entailment, atomicity, necessary multi-hop difficulty or clinical correctness. Audit these on a fixed sample in parallel.'}
