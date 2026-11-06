"""Size complete linked case bundles locally before any larger document download."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import discover_kuali_schema as discovery
from pull_files import locator


def main():
    import pandas as pd
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pairs',type=int,default=200)
    ap.add_argument('--max-gib',type=float,default=8)
    args=ap.parse_args()
    if not 1<=args.pairs<=500 or not 0<args.max_gib<=8:ap.error('Plan capped at 500 paired cases and 8 GiB; not a full pull')
    root=discovery.PROJECT_ROOT/'storage/linkage'
    inv=pd.read_parquet(root/'file_inventory.parquet')
    inline=inv.PAYLOAD_LOCATION.eq('ATTACHMENT_FILE_INLINE')&inv.SOURCE_TABLE.eq('AWARD_ATTACHMENT')
    if inline.any() and ('ATTACHMENT_FILE_ID' not in inv or inv.loc[inline,'ATTACHMENT_FILE_ID'].isna().any()):
        raise RuntimeError('Legacy inventory lacks stable attachment-file keys; use the enriched inventory format or reconcile the existing cohort with its inline supplement before replanning')
    spine=pd.read_parquet(root/'proposal_spine.parquet')
    bridge=pd.read_parquet(root/'proposal_award_bridge.parquet')
    present=inv[inv.SOURCE_TABLE.isin(['NARRATIVE_ATTACHMENT','PROPOSAL_ATTACHMENTS','AWARD_ATTACHMENT'])&inv.BLOB_LENGTH.gt(0)&inv.PAYLOAD_LOCATION.isin(['FILE_DATA','ATTACHMENT_FILE_INLINE'])].copy()
    present['PAYLOAD_KEY']=present.apply(locator,axis=1)
    # Keep all versions and types; deduplicate payload bytes, not their attachment relationships.
    narratives=present[present.SOURCE_TABLE.eq('NARRATIVE_ATTACHMENT')].copy()
    narratives['CASE_DEV_PROPOSAL_NUMBER']=narratives.DEV_PROPOSAL_NUMBER
    institutional=present[present.SOURCE_TABLE.eq('PROPOSAL_ATTACHMENTS')].merge(
        spine[['DEV_PROPOSAL_NUMBER','INST_PROPOSAL_NUMBER']].rename(columns={'DEV_PROPOSAL_NUMBER':'CASE_DEV_PROPOSAL_NUMBER'}),on='INST_PROPOSAL_NUMBER',how='inner',validate='many_to_many')
    proposals=pd.concat([narratives,institutional],ignore_index=True)
    links=bridge[bridge.FUNDING_LINK_ACTIVE.eq('Y')&bridge.CURRENT_AWARD_ID.notna()][['DEV_PROPOSAL_NUMBER','AWARD_NUMBER']].drop_duplicates()
    award_docs=present[present.SOURCE_TABLE.eq('AWARD_ATTACHMENT')].merge(
        links.rename(columns={'DEV_PROPOSAL_NUMBER':'CASE_DEV_PROPOSAL_NUMBER'}),on='AWARD_NUMBER',how='inner',validate='many_to_many')
    proposal_ids=proposals.groupby('CASE_DEV_PROPOSAL_NUMBER').PAYLOAD_KEY.agg(set).to_dict()
    award_ids=award_docs.groupby('CASE_DEV_PROPOSAL_NUMBER').PAYLOAD_KEY.agg(set).to_dict()
    sizes=present.drop_duplicates('PAYLOAD_KEY').set_index('PAYLOAD_KEY').BLOB_LENGTH.to_dict()
    s=spine[spine.BINARY_OUTCOME_ELIGIBLE].copy()
    s['YEAR']=s.DEV_CREATE_TIMESTAMP.fillna('UNKNOWN').str[:4]
    s['SPONSOR_STRATUM']=s.SPONSOR_TYPE_CODE.fillna('UNKNOWN')
    s['STRATUM']=s.YEAR+'|'+s.SPONSOR_STRATUM
    s['ORDER_HASH']=s.DEV_PROPOSAL_NUMBER.map(lambda x:hashlib.sha256(('paired-pilot-v1|'+x).encode()).hexdigest())
    funded=s[s.OUTCOME.eq('FUNDED')&s.DEV_PROPOSAL_NUMBER.isin(proposal_ids)&s.DEV_PROPOSAL_NUMBER.isin(award_ids)].sort_values(['STRATUM','ORDER_HASH'])
    funded['STRATUM_ORDER']=funded.groupby('STRATUM').cumcount()
    funded=funded.sort_values(['STRATUM_ORDER','STRATUM','ORDER_HASH'])
    controls=s[s.OUTCOME.eq('NOT_FUNDED')&s.DEV_PROPOSAL_NUMBER.isin(proposal_ids)].sort_values('ORDER_HASH')
    pools={k:list(g.DEV_PROPOSAL_NUMBER) for k,g in controls.groupby('STRATUM')}
    picked=[];payloads=set();cost=0;rejects={};used_controls=set();pairs=0
    for case in funded.to_dict('records'):
        if pairs>=args.pairs:break
        dev=case['DEV_PROPOSAL_NUMBER'];ids=proposal_ids[dev]|award_ids[dev]
        if any(sizes[x]>250*1024*1024 for x in ids):rejects['case_contains_file_over_250MiB']=rejects.get('case_contains_file_over_250MiB',0)+1;continue
        pool=pools.get(case['STRATUM'],[])
        control=next((x for x in pool if x not in used_controls and all(sizes[k]<=250*1024*1024 for k in proposal_ids[x])),None)
        combined=ids|(proposal_ids[control] if control else set())
        added=sum(int(sizes[x]) for x in combined-payloads)
        if cost+added>args.max_gib*1024**3:
            rejects['case_bundle_exceeds_remaining_byte_budget']=rejects.get('case_bundle_exceeds_remaining_byte_budget',0)+1;continue
        group=f'PAIR_{pairs+1:04d}'
        picked.append({'DEV_PROPOSAL_NUMBER':dev,'COHORT':'PAIRED_FUNDED','MATCH_GROUP':group,'STRATUM':case['STRATUM'],'MATCH_RULE':'YEAR_AND_SPONSOR_TYPE','N_PROPOSAL_FILES':len(proposal_ids[dev]),'N_AWARD_FILES':len(award_ids[dev])})
        if control:
            used_controls.add(control)
            picked.append({'DEV_PROPOSAL_NUMBER':control,'COHORT':'NOT_FUNDED_CONTROL','MATCH_GROUP':group,'STRATUM':case['STRATUM'],'MATCH_RULE':'YEAR_AND_SPONSOR_TYPE','N_PROPOSAL_FILES':len(proposal_ids[control]),'N_AWARD_FILES':0})
        payloads.update(combined);cost+=added;pairs+=1
    cases=pd.DataFrame(picked)
    selected=set(cases.DEV_PROPOSAL_NUMBER)
    refs=pd.concat([proposals[proposals.CASE_DEV_PROPOSAL_NUMBER.isin(selected)],award_docs[award_docs.CASE_DEV_PROPOSAL_NUMBER.isin(selected)]],ignore_index=True)
    refs['DOCUMENT_ROLE']=refs.SOURCE_TABLE.map({'NARRATIVE_ATTACHMENT':'PROPOSAL_MATERIAL','PROPOSAL_ATTACHMENTS':'INSTITUTIONAL_PROPOSAL_MATERIAL','AWARD_ATTACHMENT':'AWARD_OUTCOME_MATERIAL'})
    refs=refs.merge(cases,left_on='CASE_DEV_PROPOSAL_NUMBER',right_on='DEV_PROPOSAL_NUMBER',suffixes=('','_CASE'),validate='many_to_one')
    cases.to_parquet(root/'paired_sample_cases.parquet',index=False)
    refs.to_parquet(root/'paired_sample_document_references.parquet',index=False)
    payload_columns=['PAYLOAD_KEY','PAYLOAD_LOCATION','FILE_DATA_ID','BLOB_LENGTH']+(['ATTACHMENT_FILE_ID'] if 'ATTACHMENT_FILE_ID' in present else [])
    payload=present[present.PAYLOAD_KEY.isin(payloads)][payload_columns].drop_duplicates('PAYLOAD_KEY')
    payload.to_parquet(root/'paired_sample_payloads.parquet',index=False)
    report={'paired_funded_cases':pairs,'not_funded_controls':len(used_controls),'distinct_payload_ids':len(payloads),
        'payload_bytes':cost,'document_context_reference_rows':len(refs),'requested_paired_cases':args.pairs,
        'all_selected_pairs_have_proposal_and_award_metadata':all(c['N_PROPOSAL_FILES']>0 and c['N_AWARD_FILES']>0 for c in picked if c['COHORT']=='PAIRED_FUNDED'),
        'budget_gib':args.max_gib,'candidate_cases_excluded':rejects,
        'limitations':['This is a metadata plan, not completed document downloads','All narrative/institutional/award types and historical sequences retained for selected cases','Other file sources remain outside this first paired cohort','Selection is deterministic and stratified by development creation year and sponsor type; controls are not causal matches','Missing/empty payload references must be audited separately; positive files alone do not certify case completeness','Award documents and post-submission material must not leak into predictors of funding outcome','File IDs are deduplicated; identical bytes under different IDs are not deduplicated until hashing','Case IDs/filenames/content remain local and must not be committed']}
    (root/'paired_sample_plan.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':
    try:main()
    except Exception as exc:print('Paired planning stopped: '+type(exc).__name__,file=sys.stderr);sys.exit(1)
