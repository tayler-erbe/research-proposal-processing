"""Reconcile the downloaded paired cohort against all scoped attachment references."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import discover_kuali_schema as discovery


def main():
    import pandas as pd
    root=discovery.PROJECT_ROOT/'storage/linkage'
    cases=pd.read_parquet(root/'paired_sample_cases.parquet')
    spine=pd.read_parquet(root/'proposal_spine.parquet')
    links=pd.read_parquet(root/'proposal_award_bridge.parquet')
    inv=pd.read_parquet(root/'file_inventory.parquet')
    manifest=pd.read_parquet(root/'paired_pull_manifest.parquet')
    selected=set(cases.DEV_PROPOSAL_NUMBER)
    n=inv[inv.SOURCE_TABLE.eq('NARRATIVE_ATTACHMENT')&inv.DEV_PROPOSAL_NUMBER.isin(selected)].copy()
    n['CASE_DEV_PROPOSAL_NUMBER']=n.DEV_PROPOSAL_NUMBER
    i=inv[inv.SOURCE_TABLE.eq('PROPOSAL_ATTACHMENTS')].merge(
        spine[spine.DEV_PROPOSAL_NUMBER.isin(selected)][['DEV_PROPOSAL_NUMBER','INST_PROPOSAL_NUMBER']].rename(columns={'DEV_PROPOSAL_NUMBER':'CASE_DEV_PROPOSAL_NUMBER'}),on='INST_PROPOSAL_NUMBER',how='inner')
    pairs=links[links.DEV_PROPOSAL_NUMBER.isin(selected)&links.FUNDING_LINK_ACTIVE.eq('Y')&links.CURRENT_AWARD_ID.notna()][['DEV_PROPOSAL_NUMBER','AWARD_NUMBER']].drop_duplicates()
    a=inv[inv.SOURCE_TABLE.eq('AWARD_ATTACHMENT')].merge(pairs.rename(columns={'DEV_PROPOSAL_NUMBER':'CASE_DEV_PROPOSAL_NUMBER'}),on='AWARD_NUMBER',how='inner')
    refs=pd.concat([n,i,a],ignore_index=True)
    if 'PAYLOAD_KEY' not in manifest:manifest['PAYLOAD_KEY']='FILE_DATA:'+manifest.FILE_DATA_ID
    refs['PAYLOAD_KEY']='FILE_DATA:'+refs.FILE_DATA_ID
    if (root/'legacy_payload_locators.parquet').exists():
        locations=pd.read_parquet(root/'legacy_payload_locators.parquet')
        refs=refs.merge(locations[['SOURCE_TABLE','ATTACHMENT_ROWID','ATTACHMENT_FILE_ID']].rename(columns={'ATTACHMENT_FILE_ID':'RESOLVED_ATTACHMENT_FILE_ID'}),on=['SOURCE_TABLE','ATTACHMENT_ROWID'],how='left',validate='many_to_one')
        if 'ATTACHMENT_FILE_ID' not in refs:refs['ATTACHMENT_FILE_ID']=refs.RESOLVED_ATTACHMENT_FILE_ID
        else:refs['ATTACHMENT_FILE_ID']=refs.ATTACHMENT_FILE_ID.fillna(refs.RESOLVED_ATTACHMENT_FILE_ID)
        inline=refs.PAYLOAD_LOCATION.eq('ATTACHMENT_FILE_INLINE')
        refs.loc[inline,'PAYLOAD_KEY']='ATTACHMENT_FILE:'+refs.loc[inline,'ATTACHMENT_FILE_ID']
        if (root/'legacy_inline_pull_manifest.parquet').exists():
            supplement=pd.read_parquet(root/'legacy_inline_pull_manifest.parquet')
            manifest=pd.concat([manifest,supplement[~supplement.PAYLOAD_KEY.isin(manifest.PAYLOAD_KEY)]],ignore_index=True)
    if 'ATTACHMENT_FILE_ID' in refs:
        inline=refs.PAYLOAD_LOCATION.eq('ATTACHMENT_FILE_INLINE')
        refs.loc[inline,'PAYLOAD_KEY']='ATTACHMENT_FILE:'+refs.loc[inline,'ATTACHMENT_FILE_ID']
    good=set(manifest[manifest.STATUS.eq('DOWNLOADED')].PAYLOAD_KEY)
    records=[]
    for case in cases.to_dict('records'):
        rows=refs[refs.CASE_DEV_PROPOSAL_NUMBER.eq(case['DEV_PROPOSAL_NUMBER'])]
        nonempty=rows.BLOB_LENGTH.gt(0)
        downloaded=rows.PAYLOAD_KEY.isin(good)&nonempty
        proposal=rows.SOURCE_TABLE.isin(['NARRATIVE_ATTACHMENT','PROPOSAL_ATTACHMENTS'])
        award=rows.SOURCE_TABLE.eq('AWARD_ATTACHMENT')
        records.append({**case,'SCOPED_ATTACHMENT_REFERENCE_ROWS':len(rows),'NONEMPTY_REFERENCE_ROWS':int(nonempty.sum()),
            'MISSING_OR_EMPTY_REFERENCE_ROWS':int((~nonempty).sum()),'DOWNLOADED_REFERENCE_ROWS':int(downloaded.sum()),
            'HAS_DOWNLOADED_PROPOSAL_DOCUMENTS':bool((proposal&downloaded).any()),'HAS_DOWNLOADED_AWARD_DOCUMENTS':bool((award&downloaded).any()),
            'ALL_SCOPED_AVAILABLE_PAYLOADS_DOWNLOADED':bool((~nonempty|downloaded).all())})
    coverage=pd.DataFrame(records);coverage.to_parquet(root/'paired_case_coverage.parquet',index=False)
    if manifest.PAYLOAD_KEY.duplicated().any():raise RuntimeError('Duplicate manifest payload identities')
    complete=refs.merge(manifest.rename(columns={'FILE_DATA_ID':'DOWNLOADED_FILE_DATA_ID','ATTACHMENT_FILE_ID':'DOWNLOADED_ATTACHMENT_FILE_ID'}),on='PAYLOAD_KEY',how='left',validate='many_to_one')
    complete=complete.merge(cases.rename(columns={'DEV_PROPOSAL_NUMBER':'CASE_DEV_PROPOSAL_NUMBER'}),on='CASE_DEV_PROPOSAL_NUMBER',how='left',validate='many_to_one')
    complete.to_parquet(root/'paired_documents_complete.parquet',index=False)
    funded=coverage[coverage.COHORT.eq('PAIRED_FUNDED')]
    summary={'funded_cases':len(funded),'control_cases':int(coverage.COHORT.eq('NOT_FUNDED_CONTROL').sum()),
        'funded_cases_with_downloaded_proposal_and_award_docs':int((funded.HAS_DOWNLOADED_PROPOSAL_DOCUMENTS&funded.HAS_DOWNLOADED_AWARD_DOCUMENTS).sum()),
        'cases_with_all_scoped_available_payloads_downloaded':int(coverage.ALL_SCOPED_AVAILABLE_PAYLOADS_DOWNLOADED.sum()),
        'cases_with_missing_or_empty_source_references':int(coverage.MISSING_OR_EMPTY_REFERENCE_ROWS.gt(0).sum()),
        'missing_or_empty_source_references':int(coverage.MISSING_OR_EMPTY_REFERENCE_ROWS.sum()),
        'source_scope':['NARRATIVE_ATTACHMENT','PROPOSAL_ATTACHMENTS','AWARD_ATTACHMENT'],
        'scope_limit':'All attachment versions/types in these sources for sampled cases; no claim about other source families, undetectable source gaps, a single-SCN snapshot or UI validation.'}
    (root/'paired_case_coverage_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__':
    try:main()
    except Exception as exc:print('Coverage check stopped: '+type(exc).__name__,file=sys.stderr);sys.exit(1)
