"""Local payload-ID sizing; does not download content or query Oracle."""
import collections
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import discover_kuali_schema as discovery
from pull_files import PILOT_TABLES


def main():
    import pandas as pd
    root=discovery.PROJECT_ROOT/'storage/linkage'
    df=pd.read_parquet(root/'file_inventory.parquet',columns=['SOURCE_TABLE','PAYLOAD_LOCATION','FILE_DATA_ID','ATTACHMENT_ROWID','INLINE_BLOB_COLUMN','BLOB_LENGTH'])
    # A payload identity can be referenced by many attachments/versions/sources.
    # Different payload IDs with identical bytes remain separate until SHA256 is known.
    df['KEY']=df.SOURCE_TABLE.fillna('')+'|'+df.ATTACHMENT_ROWID.fillna('')+'|'+df.INLINE_BLOB_COLUMN.fillna('')
    mask=df.PAYLOAD_LOCATION.eq('FILE_DATA')
    df.loc[mask,'KEY']='FILE_DATA:'+df.loc[mask,'FILE_DATA_ID']
    known=df[df.BLOB_LENGTH.notna()&df.BLOB_LENGTH.gt(0)]
    inconsistent=known.groupby('KEY').BLOB_LENGTH.nunique()
    unique=known.drop_duplicates('KEY')
    sizes=[]
    for source,group in known.groupby('SOURCE_TABLE'):
        dedup=group.drop_duplicates('KEY')
        sizes.append({'source':source,'nonempty_reference_rows':len(group),'unique_payload_identities':len(dedup),'bytes_by_payload_identity':int(dedup.BLOB_LENGTH.sum())})
    priority=known[known.SOURCE_TABLE.isin({'NARRATIVE_ATTACHMENT','PROPOSAL_ATTACHMENTS'})].drop_duplicates('KEY')
    allowed=known[known.SOURCE_TABLE.isin(PILOT_TABLES)].drop_duplicates('KEY')
    usage=shutil.disk_usage(root)
    report={'attachment_reference_rows':len(df),'nonempty_reference_rows':len(known),'unique_nonempty_payload_identities':len(unique),
        'bytes_by_payload_identity':int(unique.BLOB_LENGTH.sum()),'payload_identities_with_conflicting_lengths':int((inconsistent>1).sum()),
        'priority_proposal_payloads':len(priority),'priority_proposal_bytes':int(priority.BLOB_LENGTH.sum()),
        'scope_allowed_payloads':len(allowed),'scope_allowed_bytes':int(allowed.BLOB_LENGTH.sum()),
        'disk_free_bytes_at_sizing':usage.free,'disk_total_bytes':usage.total,'sources':sizes,
        'caveats':['Deduplicated by payload identity, not content hash','Different content hashes are not yet known for the full inventory','Missing payload references excluded from nonempty sizing','A read-committed inventory is not a single SCN snapshot','Data/schema names and size alone do not authorize every candidate source','Content extraction products, caches and resume metadata require additional disk reserve']}
    (root/'pull_sizing.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
