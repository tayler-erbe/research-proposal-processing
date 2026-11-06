"""Resolve legacy inline attachment bytes discovered by paired-case reconciliation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import discover_kuali_schema as discovery
from linkage_formats import detect


def main():
    import pandas as pd
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=discovery.PROJECT_ROOT/'configs/database.yaml')
    ap.add_argument('--owner',type=discovery.schema_owner,default='KUALI')
    ap.add_argument('--oracle-client-lib');ap.add_argument('--call-timeout-ms',type=int,default=30000)
    args=ap.parse_args();root=discovery.PROJECT_ROOT/'storage/linkage'
    planned=pd.read_parquet(root/'legacy_inline_references_to_resolve.parquet')
    if not planned.SOURCE_TABLE.eq('AWARD_ATTACHMENT').all():raise RuntimeError('Unexpected inline source')
    locators=[];results=[]
    with discovery.discovery_connection(discovery.load_database(args.config),args) as conn:
        with conn.cursor() as cur:
            rows=planned.ATTACHMENT_ROWID.drop_duplicates().tolist()
            for start in range(0,len(rows),20):
                bind={f'r{i}':v for i,v in enumerate(rows[start:start+20])}
                values=','.join('CHARTOROWID(:'+k+')' for k in bind)
                cur.execute(f'''SELECT ROWIDTOCHAR(a.ROWID),af.FILE_ID,DBMS_LOB.GETLENGTH(af.FILE_DATA)
                    FROM {args.owner}.AWARD_ATTACHMENT a JOIN {args.owner}.ATTACHMENT_FILE af ON af.FILE_ID=a.FILE_ID
                    WHERE a.ROWID IN ({values})''',bind)
                locators.extend({'SOURCE_TABLE':'AWARD_ATTACHMENT','ATTACHMENT_ROWID':rid,'ATTACHMENT_FILE_ID':fid,'INLINE_BYTES':int(length) if length is not None else None} for rid,fid,length in cur)
            locations=pd.DataFrame(locators)
            if len(locations)!=len(rows) or locations.ATTACHMENT_ROWID.duplicated().any():raise RuntimeError('Inline locator reconciliation failed')
            joined=planned.merge(locations,on=['SOURCE_TABLE','ATTACHMENT_ROWID'],validate='many_to_one')
            if not joined.BLOB_LENGTH.eq(joined.INLINE_BYTES).all():raise RuntimeError('Inline source sizes changed')
            locations.to_parquet(root/'legacy_payload_locators.parquet',index=False)
            unique=locations[['ATTACHMENT_FILE_ID','INLINE_BYTES']].drop_duplicates()
            if unique.ATTACHMENT_FILE_ID.duplicated().any() or unique.INLINE_BYTES.sum()>1024**3:raise RuntimeError('Inline supplement needs review')
            for record in unique.to_dict('records'):
                key=record['ATTACHMENT_FILE_ID'];expected=int(record['INLINE_BYTES'])
                item={'PAYLOAD_KEY':'ATTACHMENT_FILE:'+key,'ATTACHMENT_FILE_ID':key,'STATUS':'PENDING'}
                try:
                    if expected>250*1024**2:raise ValueError('File above cap')
                    cur.execute(f'SELECT FILE_DATA FROM {args.owner}.ATTACHMENT_FILE WHERE FILE_ID=:id AND DBMS_LOB.GETLENGTH(FILE_DATA)=:n',{'id':key,'n':expected})
                    row=cur.fetchone()
                    if not row or row[0] is None:raise LookupError('Missing inline payload')
                    blob=row[0];data=blob.read() if hasattr(blob,'read') else blob
                    if len(data)!=expected:raise ValueError('Inline size changed')
                    sha=hashlib.sha256(data).hexdigest();kind=detect(data);name=sha+'.'+kind
                    target=root/'files'/name
                    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest()!=sha:
                        temp=target.with_suffix(target.suffix+'.partial');temp.write_bytes(data);temp.replace(target)
                    item.update(STATUS='DOWNLOADED',BYTES=len(data),FORMAT=kind,SHA256=sha,SAVED_NAME=name)
                except Exception as exc:item.update(STATUS='FAILED',ERROR_TYPE=type(exc).__name__)
                results.append(item)
                pd.DataFrame(results).to_parquet(root/'legacy_inline_pull_manifest.parquet',index=False)
    summary={'legacy_attachment_references':len(planned),'unique_inline_attachment_files':len(results),
        'downloaded_files':sum(r['STATUS']=='DOWNLOADED' for r in results),'downloaded_bytes':sum(r.get('BYTES',0) for r in results),
        'all_inline_payloads_downloaded':all(r['STATUS']=='DOWNLOADED' for r in results)}
    (root/'legacy_inline_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        codes=re.findall(r'(?:ORA|DPY|DPI)-\d+',str(exc))
        print('Inline supplement stopped: '+(', '.join(sorted(set(codes))) or type(exc).__name__),file=sys.stderr);sys.exit(1)
