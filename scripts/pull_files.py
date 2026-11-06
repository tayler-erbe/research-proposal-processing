"""Bounded local pilot from the reconciled inventory; never follows external file locations."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import discover_kuali_schema as discovery
from linkage_formats import detect,extract

# Scope-limited first pilot. Negotiation, personnel/bios, special review,
# workflow attachments and email content require separate scope review.
PILOT_TABLES={'NARRATIVE_ATTACHMENT','PROPOSAL_ATTACHMENTS','AWARD_ATTACHMENT',
    'BUDGET_SUB_AWARDS','BUDGET_SUB_AWARD_ATT','SUBAWARD_ATTACHMENTS',
    'SUBAWARD_AMOUNT_INFO','SUBAWARD_TEMPLATE_ATTACHMENTS','S2S_APP_ATTACHMENTS',
    'S2S_OVERRIDE_ATT','S2S_USER_ATTACHED_FORM_ATT','S2S_USER_ATTACHED_FORM_FILE',
    'S2S_USER_ATTD_FORM_ATT_FILE'}


def locator(row):
    loc=row['PAYLOAD_LOCATION']
    if loc=='FILE_DATA':return 'FILE_DATA:'+row['FILE_DATA_ID']
    if loc=='ATTACHMENT_FILE_INLINE' and row.get('ATTACHMENT_FILE_ID'):
        return 'ATTACHMENT_FILE:'+row['ATTACHMENT_FILE_ID']
    return ':'.join([loc,row['SOURCE_TABLE'],row['INLINE_BLOB_COLUMN'] or '',row['ATTACHMENT_ROWID']])


def main():
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=discovery.PROJECT_ROOT/'configs/database.yaml')
    ap.add_argument('--owner',type=discovery.schema_owner,default='KUALI')
    ap.add_argument('--oracle-client-lib');ap.add_argument('--call-timeout-ms',type=int,default=30000)
    ap.add_argument('--limit',type=int,default=200);ap.add_argument('--max-mib',type=int,default=500)
    args=ap.parse_args()
    if not 1<=args.limit<=200 or not 1<=args.max_mib<=1024:ap.error('Pilot is capped at 200 files and 1024 MiB; larger pulls need separate review')
    if not 1000<=args.call_timeout_ms<=600000:ap.error('Invalid timeout')
    root=discovery.PROJECT_ROOT/'storage/linkage';folder=root/'files';folder.mkdir(exist_ok=True)
    inventory=pd.read_parquet(root/'file_inventory.parquet')
    eligible=inventory[inventory.SOURCE_TABLE.isin(PILOT_TABLES)&inventory.BLOB_LENGTH.between(1,20*1024*1024)].copy()
    eligible['PAYLOAD_KEY']=eligible.apply(locator,axis=1)
    eligible=eligible.sort_values(['SOURCE_TABLE','CONTENT_TYPE','ATTACHMENT_ROWID'],na_position='last').drop_duplicates('PAYLOAD_KEY')
    eligible['STRATUM']=eligible.SOURCE_TABLE.astype(str)+'|'+eligible.CONTENT_TYPE.fillna('unknown')
    eligible['PICK_ORDER']=eligible.groupby('STRATUM').cumcount()
    sample=eligible.sort_values(['PICK_ORDER','STRATUM']).head(args.limit)
    manifest_path=root/'pilot_manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    previous={r['PAYLOAD_KEY']:r for r in manifest}
    selected=[];budget=0
    with discovery.discovery_connection(discovery.load_database(args.config),args) as conn:
        with conn.cursor() as cur:
            for row in sample.to_dict('records'):
                key=row['PAYLOAD_KEY'];prior=previous.get(key)
                if prior and prior.get('STATUS')=='DOWNLOADED':
                    p=folder/prior['SAVED_NAME']
                    if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==prior['SHA256']:
                        selected.append(prior);continue
                entry={'PAYLOAD_KEY':key,'FILE_DATA_ID':row['FILE_DATA_ID'],'STATUS':'PENDING'}
                if budget+int(row['BLOB_LENGTH'])>args.max_mib*1024*1024:
                    entry['STATUS']='PILOT_BYTE_BUDGET';selected.append(entry);continue
                try:
                    if row['PAYLOAD_LOCATION']=='FILE_DATA':
                        sql=f'SELECT DATA FROM {args.owner}.FILE_DATA WHERE ID=:id';binds={'id':row['FILE_DATA_ID']}
                    elif row['PAYLOAD_LOCATION']=='ATTACHMENT_FILE_INLINE':
                        table=discovery.schema_owner(row['SOURCE_TABLE'])
                        sql=f'SELECT af.FILE_DATA FROM {args.owner}.{table} a LEFT JOIN {args.owner}.ATTACHMENT_FILE af ON af.FILE_ID=a.FILE_ID WHERE a.ROWID=CHARTOROWID(:rid)';binds={'rid':row['ATTACHMENT_ROWID']}
                    else:
                        table=discovery.schema_owner(row['SOURCE_TABLE']);column=discovery.schema_owner(row['INLINE_BLOB_COLUMN'])
                        sql=f'SELECT {column} FROM {args.owner}.{table} WHERE ROWID=CHARTOROWID(:rid)';binds={'rid':row['ATTACHMENT_ROWID']}
                    cur.execute(sql,binds);found=cur.fetchone()
                    if not found or found[0] is None:raise LookupError('payload missing')
                    blob=found[0]
                    if hasattr(blob,'size') and blob.size()>20*1024*1024:raise ValueError('file grew beyond pilot limit')
                    data=blob.read() if hasattr(blob,'read') else blob
                    if len(data)!=int(row['BLOB_LENGTH']):raise ValueError('source size changed')
                    sha=hashlib.sha256(data).hexdigest();kind=detect(data);name=sha+'.'+kind
                    target=folder/name
                    if not target.exists():
                        temp=target.with_suffix(target.suffix+'.partial');temp.write_bytes(data);temp.replace(target)
                    entry.update(STATUS='DOWNLOADED',SHA256=sha,BYTES=len(data),FORMAT=kind,SAVED_NAME=name)
                    budget+=len(data)
                except Exception as exc:
                    entry['STATUS']='FAILED';entry['ERROR_TYPE']=type(exc).__name__
                selected.append(entry)
                manifest_path.write_text(json.dumps(selected,indent=2,default=str)+'\n',encoding='utf-8')
                print(json.dumps({'files_attempted':len(selected),'downloaded':sum(r['STATUS']=='DOWNLOADED' for r in selected),'bytes_downloaded_this_run':budget}),flush=True)
    manifest_path.write_text(json.dumps(selected,indent=2,default=str)+'\n',encoding='utf-8')
    pq.write_table(pa.Table.from_pylist(selected),root/'pull_manifest.parquet',compression='zstd')
    texts=[]
    for row in selected:
        if row['STATUS']=='DOWNLOADED':
            texts.append({'PAYLOAD_KEY':row['PAYLOAD_KEY'],**extract(folder/row['SAVED_NAME'],row['FORMAT'])})
    if texts:pq.write_table(pa.Table.from_pylist(texts),root/'document_text.parquet',compression='zstd')
    summary={'pilot':True,'selected_files':len(sample),'downloaded':sum(r['STATUS']=='DOWNLOADED' for r in selected),
        'downloaded_bytes_this_run':budget,'formats':dict(pd.Series([r.get('FORMAT','not_downloaded') for r in selected]).value_counts()),
        'text_statuses':dict(pd.Series([r['EXTRACT_STATUS'] for r in texts]).value_counts()),
        'scope_note':'Pilot only; excludes files over 20 MiB and scope-held sources. OLE/RTF/office spreadsheets and presentations retained with explicit unsupported extraction status. No OCR.'}
    (root/'pilot_summary.json').write_text(json.dumps(summary,indent=2,default=int)+'\n',encoding='utf-8')
    print(json.dumps(summary,default=int))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('Pilot stopped: '+type(exc).__name__,file=sys.stderr);sys.exit(1)
