"""Download the locally sized paired sample, not the full Kuali file population."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import discover_kuali_schema as discovery
from linkage_formats import detect


def bounded_batches(rows, max_files=20, max_bytes=32*1024*1024):
    batch=[];total=0
    for row in rows:
        size=int(row['BLOB_LENGTH'])
        if batch and (len(batch)>=max_files or total+size>max_bytes or row.get('PAYLOAD_LOCATION','FILE_DATA')!=batch[0].get('PAYLOAD_LOCATION','FILE_DATA')):
            yield batch;batch=[];total=0
        batch.append(row);total+=size
    if batch:yield batch


def main():
    import pandas as pd
    import oracledb
    # Direct byte fetch avoids a separate LOB round-trip per file. Queries below
    # require the inventoried length, bounding returned bytes per batch.
    oracledb.defaults.fetch_lobs=False
    root=discovery.PROJECT_ROOT/'storage/linkage';folder=root/'files';folder.mkdir(exist_ok=True)
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=discovery.PROJECT_ROOT/'configs/database.yaml')
    ap.add_argument('--owner',type=discovery.schema_owner,default='KUALI')
    ap.add_argument('--oracle-client-lib');ap.add_argument('--call-timeout-ms',type=int,default=30000)
    args=ap.parse_args()
    plan=json.loads((root/'paired_sample_plan.json').read_text())
    if plan['payload_bytes']>8*1024**3:raise RuntimeError('Sized sample exceeds authorized tool limit')
    if shutil.disk_usage(root).free<max(50*1024**3,2*plan['payload_bytes']):raise RuntimeError('Insufficient sample disk reserve')
    jobs=pd.read_parquet(root/'paired_sample_payloads.parquet')
    if 'PAYLOAD_LOCATION' not in jobs:jobs['PAYLOAD_LOCATION']='FILE_DATA'
    if 'PAYLOAD_KEY' not in jobs:jobs['PAYLOAD_KEY']='FILE_DATA:'+jobs.FILE_DATA_ID
    if not jobs.PAYLOAD_LOCATION.isin(['FILE_DATA','ATTACHMENT_FILE_INLINE']).all():raise RuntimeError('Unsupported sized payload location')
    if (jobs.PAYLOAD_LOCATION.eq('ATTACHMENT_FILE_INLINE')).any() and 'ATTACHMENT_FILE_ID' not in jobs:raise RuntimeError('Missing stable inline key')
    if len(jobs)!=plan['distinct_payload_ids'] or int(jobs.BLOB_LENGTH.sum())!=plan['payload_bytes']:raise RuntimeError('Sample sizing manifest changed')
    if jobs.PAYLOAD_KEY.duplicated().any():raise RuntimeError('Duplicate planned payload identities')
    log=root/'paired_pull_checkpoint.jsonl';previous={}
    if log.exists():
        for line in log.read_text(encoding='utf-8').splitlines():
            try:record=json.loads(line)
            except ValueError:continue
            previous[record.get('PAYLOAD_KEY') or 'FILE_DATA:'+record['FILE_DATA_ID']]=record
    pilot=root/'pilot_manifest.json'
    if pilot.exists():
        for r in json.loads(pilot.read_text()):
            if r.get('FILE_DATA_ID') and r.get('STATUS')=='DOWNLOADED':previous.setdefault('FILE_DATA:'+r['FILE_DATA_ID'],r)
    completed=[];pending=[]
    for row in jobs.to_dict('records'):
        prior=previous.get(row['PAYLOAD_KEY'])
        if prior and prior.get('STATUS')=='DOWNLOADED':
            path=folder/prior['SAVED_NAME']
            if path.is_file() and path.stat().st_size==int(row['BLOB_LENGTH']) and hashlib.sha256(path.read_bytes()).hexdigest()==prior['SHA256']:
                prior['PAYLOAD_KEY']=row['PAYLOAD_KEY'];completed.append(prior);continue
        pending.append(row)
    size_by_key={r['PAYLOAD_KEY']:int(r['BLOB_LENGTH']) for r in pending}
    downloaded_this_run=0
    with discovery.discovery_connection(discovery.load_database(args.config),args) as conn, log.open('a',encoding='utf-8') as journal:
        with conn.cursor() as cur:
            cur.arraysize=20
            for batch in bounded_batches(pending):
                inline=batch[0]['PAYLOAD_LOCATION']=='ATTACHMENT_FILE_INLINE'
                id_column='FILE_ID' if inline else 'ID';data_column='FILE_DATA' if inline else 'DATA'
                table='ATTACHMENT_FILE' if inline else 'FILE_DATA'
                ids=[r['ATTACHMENT_FILE_ID'] if inline else r['FILE_DATA_ID'] for r in batch]
                binds={}
                terms=[]
                for i,row in enumerate(batch):
                    binds[f'id{i}']=ids[i];binds[f'bytes{i}']=int(row['BLOB_LENGTH'])
                    terms.append(f'({id_column}=:id{i} AND DBMS_LOB.GETLENGTH({data_column})=:bytes{i})')
                try:
                    cur.execute(f'SELECT {id_column},{data_column} FROM {args.owner}.{table} WHERE '+ ' OR '.join(terms),binds)
                    found={key:blob for key,blob in cur.fetchall()}
                except Exception as exc:
                    codes=re.findall(r'(?:ORA|DPY|DPI)-\d+',str(exc))
                    for row,key in zip(batch,ids):
                        record={'FILE_DATA_ID':row.get('FILE_DATA_ID'),'PAYLOAD_KEY':row['PAYLOAD_KEY'],'STATUS':'QUERY_FAILED','ERROR_CODES':sorted(set(codes)) or [type(exc).__name__]}
                        completed.append(record);journal.write(json.dumps(record)+'\n')
                    journal.flush();continue
                for job,key in zip(batch,ids):
                    record={'FILE_DATA_ID':job.get('FILE_DATA_ID'),'PAYLOAD_KEY':job['PAYLOAD_KEY'],'ATTACHMENT_FILE_ID':job.get('ATTACHMENT_FILE_ID'),'STATUS':'PENDING'}
                    try:
                        blob=found.get(key)
                        if blob is None:raise LookupError('Missing payload')
                        expected=size_by_key[job['PAYLOAD_KEY']]
                        if expected>250*1024*1024 or (hasattr(blob,'size') and blob.size()!=expected):raise ValueError('Size changed or over limit')
                        data=blob.read() if hasattr(blob,'read') else blob
                        if len(data)!=expected:raise ValueError('Size changed')
                        sha=hashlib.sha256(data).hexdigest();kind=detect(data);name=sha+'.'+kind;path=folder/name
                        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
                            temp=path.with_suffix(path.suffix+'.partial');temp.write_bytes(data);temp.replace(path)
                        record.update(STATUS='DOWNLOADED',SHA256=sha,BYTES=len(data),FORMAT=kind,SAVED_NAME=name)
                        downloaded_this_run+=len(data)
                    except Exception as exc:
                        record['STATUS']='FAILED';record['ERROR_TYPE']=type(exc).__name__
                    completed.append(record);journal.write(json.dumps(record)+'\n');journal.flush()
                print(json.dumps({'planned_files':len(jobs),'processed_files':len(completed),'downloaded_files':sum(r['STATUS']=='DOWNLOADED' for r in completed),'downloaded_bytes_this_run':downloaded_this_run}),flush=True)
    frame=pd.DataFrame(completed);frame.to_parquet(root/'paired_pull_manifest.parquet',index=False)
    success=frame[frame.STATUS.eq('DOWNLOADED')]
    references=pd.read_parquet(root/'paired_sample_document_references.parquet')
    if 'PAYLOAD_KEY' not in references:references['PAYLOAD_KEY']='FILE_DATA:'+references.FILE_DATA_ID
    refs=references.merge(frame.rename(columns={'FILE_DATA_ID':'DOWNLOADED_FILE_DATA_ID','ATTACHMENT_FILE_ID':'DOWNLOADED_ATTACHMENT_FILE_ID'}),on='PAYLOAD_KEY',how='left',validate='many_to_one')
    refs.to_parquet(root/'paired_documents.parquet',index=False)
    cases=pd.read_parquet(root/'paired_sample_cases.parquet')
    docs=refs[refs.STATUS.eq('DOWNLOADED')]
    roles=docs.groupby(['CASE_DEV_PROPOSAL_NUMBER','DOCUMENT_ROLE']).size().unstack(fill_value=0)
    roles['HAS_PROPOSAL']=roles.get('PROPOSAL_MATERIAL',0)+roles.get('INSTITUTIONAL_PROPOSAL_MATERIAL',0)>0
    roles['HAS_AWARD']=roles.get('AWARD_OUTCOME_MATERIAL',0)>0
    funded=set(cases[cases.COHORT.eq('PAIRED_FUNDED')].DEV_PROPOSAL_NUMBER)
    complete_sides=roles.loc[roles.index.isin(funded)]
    good=set(success.PAYLOAD_KEY)
    summary={'planned_file_ids':len(jobs),'downloaded_file_ids':len(success),'failed_file_ids':len(jobs)-len(success),
        'downloaded_payload_bytes':int(success.BYTES.sum()) if len(success) else 0,'distinct_content_hashes':int(success.SHA256.nunique()) if len(success) else 0,
        'formats':{str(k):int(v) for k,v in success.FORMAT.value_counts().items()} if len(success) else {},
        'funded_cases_with_downloaded_documents_on_both_sides':int((complete_sides.HAS_PROPOSAL&complete_sides.HAS_AWARD).sum()),
        'planned_funded_cases':len(funded),'document_context_rows':len(refs),
        'all_planned_payloads_downloaded':set(jobs.PAYLOAD_KEY)<=good,
        'extraction_status':'Not extracted by this script; procurement OCR integration pending',
        'scope':'Sized paired pilot only, all in-scope attachment versions retained; not a full population pull or certification that every source document exists'}
    (root/'paired_pull_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary))
    return 0 if summary['all_planned_payloads_downloaded'] else 2


if __name__=='__main__':
    try:sys.exit(main())
    except Exception as exc:print('Paired pull stopped: '+type(exc).__name__,file=sys.stderr);sys.exit(1)
