"""Read-only attachment metadata union. No BLOB bytes, personal names or document text."""
import argparse
import datetime
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import discover_kuali_schema as discovery

SYSTEM = ('KC_QRTZ_', 'KRSB_QRTZ_', 'KRNS_', 'CX_HRAPI_')
BACKUP = re.compile(r'^(?:CHG\d|V\d)|_V\d|_\d{8}$')
HOLD = {'PERSON_SIGNATURE', 'PERSON_BIOSKETCH', 'EXCON_PROJECT_EMAIL_CONTENT', 'KREW_ATT_T'}
COMMON = {
    'ATTACHMENT_FILE_ID': (),
    'FILE_NAME': ('FILE_NAME','SUB_AWARD_XFD_FILE_NAME'),
    'CONTENT_TYPE': ('CONTENT_TYPE','MIME_TYPE'),
    'DEV_PROPOSAL_NUMBER': ('PROPOSAL_NUMBER',),
    'INST_PROPOSAL_NUMBER': (),
    'AWARD_NUMBER': ('AWARD_NUMBER',),
    'SUBAWARD_CODE': ('SUBAWARD_CODE',),
    'SEQUENCE_NUMBER': ('SEQUENCE_NUMBER',),
    'ATTACHMENT_TYPE_CODE': ('ATTACHMENT_TYPE_CODE','TYPE_CODE','NARRATIVE_TYPE_CODE','DOCUMENT_TYPE_CODE'),
    'UPLOAD_TIMESTAMP': ('UPLOAD_TIMESTAMP',),
    'UPDATE_TIMESTAMP': ('UPDATE_TIMESTAMP',),
}
FIELDS = ['SOURCE_TABLE','ATTACHMENT_ROWID','FILE_REFERENCE_COLUMN','FILE_DATA_ID','INLINE_BLOB_COLUMN',
          'ATTACHMENT_FILE_ID','FILE_NAME','CONTENT_TYPE','DEV_PROPOSAL_NUMBER','INST_PROPOSAL_NUMBER','AWARD_NUMBER','SUBAWARD_CODE',
          'SEQUENCE_NUMBER','ATTACHMENT_TYPE_CODE','UPLOAD_TIMESTAMP','UPDATE_TIMESTAMP',
          'PAYLOAD_LOCATION','BLOB_LENGTH','HAS_INLINE_BLOB','FILE_DATA_ROW_EXISTS','HAS_EXTERNAL_LOCATION']


def disposition(table):
    if BACKUP.search(table):
        return 'excluded_backup_or_migration_copy'
    if table.startswith(SYSTEM):
        return 'excluded_application_infrastructure'
    if table in HOLD:
        return 'hold_scope_and_personal_content_review'
    if table in {'FILE_DATA','ATTACHMENT_FILE'}:
        return 'shared_payload_store_reconciled_through_attachments'
    return 'inventory_candidate_no_content_download'


def quote(name):
    return discovery.schema_owner(name)


def query_for(owner, table, columns, reference=None, inline=None):
    """One row per attachment / payload reference; LEFT joins retain orphans."""
    owner,table=quote(owner),quote(table)
    cols=set(columns)
    via_file = reference == 'FILE_ID'
    joins = ''
    file_id = 'a.'+quote(reference) if reference else 'NULL'
    fallback = None
    if via_file:
        joins += f' LEFT JOIN {owner}.ATTACHMENT_FILE af ON af.FILE_ID = a.FILE_ID'
        file_id='af.FILE_DATA_ID'
        fallback='af.FILE_DATA'
    if reference:
        joins += f' LEFT JOIN {owner}.FILE_DATA fd ON fd.ID = {file_id}'
    available = []
    if reference:available.append(('FILE_DATA','fd.DATA'))
    if fallback:available.append(('ATTACHMENT_FILE_INLINE',fallback))
    if inline:available.append(('SOURCE_INLINE','a.'+quote(inline)))
    location = 'CASE '+''.join(f"WHEN {expr} IS NOT NULL THEN '{label}' " for label,expr in available)+"ELSE 'MISSING' END"
    length = 'CASE '+''.join(f'WHEN {expr} IS NOT NULL THEN DBMS_LOB.GETLENGTH({expr}) ' for _,expr in available)+'ELSE NULL END'
    selected = [f"'{table}' AS SOURCE_TABLE", 'ROWIDTOCHAR(a.ROWID) AS ATTACHMENT_ROWID',
                (f"'{reference}'" if reference else 'NULL')+' AS FILE_REFERENCE_COLUMN',
                file_id+' AS FILE_DATA_ID', (f"'{inline}'" if inline else 'NULL')+' AS INLINE_BLOB_COLUMN']
    for alias,candidates in COMMON.items():
        column=next((c for c in candidates if c in cols),None)
        if alias=='DEV_PROPOSAL_NUMBER' and table=='PROPOSAL_ATTACHMENTS':column=None
        if alias=='INST_PROPOSAL_NUMBER' and table=='PROPOSAL_ATTACHMENTS':column='PROPOSAL_NUMBER'
        expr=('a.'+column) if column else 'NULL'
        if via_file and alias in {'FILE_NAME','CONTENT_TYPE'}:expr='af.'+alias
        if via_file and alias=='ATTACHMENT_FILE_ID':expr='af.FILE_ID'
        selected.append(expr+' AS '+alias)
    selected += [location+' AS PAYLOAD_LOCATION',length+' AS BLOB_LENGTH',
        ('CASE WHEN '+(fallback or ('a.'+quote(inline) if inline else 'NULL'))+' IS NOT NULL THEN 1 ELSE 0 END')+' AS HAS_INLINE_BLOB',
        ('CASE WHEN fd.ID IS NOT NULL THEN 1 ELSE 0 END' if reference else '0')+' AS FILE_DATA_ROW_EXISTS',
        ('CASE WHEN fd.LOCATION IS NOT NULL THEN 1 ELSE 0 END' if reference else '0')+' AS HAS_EXTERNAL_LOCATION']
    return 'SELECT '+', '.join(selected)+f' FROM {owner}.{table} a'+joins


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=discovery.PROJECT_ROOT/'configs/database.yaml')
    ap.add_argument('--owner',type=discovery.schema_owner,default='KUALI')
    ap.add_argument('--oracle-client-lib')
    ap.add_argument('--call-timeout-ms',type=int,default=30000)
    args=ap.parse_args()
    if not 1000<=args.call_timeout_ms<=600000:ap.error('invalid call timeout')
    root=discovery.PROJECT_ROOT/'storage/linkage'
    root.mkdir(parents=True,exist_ok=True)
    out=root/'file_inventory.parquet'
    if out.exists():raise RuntimeError('Existing inventory preserved; choose a separate checkout/run before rebuilding')
    schema=pa.schema([(name,pa.int64() if name in {'BLOB_LENGTH','HAS_INLINE_BLOB','FILE_DATA_ROW_EXISTS','HAS_EXTERNAL_LOCATION'} else pa.string()) for name in FIELDS])
    summary={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'consistency':'read-committed across queries; not a single database SCN snapshot','sources':[],'excluded':[]}
    temp=out.with_suffix('.parquet.partial')
    writer=pq.ParquetWriter(temp,schema,compression='zstd')
    try:
        with discovery.discovery_connection(discovery.load_database(args.config),args) as conn:
            with conn.cursor() as cur:
                cur.arraysize=2000
                cur.execute(discovery.FILE_BEARING_SQL,{'owner':args.owner})
                candidate_names=sorted({row[0] for row in cur})
                print(json.dumps({'file_bearing_tables':len(candidate_names),'phase':'metadata_only'}),flush=True)
                binds={'o':args.owner,**{f't{i}':t for i,t in enumerate(candidate_names)}}
                placeholders=','.join(':'+f't{i}' for i in range(len(candidate_names)))
                cur.execute(f'SELECT table_name,column_name,data_type FROM all_tab_columns WHERE owner=:o AND table_name IN ({placeholders}) ORDER BY table_name,column_id',binds)
                tables={}
                for table,column,typ in cur:tables.setdefault(table,{})[column]=typ
                candidates={t:c for t,c in tables.items() if 'FILE_DATA_ID' in c or 'FILE_ID' in c or 'BLOB' in c.values()}
                for table,columns in sorted(candidates.items()):
                    state=disposition(table)
                    if state!='inventory_candidate_no_content_download':
                        summary['excluded'].append({'table':table,'disposition':state});continue
                    refs=[c for c in ('FILE_DATA_ID','XML_DATA_ID') if c in columns]
                    if not refs and 'FILE_ID' in columns:refs=['FILE_ID']
                    inlines=[c for c,t in columns.items() if t=='BLOB']
                    variants=[(r,None) for r in refs]+[(None,c) for c in inlines]
                    # A table with multiple payloads intentionally has one row per payload reference.
                    for ref,inline in variants:
                        sql=query_for(args.owner,table,columns,ref,inline)
                        label=table+':'+(ref or inline)
                        start=time.monotonic()
                        print(json.dumps({'source':label,'phase':'inventory_metadata'}),flush=True)
                        cur.execute(f'SELECT COUNT(*) FROM {args.owner}.{quote(table)}')
                        expected=int(cur.fetchone()[0])
                        cur.execute(sql)
                        count=0;total_bytes=0;missing=0;external=0
                        while True:
                            rows=cur.fetchmany(2000)
                            if not rows:break
                            records=[]
                            for row in rows:
                                item={}
                                for field,value in zip(FIELDS,row):
                                    item[field]=int(value) if value is not None and pa.types.is_integer(schema.field(field).type) else None if value is None else value.isoformat() if hasattr(value,'isoformat') else str(value)
                                records.append(item)
                                total_bytes+=item['BLOB_LENGTH'] or 0
                                missing+=item['PAYLOAD_LOCATION']=='MISSING'
                                external+=item['HAS_EXTERNAL_LOCATION'] or 0
                            writer.write_table(pa.Table.from_pylist(records,schema=schema))
                            count+=len(rows)
                        result={'source':label,'source_rows':expected,'inventory_rows':count,'counts_match':expected==count,'referenced_bytes_not_deduplicated':total_bytes,'missing_payload_rows':missing,'external_location_rows':external,'seconds':round(time.monotonic()-start,2)}
                        summary['sources'].append(result)
                        (root/'inventory_reconciliation.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
                        print(json.dumps(result),flush=True)
        writer.close()
        summary['complete']=all(s['counts_match'] for s in summary['sources'])
        summary['rows']=sum(s['inventory_rows'] for s in summary['sources'])
        summary['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        (root/'inventory_reconciliation.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
        if not summary['complete']:raise RuntimeError('Source/join count mismatch; partial inventory retained for diagnosis')
        temp.replace(out)
        print(json.dumps({'inventory_rows':summary['rows'],'source_variants':len(summary['sources']),'binary_bytes_downloaded':0}))
    finally:
        writer.close()


if __name__=='__main__':
    try:main()
    except Exception as exc:
        codes=re.findall(r'(?:ORA|DPY|DPI)-\d+',str(exc))
        print('Inventory stopped: '+(', '.join(sorted(set(codes))) or type(exc).__name__),file=sys.stderr)
        sys.exit(1)
