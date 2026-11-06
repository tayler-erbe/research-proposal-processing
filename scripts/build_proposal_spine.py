"""Local proposal/outcome context and many-to-many award links; SELECT only."""
import argparse
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import discover_kuali_schema as discovery

# Confirmed by owner against the live institutional-proposal lookup.
STATUS_TO_OUTCOME={'1':'PENDING','2':'FUNDED','3':'NOT_FUNDED','5':'WITHDRAWN','8':'VOID'}


def fetch(cur,sql):
    if not sql.lstrip().upper().startswith('SELECT'):raise ValueError('SELECT required')
    cur.execute(sql)
    columns=[d[0] for d in cur.description]
    rows=[]
    while True:
        batch=cur.fetchmany(5000)
        if not batch:break
        rows.extend({k:None if v is None else v.isoformat() if hasattr(v,'isoformat') else str(v) for k,v in zip(columns,row)} for row in batch)
    return rows


def main():
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=discovery.PROJECT_ROOT/'configs/database.yaml')
    ap.add_argument('--owner',type=discovery.schema_owner,default='KUALI')
    ap.add_argument('--oracle-client-lib')
    ap.add_argument('--call-timeout-ms',type=int,default=30000)
    args=ap.parse_args()
    if not 1000<=args.call_timeout_ms<=600000:ap.error('invalid timeout')
    o=args.owner
    root=discovery.PROJECT_ROOT/'storage/linkage';root.mkdir(parents=True,exist_ok=True)
    if (root/'proposal_spine.parquet').exists():raise RuntimeError('Existing spine preserved')
    with discovery.discovery_connection(discovery.load_database(args.config),args) as conn:
        with conn.cursor() as cur:
            cur.arraysize=2000
            # A current row must be unique; never hide conflicting current versions with ROW_NUMBER.
            for table,key,state in [('PROPOSAL','PROPOSAL_NUMBER','PROPOSAL_SEQUENCE_STATUS'),('AWARD','AWARD_NUMBER','AWARD_SEQUENCE_STATUS')]:
                cur.execute(f"SELECT COUNT(*) FROM (SELECT {key} FROM {o}.{table} WHERE {state}='ACTIVE' GROUP BY {key} HAVING COUNT(*)>1)")
                if cur.fetchone()[0]:raise RuntimeError('Ambiguous current versions: '+table)
            spine=fetch(cur,f'''SELECT ep.PROPOSAL_NUMBER AS DEV_PROPOSAL_NUMBER,
                ep.TITLE AS DEV_TITLE, ep.STATUS_CODE AS DEV_STATUS_CODE,
                ep.CREATE_TIMESTAMP AS DEV_CREATE_TIMESTAMP, ep.UPDATE_TIMESTAMP AS DEV_UPDATE_TIMESTAMP,
                ep.OWNED_BY_UNIT AS LEAD_UNIT_NUMBER, u.UNIT_NAME AS LEAD_UNIT_NAME,
                u.PARENT_UNIT_NUMBER, ep.SPONSOR_CODE, s.SPONSOR_NAME, s.SPONSOR_TYPE_CODE,
                ep.PRIME_SPONSOR_CODE, ep.ACTIVITY_TYPE_CODE, ep.PROPOSAL_TYPE_CODE,
                ep.REQUESTED_START_DATE_INITIAL, ep.REQUESTED_END_DATE_INITIAL,
                ep.DEADLINE_DATE, pad.DATE_SUBMITTED_TO_AGENCY,
                pad.INST_PROPOSAL_ID AS LINKED_INST_VERSION_ID,
                linked.PROPOSAL_NUMBER AS INST_PROPOSAL_NUMBER,
                ip.PROPOSAL_ID AS CURRENT_INST_PROPOSAL_ID, ip.SEQUENCE_NUMBER AS INST_SEQUENCE_NUMBER,
                ip.STATUS_CODE AS INST_STATUS_CODE, ps.DESCRIPTION AS INST_STATUS_DESCRIPTION,
                ip.TOTAL_DIRECT_COST_TOTAL AS REQUESTED_DIRECT_TOTAL,
                ip.TOTAL_INDIRECT_COST_TOTAL AS REQUESTED_INDIRECT_TOTAL
                FROM {o}.EPS_PROPOSAL ep
                LEFT JOIN {o}.PROPOSAL_ADMIN_DETAILS pad ON pad.DEV_PROPOSAL_NUMBER=ep.PROPOSAL_NUMBER
                LEFT JOIN {o}.PROPOSAL linked ON linked.PROPOSAL_ID=pad.INST_PROPOSAL_ID
                LEFT JOIN {o}.PROPOSAL ip ON ip.PROPOSAL_NUMBER=linked.PROPOSAL_NUMBER AND ip.PROPOSAL_SEQUENCE_STATUS='ACTIVE'
                LEFT JOIN {o}.PROPOSAL_STATUS ps ON ps.PROPOSAL_STATUS_CODE=ip.STATUS_CODE
                LEFT JOIN {o}.SPONSOR s ON s.SPONSOR_CODE=ep.SPONSOR_CODE
                LEFT JOIN {o}.UNIT u ON u.UNIT_NUMBER=ep.OWNED_BY_UNIT''')
            cur.execute(f'SELECT COUNT(*) FROM {o}.EPS_PROPOSAL');expected=int(cur.fetchone()[0])
            if len(spine)!=expected or len({r['DEV_PROPOSAL_NUMBER'] for r in spine})!=expected:
                raise RuntimeError('Proposal join expanded or dropped source rows')
            bridges=fetch(cur,f'''SELECT DISTINCT pad.DEV_PROPOSAL_NUMBER,
                ip.PROPOSAL_NUMBER AS INST_PROPOSAL_NUMBER, ipv.PROPOSAL_ID AS FUNDED_INST_VERSION_ID,
                afp.AWARD_FUNDING_PROPOSAL_ID, afp.ACTIVE AS FUNDING_LINK_ACTIVE,
                a.AWARD_ID AS LINKED_AWARD_VERSION_ID, a.AWARD_NUMBER,
                current_a.AWARD_ID AS CURRENT_AWARD_ID
                FROM {o}.PROPOSAL_ADMIN_DETAILS pad
                JOIN {o}.PROPOSAL ip ON ip.PROPOSAL_ID=pad.INST_PROPOSAL_ID
                JOIN {o}.PROPOSAL ipv ON ipv.PROPOSAL_NUMBER=ip.PROPOSAL_NUMBER
                JOIN {o}.AWARD_FUNDING_PROPOSALS afp ON afp.PROPOSAL_ID=ipv.PROPOSAL_ID
                JOIN {o}.AWARD a ON a.AWARD_ID=afp.AWARD_ID
                LEFT JOIN {o}.AWARD current_a ON current_a.AWARD_NUMBER=a.AWARD_NUMBER AND current_a.AWARD_SEQUENCE_STATUS='ACTIVE' ''')
            awards=fetch(cur,f'''SELECT a.AWARD_NUMBER, a.AWARD_ID AS CURRENT_AWARD_ID,
                a.SEQUENCE_NUMBER, a.STATUS_CODE, st.DESCRIPTION AS STATUS_DESCRIPTION,
                a.TITLE, a.SPONSOR_CODE, a.PRIME_SPONSOR_CODE, a.LEAD_UNIT_NUMBER,
                a.BEGIN_DATE, a.AWARD_EFFECTIVE_DATE, a.AWARD_EXECUTION_DATE, a.CLOSEOUT_DATE
                FROM {o}.AWARD a LEFT JOIN {o}.AWARD_STATUS st ON st.STATUS_CODE=a.STATUS_CODE
                WHERE a.AWARD_SEQUENCE_STATUS='ACTIVE' ''')
            units=fetch(cur,f'SELECT UNIT_NUMBER, UNIT_NAME, PARENT_UNIT_NUMBER, ACTIVE_FLAG FROM {o}.UNIT')
    # Preserve raw link provenance; current/no-current award state does not erase historical links.
    by_dev={}
    for row in bridges:by_dev.setdefault(row['DEV_PROPOSAL_NUMBER'],set()).add(row['AWARD_NUMBER'])
    for row in spine:
        row['OUTCOME']=STATUS_TO_OUTCOME.get(row['INST_STATUS_CODE'],'UNKNOWN')
        row['N_AWARDS']=len(by_dev.get(row['DEV_PROPOSAL_NUMBER'],set()))
        row['OUTCOME_LINK_REVIEW_REQUIRED']=(row['OUTCOME']=='NOT_FUNDED' and row['N_AWARDS']>0) or (row['OUTCOME']=='FUNDED' and row['N_AWARDS']==0)
        row['BINARY_OUTCOME_ELIGIBLE']=row['OUTCOME'] in {'FUNDED','NOT_FUNDED'} and not row['OUTCOME_LINK_REVIEW_REQUIRED']
    for name,records in [('proposal_spine',spine),('proposal_award_bridge',bridges),('award_spine',awards),('unit_hierarchy',units)]:
        table=pa.Table.from_pylist(records)
        target=root/(name+'.parquet');temp=target.with_suffix('.parquet.partial')
        pq.write_table(table,temp,compression='zstd');temp.replace(target)
    counts=dict(pd.Series([r['OUTCOME'] for r in spine]).value_counts().items())
    summary={'dev_proposals':len(spine),'current_awards':len(awards),'bridge_provenance_rows':len(bridges),
        'distinct_dev_award_pairs':len({(r['DEV_PROPOSAL_NUMBER'],r['AWARD_NUMBER']) for r in bridges}),
        'outcomes':{str(k):int(v) for k,v in counts.items()},
        'outcome_link_review_required':sum(r['OUTCOME_LINK_REVIEW_REQUIRED'] for r in spine),
        'personal_names_or_emails_extracted':False,
        'money_note':'Requested amounts are raw strings; awarded totals and allocation are not inferred or summed from the versioned transaction ledger.',
        'scope_note':'Current award spine excludes award numbers with no ACTIVE version; bridge retains historical funding link references, including their ACTIVE flag without filtering. Unit hierarchy is preserved without assuming a universal college level. PI identifiers not extracted.'}
    (root/'spine_reconciliation.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        codes=re.findall(r'(?:ORA|DPY|DPI)-\d+',str(exc))
        print('Spine stopped: '+(', '.join(sorted(set(codes))) or type(exc).__name__),file=sys.stderr);sys.exit(1)
