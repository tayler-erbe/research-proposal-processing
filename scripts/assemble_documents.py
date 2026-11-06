"""Streaming local document union and explicit many-to-many context bridge."""
import collections
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import discover_kuali_schema as discovery
from pull_files import locator


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    root=discovery.PROJECT_ROOT/'storage/linkage'
    spine=pq.read_table(root/'proposal_spine.parquet').to_pylist()
    by_dev={r['DEV_PROPOSAL_NUMBER']:r for r in spine}
    by_inst=collections.defaultdict(set);by_award=collections.defaultdict(set)
    for r in spine:
        if r['INST_PROPOSAL_NUMBER']:by_inst[r['INST_PROPOSAL_NUMBER']].add(r['DEV_PROPOSAL_NUMBER'])
    for r in pq.read_table(root/'proposal_award_bridge.parquet').to_pylist():
        by_award[r['AWARD_NUMBER']].add(r['DEV_PROPOSAL_NUMBER'])
    manifests={r['PAYLOAD_KEY']:r for r in json.loads((root/'pilot_manifest.json').read_text())}
    texts={r['PAYLOAD_KEY']:r for r in pq.read_table(root/'document_text.parquet').to_pylist()}
    source=pq.ParquetFile(root/'file_inventory.parquet')
    schema=pa.schema(list(source.schema_arrow)+[
        ('DOCUMENT_KEY',pa.string()),('PAYLOAD_KEY',pa.string()),('DOWNLOAD_STATUS',pa.string()),
        ('DETECTED_FORMAT',pa.string()),('EXTRACT_STATUS',pa.string()),('TEXT',pa.string()),
        ('SUCCESSFULLY_PARSED',pa.bool_()),('MATCHED_DEV_PROPOSALS_JSON',pa.string()),
        ('N_MATCHED_DEV_PROPOSALS',pa.int64()),('CONTEXT_RESOLUTION',pa.string()),
        ('OUTCOME',pa.string()),('BINARY_OUTCOME_ELIGIBLE',pa.bool_())])
    bridge_schema=pa.schema([('DOCUMENT_KEY',pa.string()),('DEV_PROPOSAL_NUMBER',pa.string()),('RELATION',pa.string())])
    partial=root/'documents.parquet.partial';bridge_partial=root/'document_proposal_bridge.parquet.partial'
    counts=collections.Counter();rows_written=0;bridge_rows=0
    with pq.ParquetWriter(partial,schema,compression='zstd') as writer, pq.ParquetWriter(bridge_partial,bridge_schema,compression='zstd') as bridge_writer:
        for batch in source.iter_batches(batch_size=10000):
            documents=[];links=[]
            for row in batch.to_pylist():
                identity='|'.join(str(row[k] or '') for k in ['SOURCE_TABLE','ATTACHMENT_ROWID','FILE_REFERENCE_COLUMN','INLINE_BLOB_COLUMN'])
                key=hashlib.sha256(identity.encode()).hexdigest();payload=locator(row)
                manifest=manifests.get(payload,{});text=texts.get(payload,{})
                dev=row['DEV_PROPOSAL_NUMBER'];inst=row['INST_PROPOSAL_NUMBER'];award=row['AWARD_NUMBER']
                if dev and dev in by_dev:matches={dev};relation='DIRECT_DEV'
                elif inst and inst in by_inst:matches=by_inst[inst];relation='INSTITUTIONAL_BRIDGE'
                elif award and award in by_award:matches=by_award[award];relation='AWARD_BRIDGE'
                else:matches=set();relation='UNRESOLVED'
                one=by_dev[next(iter(matches))] if len(matches)==1 else None
                row.update(DOCUMENT_KEY=key,PAYLOAD_KEY=payload,DOWNLOAD_STATUS=manifest.get('STATUS','NOT_SELECTED_FOR_PILOT'),
                    DETECTED_FORMAT=manifest.get('FORMAT'),EXTRACT_STATUS=text.get('EXTRACT_STATUS','NOT_EXTRACTED'),
                    TEXT=text.get('TEXT'),SUCCESSFULLY_PARSED=text.get('SUCCESSFULLY_PARSED',False),
                    MATCHED_DEV_PROPOSALS_JSON=json.dumps(sorted(matches)),N_MATCHED_DEV_PROPOSALS=len(matches),
                    CONTEXT_RESOLUTION=relation,OUTCOME=one['OUTCOME'] if one else 'MULTIPLE_CONTEXTS' if matches else 'UNKNOWN',
                    BINARY_OUTCOME_ELIGIBLE=one['BINARY_OUTCOME_ELIGIBLE'] if one else False)
                documents.append(row)
                links.extend({'DOCUMENT_KEY':key,'DEV_PROPOSAL_NUMBER':d,'RELATION':relation} for d in sorted(matches))
                counts[relation]+=1
            writer.write_table(pa.Table.from_pylist(documents,schema=schema));rows_written+=len(documents)
            if links:bridge_writer.write_table(pa.Table.from_pylist(links,schema=bridge_schema));bridge_rows+=len(links)
    if rows_written!=source.metadata.num_rows:raise RuntimeError('Document union changed inventory cardinality')
    partial.replace(root/'documents.parquet');bridge_partial.replace(root/'document_proposal_bridge.parquet')
    summary={'inventory_rows':source.metadata.num_rows,'document_rows':rows_written,'document_proposal_bridge_rows':bridge_rows,
        'context_counts':dict(counts),'pilot_only':True,'unique_pilot_payloads_with_text_attempts':len(texts),
        'limits':['Not a complete document content pull','Other source contexts still require adapters','ROWID-based document keys are run-scoped, not permanent business identifiers','Requested/awarded money is not allocated across many-to-many proposal links','Schema/source reads were not captured at one database SCN']}
    (root/'assembly_reconciliation.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__':
    try:main()
    except Exception as exc:print('Assembly stopped: '+type(exc).__name__,file=sys.stderr);sys.exit(1)
