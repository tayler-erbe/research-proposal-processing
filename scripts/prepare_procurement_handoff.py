"""Prepare local-only manifests for the existing procurement intake; no OCR or database writes."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import discover_kuali_schema as discovery


def main():
    import pandas as pd
    root=discovery.PROJECT_ROOT/'storage/linkage';out=root/'parser_handoff';out.mkdir(exist_ok=True)
    data=pd.read_parquet(root/'paired_documents_complete.parquet')
    if not data.STATUS.eq('DOWNLOADED').all():raise RuntimeError('Missing downloads in handoff')
    unique=data[['SHA256','SAVED_NAME','FORMAT','BYTES']].drop_duplicates('SHA256')
    pdfs=[];office=[];other=[]
    for row in unique.to_dict('records'):
        name=row['SAVED_NAME']
        if Path(name).name!=name:raise ValueError('Nonlocal manifest path')
        path=root/'files'/name
        if not path.is_file() or path.stat().st_size!=int(row['BYTES']):raise RuntimeError('Stored source changed')
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(4*1024*1024),b''):digest.update(block)
        if digest.hexdigest()!=row['SHA256']:raise RuntimeError('Stored source hash mismatch')
        item={'sha256':row['SHA256'],'path':str(path.resolve()),'format':row['FORMAT'],'bytes':int(row['BYTES'])}
        if row['FORMAT']=='pdf':pdfs.append(item['path'])
        elif row['FORMAT'] in {'docx','ole','xlsx','pptx'}:office.append(item)
        else:other.append(item)
    # intake_pipeline.ingest.ingest expects a JSON list of PDF filenames.
    (out/'pdf_input_manifest.json').write_text(json.dumps(pdfs,indent=2)+'\n',encoding='utf-8')
    (out/'office_conversion_queue.json').write_text(json.dumps(office,indent=2)+'\n',encoding='utf-8')
    (out/'other_format_queue.json').write_text(json.dumps(other,indent=2)+'\n',encoding='utf-8')
    context=data[['SHA256','PAYLOAD_KEY','SOURCE_TABLE','ATTACHMENT_ROWID','CASE_DEV_PROPOSAL_NUMBER','COHORT','MATCH_GROUP','AWARD_NUMBER','INST_PROPOSAL_NUMBER']].drop_duplicates()
    context.to_parquet(out/'research_document_context_bridge.parquet',index=False)
    report={'verified_distinct_payload_hashes':len(unique),'pdf_manifest_files':len(pdfs),'office_conversion_files':len(office),
        'other_format_files':len(other),'research_context_rows':len(context),'ocr_executed':False,'inference_executed':False,'procurement_database_opened':False,
        'contract':'PDF filenames are compatible with the reviewed procurement intake function. Preserve the SHA256 context bridge; no proposal-to-award relationships may be rebuilt from filenames.',
        'limits':'Office conversion and page numbering, parser quality, entitlements and per-page evidence must be validated before integration. Generating this manifest is not execution/acceptance of the procurement pipeline.'}
    (out/'handoff_summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':
    try:main()
    except Exception as exc:print('Parser handoff stopped: '+type(exc).__name__,file=sys.stderr);sys.exit(1)
