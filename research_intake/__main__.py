"""Command line for the research intake prototype.

  python -m research_intake --root PRIVATE_DIR ingest-paired --linkage storage/linkage
  python -m research_intake --root PRIVATE_DIR ingest --manifest manifest.json
  python -m research_intake --root PRIVATE_DIR classify [--type-map codes.json]
  python -m research_intake --root PRIVATE_DIR plan --stage page_ocr|extract [--backend stub] [--model TAG]
  python -m research_intake --root PRIVATE_DIR run --stage page_ocr|extract [--backend stub] [--limit N]
  python -m research_intake --root PRIVATE_DIR export
  python -m research_intake --root PRIVATE_DIR search "green energy" [--family proposal]
  python -m research_intake --root PRIVATE_DIR review --source ID --run RUN_KEY --field F --decision accepted|rejected|corrected --reviewer NAME [--value V] [--reason R]
  python -m research_intake --root PRIVATE_DIR status | pause | resume | retry --job ID
  python -m research_intake synthetic --out DIR [--cases N]
  python -m research_intake --root PRIVATE_DIR evaluate-synthetic --truth DIR/ground_truth.json
"""
import argparse
import json
from pathlib import Path
import sys
import time


def main(argv=None):
    p = argparse.ArgumentParser(prog='research_intake', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, help='private runtime directory, outside the repository')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('ingest-paired').add_argument('--linkage', type=Path, required=True)
    sub.add_parser('ingest').add_argument('--manifest', type=Path, required=True)
    sub.add_parser('classify').add_argument('--type-map', type=Path)
    for name in ('plan', 'run'):
        a = sub.add_parser(name)
        a.add_argument('--stage', choices=['page_ocr', 'extract'], required=True)
        a.add_argument('--backend', choices=['stub', 'ollama'])
        a.add_argument('--model', help='override the routed model, e.g. an installed Ollama tag')
        if name == 'run':
            a.add_argument('--limit', type=int)
    for name in ('status', 'pause', 'resume', 'export'):
        sub.add_parser(name)
    a = sub.add_parser('search')
    a.add_argument('query')
    a.add_argument('--family', choices=['proposal', 'award'])
    a.add_argument('--limit', type=int, default=20)
    sub.add_parser('retry').add_argument('--job', required=True)
    a = sub.add_parser('review')
    for flag in ('--source', '--run', '--field', '--reviewer'):
        a.add_argument(flag, required=True)
    a.add_argument('--decision', choices=['accepted', 'rejected', 'corrected'], required=True)
    a.add_argument('--value')
    a.add_argument('--reason')
    a = sub.add_parser('synthetic')
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--cases', type=int, default=6)
    sub.add_parser('evaluate-synthetic').add_argument('--truth', type=Path, required=True)
    args = p.parse_args(argv)

    if args.command == 'synthetic':
        from .store import REPO
        out = args.out.resolve()
        if out == REPO or REPO in out.parents:
            sys.exit('Write synthetic documents outside the repository')
        from .synthetic import generate
        print(json.dumps({'manifest': str(generate(out, args.cases))}))
        return
    if not args.root:
        sys.exit('--root is required')
    from .store import connect, status
    db = connect(args.root)
    if args.command == 'ingest-paired':
        from .ingest import ingest_paired
        print(json.dumps({'sources': len(set(ingest_paired(db, args.linkage)))}))
    elif args.command == 'ingest':
        from .ingest import ingest_manifest, load_manifest
        print(json.dumps({'sources': len(set(ingest_manifest(db, load_manifest(args.manifest))))}))
    elif args.command == 'classify':
        from .classify import classify
        code_map = json.loads(args.type_map.read_text(encoding='utf8')) if args.type_map else None
        print(json.dumps({'classified': classify(db, code_map)}))
    elif args.command == 'plan':
        from .pipeline import plan
        print(json.dumps({'documents_scheduled': plan(db, args.stage, args.backend, args.model)}))
    elif args.command == 'run':
        from .pipeline import run
        print(json.dumps({'jobs_finished': run(db, args.stage, args.backend, args.limit, args.model)}))
    elif args.command in ('pause', 'resume'):
        with db:
            db.execute('INSERT OR REPLACE INTO controls VALUES(?,?)', ('pause', '1' if args.command == 'pause' else '0'))
    elif args.command == 'retry':
        with db:
            n = db.execute("UPDATE jobs SET state='queued' WHERE id=? AND state IN ('failed','interrupted') AND attempts<3",
                           (args.job,)).rowcount
        print(json.dumps({'requeued': n}))
    elif args.command == 'export':
        from .export import export
        print(json.dumps(export(db, args.root)))
    elif args.command == 'search':
        # Literal full-text search over source spans; results are saved locally, not printed.
        sql = ('SELECT s.source,s.page,snippet(search,3,"[","]","…",20) AS excerpt,c.family,c.doc_type FROM search s '
               'LEFT JOIN classifications c ON c.source=s.source WHERE search MATCH ?')
        params = ['"' + args.query.replace('"', '""') + '"']
        if args.family:
            sql += ' AND c.family=?'
            params.append(args.family)
        rows = [dict(r) for r in db.execute(sql + ' LIMIT ?', params + [max(1, min(args.limit, 200))])]
        (args.root / 'search-results.json').write_text(json.dumps(rows, indent=2), encoding='utf8')
        print(json.dumps({'matches': len(rows), 'saved_to': str(args.root / 'search-results.json')}))
        return
    elif args.command == 'review':
        with db:
            db.execute('INSERT INTO reviews VALUES(?,?,?,?,?,?,?)', (args.source + ':' + args.run, args.field, args.reviewer,
                                                                    args.decision, args.value, args.reason, time.time()))
    elif args.command == 'evaluate-synthetic':
        from .synthetic import evaluate
        print(json.dumps(evaluate(db, args.truth), indent=2))
        return
    print(json.dumps(status(db)))


if __name__ == '__main__':
    main()
