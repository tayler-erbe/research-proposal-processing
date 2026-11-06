"""Count-only join validation after schema discovery; never downloads records/BLOBs."""
import argparse
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import discover_kuali_schema as discovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=discovery.PROJECT_ROOT/'configs/database.yaml')
    parser.add_argument('--owner', type=discovery.schema_owner, default='KUALI')
    parser.add_argument('--oracle-client-lib')
    parser.add_argument('--call-timeout-ms', type=int, default=30000)
    args = parser.parse_args()
    if not 1000 <= args.call_timeout_ms <= 600000:
        parser.error('call timeout must be between 1000 and 600000 ms')
    db = discovery.load_database(args.config)
    output = discovery.PROJECT_ROOT/'storage/discovery/join_validation.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with discovery.discovery_connection(db, args) as connection:
        with connection.cursor() as cursor:
            for label, sql in discovery.build_funnel(args.owner) + discovery.build_shape_checks(args.owner):
                start = time.monotonic()
                try:
                    headers, rows = discovery.run(cursor, sql)
                    result = {'check':label, 'status':'ok','columns':headers,'rows':rows}
                except Exception as exc:
                    result = {'check':label,'status':'failed','error_codes':sorted(set(re.findall(r'(?:ORA|DPY|DPI)-\d+',str(exc)))) or [type(exc).__name__]}
                result['seconds'] = round(time.monotonic()-start, 2)
                results.append(result)
                output.write_text(json.dumps(results, indent=2, default=str)+'\n', encoding='utf-8')
                print(json.dumps(result, default=str), flush=True)
    return 0 if all(r['status']=='ok' for r in results) else 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('Validation stopped: '+type(exc).__name__, file=sys.stderr)
        sys.exit(1)
