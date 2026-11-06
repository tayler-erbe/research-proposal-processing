"""Exercise discovery safeguards without a database or driver import."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'scripts/discover_kuali_schema.py'
spec = importlib.util.spec_from_file_location('discovery', path)
discovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(discovery)


class Cursor:
    description = [('COUNT',)]
    def __init__(self, rows):
        self.rows, self.executions = rows, []
    def execute(self, sql, binds):
        self.executions.append((sql, binds))
    def fetchmany(self, limit):
        return self.rows[:limit]


class DiscoveryTests(unittest.TestCase):
    def test_owner_rejects_sql_injection(self):
        self.assertEqual(discovery.schema_owner('kuali'), 'KUALI')
        for value in ['KUALI;DELETE', 'KUALI.TABLE', '"KUALI"', 'KUALI --']:
            with self.assertRaises(Exception):
                discovery.schema_owner(value)

    def test_non_select_never_reaches_driver(self):
        cur = Cursor([])
        with self.assertRaises(ValueError):
            discovery.run(cur, 'DELETE FROM anything')
        self.assertEqual(cur.executions, [])

    def test_large_metadata_is_explicit_failure(self):
        with self.assertRaises(ValueError):
            discovery.run(Cursor([(1,)] * 10001), 'SELECT 1 FROM dual')

    def test_file_discovery_covers_unexpected_table_names(self):
        self.assertIn("data_type = 'BLOB'", discovery.FILE_BEARING_SQL)
        self.assertIn("'FILE_ID'", discovery.FILE_BEARING_SQL)
        self.assertNotIn('table_name LIKE', discovery.FILE_BEARING_SQL)

    def test_errors_do_not_expose_driver_message(self):
        report = discovery.Report()
        def fail():
            raise RuntimeError('ORA-00942 endpoint=PRIVATE_HOST password=DO_NOT_PRINT')
        discovery.safe(report, 'lookup', fail)
        text = '\n'.join(report.lines)
        self.assertIn('ORA-00942', text)
        self.assertNotIn('PRIVATE_HOST', text)
        self.assertNotIn('DO_NOT_PRINT', text)

    def test_funnel_and_shape_sql_only_aggregate_data(self):
        for _, sql in discovery.build_funnel('KUALI') + discovery.build_shape_checks('KUALI'):
            self.assertTrue(sql.lstrip().upper().startswith('SELECT'))
            self.assertIn('COUNT(', sql.upper())


if __name__ == '__main__':
    unittest.main()
