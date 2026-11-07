"""End-to-end plumbing tests for research_intake on synthetic documents (stub backend)."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

HAVE_DEPS = all(importlib.util.find_spec(m) for m in ('pymupdf', 'docx', 'reportlab', 'pandas'))


@unittest.skipUnless(HAVE_DEPS, 'requires pymupdf, python-docx, reportlab and pandas')
class ResearchIntakeEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from research_intake.classify import classify
        from research_intake.ingest import ingest_manifest, load_manifest
        from research_intake.pipeline import plan, run
        from research_intake.store import connect
        from research_intake.synthetic import generate
        cls.tmp = Path(tempfile.mkdtemp())
        cls.manifest = generate(cls.tmp / 'docs', cases=4)
        cls.db = connect(cls.tmp / 'run')
        ingest_manifest(cls.db, load_manifest(cls.manifest))
        classify(cls.db)
        plan(cls.db, 'page_ocr', 'stub')
        run(cls.db, 'page_ocr', 'stub')
        plan(cls.db, 'extract', 'stub')
        run(cls.db, 'extract', 'stub')

    @classmethod
    def tearDownClass(cls):
        cls.db.close()
        shutil.rmtree(cls.tmp)

    def test_runtime_inside_repo_refused(self):
        from research_intake.store import REPO, connect
        with self.assertRaises(ValueError):
            connect(REPO / 'storage' / 'should_not_exist')

    def test_identical_bytes_are_one_source_with_many_associations(self):
        rows = self.db.execute('SELECT source,count(DISTINCT case_id) AS n FROM associations GROUP BY source HAVING n>1').fetchall()
        self.assertTrue(rows, 'shared boilerplate should be deduplicated by content hash')

    def test_scanned_pages_are_never_reported_as_absent(self):
        from research_intake.pipeline import consolidated
        scanned = {r['source'] for r in self.db.execute("SELECT source FROM pages WHERE state='needs_ocr'")}
        self.assertTrue(scanned)
        outcomes = {m['outcome'] for m in consolidated(self.db) if m['source'] in scanned}
        self.assertTrue(outcomes)
        self.assertFalse(outcomes & {'not_found_in_full_text'})

    def test_evidence_text_is_copied_from_source_spans(self):
        from research_intake.pipeline import consolidated
        spans = {r['id']: r['text'] for r in self.db.execute('SELECT id,text FROM spans')}
        found = [m for m in consolidated(self.db) if m['outcome'] == 'found']
        self.assertTrue(found)
        for m in found:
            for e in m['evidence']:
                self.assertEqual(e['text'], spans[e['span']])

    def test_rerun_is_idempotent(self):
        from research_intake.pipeline import plan
        before = self.db.execute('SELECT count(*) FROM jobs').fetchone()[0]
        plan(self.db, 'extract', 'stub')
        self.assertEqual(before, self.db.execute('SELECT count(*) FROM jobs').fetchone()[0])

    def test_award_facts_never_become_predictors(self):
        from research_intake.export import export
        import pandas as pd
        export(self.db, self.tmp / 'run')
        cases = pd.read_csv(self.tmp / 'run' / 'tables' / 'case_features.csv')
        award_fields = {'award_terms_flags', 'fa_terms', 'reporting_requirements', 'award_action'}
        self.assertFalse(award_fields & set(cases.columns), 'award fields must be prefixed award__')
        funded = cases[cases.cohort == 'PAIRED_FUNDED']
        self.assertTrue(funded['award__award_terms_flags'].notna().all())

    def test_synthetic_scoring_runs(self):
        from research_intake.synthetic import evaluate
        result = evaluate(self.db, self.manifest.parent / 'ground_truth.json')
        self.assertIn('SYNTHETIC', result['label'])
        self.assertEqual(result['doc_type_correct'].split('/')[0], result['doc_type_correct'].split('/')[1])

    def test_synthetic_pdf_text_matches_generator_text(self):
        import pymupdf
        text = pymupdf.open(self.manifest.parent / 'SYN-0001_award.pdf')[0].get_text()
        self.assertIn('limits F&A to this rate', text)
        self.assertNotIn('F&A;', text)


@unittest.skipUnless(HAVE_DEPS and importlib.util.find_spec('pyarrow'), 'requires pyarrow')
class PairedSampleAdapter(unittest.TestCase):
    """Mimics storage/linkage/paired_documents_complete.parquet written by verify_paired_sample.py."""

    def test_ingest_paired_keeps_kuali_relationships(self):
        import hashlib
        import pandas as pd
        from research_intake.ingest import ingest_paired
        from research_intake.store import connect
        from research_intake.synthetic import _docx
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / 'linkage' / 'files').mkdir(parents=True)
            path = tmp / 'linkage' / 'files' / 'abc.docx'
            _docx(path, ['Project Description', 'We will use a survey approach.'])
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            base = {'STATUS': 'DOWNLOADED', 'SAVED_NAME': 'abc.docx', 'SHA256': sha, 'FORMAT': 'docx',
                    'SOURCE_TABLE': 'NARRATIVE_ATTACHMENT', 'COHORT': 'PAIRED_FUNDED', 'MATCH_GROUP': 'G1'}
            pd.DataFrame([dict(base, CASE_DEV_PROPOSAL_NUMBER='101', DOCUMENT_ROLE='PROPOSAL_MATERIAL', PAYLOAD_KEY='p1',
                               INST_PROPOSAL_NUMBER='IP1', AWARD_NUMBER=None),
                          dict(base, CASE_DEV_PROPOSAL_NUMBER='102', DOCUMENT_ROLE='PROPOSAL_MATERIAL', PAYLOAD_KEY='p2',
                               INST_PROPOSAL_NUMBER=None, AWARD_NUMBER=None),
                          dict(base, STATUS='FAILED', SAVED_NAME='missing.pdf', SHA256='x', CASE_DEV_PROPOSAL_NUMBER='103',
                               DOCUMENT_ROLE='AWARD_OUTCOME_MATERIAL', PAYLOAD_KEY='p3')]
                         ).to_parquet(tmp / 'linkage' / 'paired_documents_complete.parquet')
            db = connect(tmp / 'run')
            self.assertEqual(len(set(ingest_paired(db, tmp / 'linkage'))), 1)
            rows = db.execute('SELECT case_id,predictor_eligibility FROM associations ORDER BY case_id').fetchall()
            self.assertEqual([tuple(r) for r in rows], [('101', 'eligible_pre_decision'), ('102', 'eligible_pre_decision')])
            path.write_bytes(path.read_bytes() + b' ')
            with self.assertRaises(ValueError):
                ingest_paired(db, tmp / 'linkage')   # changed bytes must not be accepted silently
            db.close()
        finally:
            shutil.rmtree(tmp)


class EvidenceValidation(unittest.TestCase):
    def setUp(self):
        from research_intake.evidence import load_schema
        self.schema = load_schema()
        self.defs = {f['id']: f for f in self.schema['fields']}
        self.aliases = {'E1': {'id': 's1', 'page': 1, 'locator': {}, 'text': 'IRB approval is required.'}}

    def check(self, field, item):
        from research_intake.evidence import validate
        return validate(self.schema, [self.defs[field]], self.aliases, {'fields': {field: item}})[field]

    def test_unknown_evidence_id_rejected(self):
        r = self.check('specific_aims', {'outcome': 'found', 'evidence': ['E9']})
        self.assertEqual(r['outcome'], 'rejected')

    def test_found_without_evidence_rejected(self):
        r = self.check('specific_aims', {'outcome': 'found', 'evidence': []})
        self.assertEqual(r['outcome'], 'rejected')

    def test_vocabulary_value_outside_list_rejected(self):
        r = self.check('compliance_flags', {'outcome': 'found', 'evidence': [],
                                            'values': [{'value': 'made_up_flag', 'evidence': ['E1']}]})
        self.assertEqual(r['outcome'], 'rejected')

    def test_not_found_with_evidence_rejected(self):
        r = self.check('specific_aims', {'outcome': 'not_found_in_excerpts', 'evidence': ['E1']})
        self.assertEqual(r['outcome'], 'rejected')

    def test_valid_vocabulary_value_copies_text(self):
        r = self.check('compliance_flags', {'outcome': 'found', 'evidence': [],
                                            'values': [{'value': 'human_subjects', 'evidence': ['E1']}]})
        self.assertEqual(r['values'][0]['evidence'][0]['text'], 'IRB approval is required.')

    def test_consolidation_distinguishes_unread_from_absent(self):
        from research_intake.evidence import consolidate
        miss = {'outcome': 'not_found_in_excerpts', 'evidence': []}
        f = self.defs['specific_aims']
        self.assertEqual(consolidate(f, [miss, miss], 2, 0)['outcome'], 'not_found_in_full_text')
        self.assertEqual(consolidate(f, [miss, miss], 2, 1)['outcome'], 'not_found_in_read_text')
        self.assertEqual(consolidate(f, [miss], 2, 0)['outcome'], 'incomplete')

    def test_conflicting_enum_values_are_ambiguous(self):
        from research_intake.evidence import consolidate
        ev = [{'span': 's1', 'page': 1, 'text': 't'}]
        a = {'outcome': 'found', 'value': 'basic', 'evidence': ev}
        b = {'outcome': 'found', 'value': 'applied', 'evidence': ev}
        r = consolidate(self.defs['translational_stage'], [a, b], 2, 0)
        self.assertEqual(r['outcome'], 'ambiguous')
        self.assertEqual(r['competing_values'], ['applied', 'basic'])


if __name__ == '__main__':
    unittest.main()
