from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import yaml  # noqa: F401
except ImportError:
    yaml = None


@unittest.skipIf(yaml is None, 'PyYAML not installed (requirements-synthetic.txt)')
class SyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from adapters import mapping
        from synthetic.generate import generate
        cls.mapping = mapping
        cls.profile = mapping.load_yaml(ROOT / 'synthetic/profiles/placeholder.yaml')
        cls.tables = generate(cls.profile, seed=3, scale=0.25)

    def test_deterministic_for_a_seed(self):
        from synthetic.generate import generate
        again = generate(self.profile, seed=3, scale=0.25)
        self.assertEqual(again, self.tables)
        self.assertNotEqual(generate(self.profile, seed=4, scale=0.25)['dev_proposal'], self.tables['dev_proposal'])

    def test_real_key_formats(self):
        for r in self.tables['inst_proposal']:
            self.assertRegex(r['inst_proposal_number'], r'^\d{8}$')
        for a in self.tables['award']:
            self.assertRegex(a['award_number'], r'^\d{6}-\d{5}$')
        for d in self.tables['document']:
            if d['file_data_id']:
                self.assertRegex(d['file_data_id'], r'^[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}$')
        for p in self.tables['dev_proposal']:
            self.assertIn(p['campus'], ('UIUC', 'UIC'))

    def test_foreign_keys_resolve(self):
        inst = {r['inst_version_id'] for r in self.tables['inst_proposal']}
        awards = {r['award_version_id'] for r in self.tables['award']}
        devs = {r['dev_proposal_number'] for r in self.tables['dev_proposal']}
        for p in self.tables['dev_proposal']:
            self.assertTrue(p['linked_inst_version_id'] is None or p['linked_inst_version_id'] in inst)
        for link in self.tables['proposal_award_link']:
            self.assertIn(link['inst_version_id'], inst)
            self.assertIn(link['award_version_id'], awards)
        for d in self.tables['document']:
            if d['source'] == 'AWARD':
                self.assertIn(d['award_version_id'], awards)
            else:
                self.assertIn(d['dev_proposal_number'], devs)

    def test_one_current_version_per_business_number(self):
        from adapters.views import spine
        spine(self.tables)  # raises on ambiguous current institutional versions
        current = {}
        for a in self.tables['award']:
            current[a['award_number']] = current.get(a['award_number'], 0) + a['is_current']
        self.assertTrue(all(v == 1 for v in current.values()))

    def test_kuali_shape_round_trips_to_identical_canonical_tables(self):
        m = self.mapping
        real = m.to_source(self.tables, 'kuali')
        self.assertIn('PROPOSAL', real)
        self.assertEqual(real['PROPOSAL'][0]['PROPOSAL_SEQUENCE_STATUS'] in ('ACTIVE', 'ARCHIVED'), True)
        back = m.derive(m.cast_all(m.from_source(real, 'kuali')))
        key = {'dev_proposal': 'dev_proposal_number', 'inst_proposal': 'inst_version_id', 'award': 'award_version_id',
               'proposal_award_link': 'funding_link_id', 'sponsor': 'sponsor_code', 'document': 'attachment_key'}
        for entity, k in key.items():
            ordered = lambda rows: sorted(rows, key=lambda r: (str(r.get('source')), str(r[k])))
            self.assertEqual(ordered(back[entity]), ordered(self.tables[entity]), entity)

    def test_csv_write_and_config_switch_load_the_same_tables(self):
        from synthetic.generate import write
        m = self.mapping
        with tempfile.TemporaryDirectory() as tmp:
            write(self.tables, tmp, {'seed': 3})
            syn = m.load_canonical({'data_source': 'synthetic', 'paths': {'synthetic': tmp}})
            kuali = m.load_canonical({'data_source': 'kuali', 'paths': {'kuali': str(Path(tmp) / 'kuali')}})
        self.assertEqual(len(syn['document']), len(self.tables['document']))
        self.assertEqual(sorted(syn), sorted(kuali))
        for entity in syn:
            self.assertEqual(sorted(map(str, syn[entity])), sorted(map(str, kuali[entity])), entity)

    def test_fidelity_passes_against_its_own_profile(self):
        from synthetic.fidelity import compare, summary
        from synthetic.generate import generate
        from synthetic.profile import profile_tables
        measured, counts = profile_tables(generate(self.profile, seed=11, scale=3))
        thresholds = self.mapping.load_yaml(ROOT / 'synthetic/fidelity_thresholds.yaml')
        result = summary(compare(self.profile, measured, thresholds, counts))
        self.assertTrue(result['ok'], result)

    def test_augmentation_is_flagged_by_fidelity(self):
        from synthetic.augment import oversample_category
        from synthetic.fidelity import compare
        from synthetic.generate import generate
        from synthetic.profile import profile_tables
        bent, _note = oversample_category(self.profile, 'dev_proposal.sponsor_type', 'INDUSTRY', 4)
        measured, counts = profile_tables(generate(bent, seed=11, scale=1))
        thresholds = self.mapping.load_yaml(ROOT / 'synthetic/fidelity_thresholds.yaml')
        failed = {c['name'] for c in compare(self.profile, measured, thresholds, counts) if c['passed'] is False}
        self.assertIn('dev_proposal.sponsor_type', failed)

    def test_disclosure_suppresses_small_cells_and_rounds(self):
        from synthetic.profile import disclose, profile_tables
        profile, counts = profile_tables(self.tables)
        profile['categorical']['toy'] = {'a': 0.5, 'b': 0.49, 'c': 0.01}
        counts['categorical']['toy'] = {'a': 50, 'b': 49, 'c': 1}
        out = disclose(profile, counts)
        self.assertNotIn('c', out['categorical']['toy'])
        # One merged cell of 1 could be recovered by subtraction, so the next smallest joins it.
        self.assertEqual(set(out['categorical']['toy']), {'a', 'SUPPRESSED'})
        self.assertEqual(out['meta']['kind'], 'real-disclosed')
        self.assertEqual(out['meta']['n_dev_proposals'] % 10, 0)

    def test_leakage_detects_copied_identifiers_and_text(self):
        from synthetic.leakage import check
        clean = check(self.tables, {'dev_proposal': [], 'inst_proposal': [], 'award': [], 'proposal_award_link': [],
                                    'sponsor': [], 'document': []})
        self.assertTrue(clean['clean'])
        leaked = check(self.tables, self.tables, ['one two three four five six seven eight nine'],
                       ['zero one two three four five six seven eight'])
        self.assertFalse(leaked['clean'])
        self.assertGreater(leaked['identifier_overlaps']['dev_proposal.dev_proposal_number'], 0)
        self.assertGreater(leaked['quasi_identifier_matches'], 0)
        self.assertEqual(leaked['text_ngram_overlaps'], 1)

    def test_build_select_is_read_only(self):
        sql = self.mapping.build_select('AWARD_FUNDING_PROPOSALS')
        self.assertTrue(sql.startswith('SELECT '))
        self.assertIn('FROM KUALI.AWARD_FUNDING_PROPOSALS', sql)


if __name__ == '__main__':
    unittest.main()
