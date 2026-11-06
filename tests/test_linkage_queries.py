import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import build_file_inventory as inventory
import build_proposal_spine as spine


class LinkageQueryTests(unittest.TestCase):
    def test_orphans_and_all_versions_remain_visible(self):
        sql=inventory.query_for('KUALI','AWARD_ATTACHMENT',{'FILE_ID','AWARD_NUMBER','SEQUENCE_NUMBER'},'FILE_ID')
        self.assertIn('LEFT JOIN KUALI.ATTACHMENT_FILE',sql)
        self.assertIn('LEFT JOIN KUALI.FILE_DATA',sql)
        self.assertNotIn('WHERE',sql)
        self.assertIn('af.FILE_DATA IS NOT NULL',sql)
        self.assertIn('af.FILE_ID AS ATTACHMENT_FILE_ID',sql)

    def test_inline_bytes_not_selected(self):
        sql=inventory.query_for('KUALI','BUDGET_SUB_AWARD_ATT',{'ATTACHMENT'},inline='ATTACHMENT')
        self.assertIn('DBMS_LOB.GETLENGTH(a.ATTACHMENT)',sql)
        self.assertNotIn('a.ATTACHMENT AS',sql)

    def test_institutional_identifier_not_mislabelled_development(self):
        sql=inventory.query_for('KUALI','PROPOSAL_ATTACHMENTS',{'FILE_DATA_ID','PROPOSAL_NUMBER'},'FILE_DATA_ID')
        self.assertIn('NULL AS DEV_PROPOSAL_NUMBER',sql)
        self.assertIn('a.PROPOSAL_NUMBER AS INST_PROPOSAL_NUMBER',sql)

    def test_nonresearch_blobs_and_migration_copies_not_auto_downloaded(self):
        for name in ['PERSON_SIGNATURE','KREW_ATT_T','CHG35532_FILE_DATA','KC_QRTZ_JOB_DETAILS']:
            self.assertNotEqual(inventory.disposition(name),'inventory_candidate_no_content_download')

    def test_owner_approved_labels_do_not_turn_unknown_into_negative(self):
        self.assertEqual(spine.STATUS_TO_OUTCOME['2'],'FUNDED')
        self.assertEqual(spine.STATUS_TO_OUTCOME['3'],'NOT_FUNDED')
        self.assertEqual(spine.STATUS_TO_OUTCOME['8'],'VOID')
        self.assertEqual(spine.STATUS_TO_OUTCOME.get(None,'UNKNOWN'),'UNKNOWN')


if __name__=='__main__':unittest.main()
