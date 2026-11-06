from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pull_paired_sample import bounded_batches


class PairedTransferTests(unittest.TestCase):
    def test_batches_respect_both_limits_and_preserve_every_row(self):
        rows=[{'FILE_DATA_ID':str(i),'BLOB_LENGTH':3} for i in range(21)]
        batches=list(bounded_batches(rows,max_files=5,max_bytes=10))
        self.assertEqual([r for b in batches for r in b],rows)
        self.assertTrue(all(len(b)<=5 and sum(r['BLOB_LENGTH'] for r in b)<=10 for b in batches))

    def test_large_file_is_isolated_and_empty_input_is_empty(self):
        rows=[{'BLOB_LENGTH':20},{'BLOB_LENGTH':3},{'BLOB_LENGTH':3}]
        self.assertEqual([len(b) for b in bounded_batches(rows,max_bytes=10)],[1,2])
        self.assertEqual(list(bounded_batches([])),[])

    def test_shared_and_inline_payload_stores_are_separate_batches(self):
        rows=[{'BLOB_LENGTH':1,'PAYLOAD_LOCATION':'FILE_DATA'},
              {'BLOB_LENGTH':1,'PAYLOAD_LOCATION':'ATTACHMENT_FILE_INLINE'}]
        self.assertEqual([len(b) for b in bounded_batches(rows)],[1,1])


if __name__=='__main__':unittest.main()
