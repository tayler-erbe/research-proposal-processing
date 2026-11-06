import io
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from linkage_formats import detect,extract


class FormatTests(unittest.TestCase):
    def test_zip_container_is_not_automatically_word(self):
        for member,kind in [('word/document.xml','docx'),('xl/workbook.xml','xlsx'),('random.txt','zip')]:
            stream=io.BytesIO()
            with zipfile.ZipFile(stream,'w') as z:z.writestr(member,'sample')
            self.assertEqual(detect(stream.getvalue()),kind)

    def test_ole_is_not_automatically_doc(self):
        self.assertEqual(detect(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'+b'unknown'),'ole')
        self.assertEqual(detect(b'PK\x03\x04broken'),'invalid_zip')

    def test_unsupported_is_explicit_and_preserves_quality_fields(self):
        result=extract(Path('not-opened.ole'),'ole')
        self.assertEqual(result['EXTRACT_STATUS'],'UNSUPPORTED_FORMAT')
        self.assertFalse(result['SUCCESSFULLY_PARSED'])
        self.assertEqual(result['LENGTH'],0)

    def test_text_quality_and_missing_file_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'sample.txt';p.write_text('Research imaging reporting ' * 20,encoding='utf-8')
            self.assertTrue(extract(p,'txt')['SUCCESSFULLY_PARSED'])
            self.assertEqual(extract(p.with_name('missing.txt'),'txt')['EXTRACT_STATUS'],'EXTRACTION_FAILED')


if __name__=='__main__':unittest.main()
