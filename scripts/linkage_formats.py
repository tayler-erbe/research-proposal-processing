"""Detect file bytes without trusting attachment names; never runs macros or converters."""
import io
import zipfile


def detect(data):
    if data.startswith(b'%PDF-'):return 'pdf'
    if data.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'):return 'ole'
    if data.lstrip().startswith(b'{\\rtf'):return 'rtf'
    if data.startswith(b'PK\x03\x04'):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                names=set(archive.namelist())
                if 'word/document.xml' in names:return 'docx'
                if 'xl/workbook.xml' in names:return 'xlsx'
                if 'ppt/presentation.xml' in names:return 'pptx'
                return 'zip'
        except zipfile.BadZipFile:return 'invalid_zip'
    try:
        text=data.decode('utf-8-sig')
        if text and sum(ord(c)<32 and c not in '\r\n\t' for c in text)/len(text)<.01:return 'txt'
    except UnicodeDecodeError:pass
    return 'bin'


def extract(path,kind):
    """Return explicit status; unsupported formats and scans remain in the manifest."""
    result={'FORMAT':kind,'N_PAGES':None,'TEXT':'','EXTRACT_STATUS':'UNSUPPORTED_FORMAT',
            'LENGTH':0,'ALPHA_RATIO':0.0,'SUCCESSFULLY_PARSED':False,'ERROR_TYPE':None,
            'EXTRACTION_SCOPE':'NONE'}
    try:
        if kind=='pdf':
            from pypdf import PdfReader
            reader=PdfReader(path)
            if reader.is_encrypted:
                result['EXTRACT_STATUS']='ENCRYPTED';return result
            result['N_PAGES']=len(reader.pages)
            pages=[page.extract_text() or '' for page in reader.pages]
            result['TEXT']='\n\n'.join(pages)
            result['EXTRACTION_SCOPE']='PDF_TEXT_LAYER_NO_OCR'
            result['EXTRACT_STATUS']='EXTRACTED' if result['TEXT'].strip() else 'NEEDS_OCR_REVIEW'
        elif kind=='docx':
            from docx import Document
            # Bound total uncompressed ZIP size before parsing any document XML.
            with zipfile.ZipFile(path) as archive:
                if sum(i.file_size for i in archive.infolist())>100*1024*1024:
                    result['EXTRACT_STATUS']='ZIP_EXPANSION_LIMIT';return result
            document=Document(path)
            from docx.text.paragraph import Paragraph
            blocks=[]
            for block in document.iter_inner_content():
                if isinstance(block,Paragraph):blocks.append(block.text)
                else:blocks.extend('\t'.join(c.text for c in row.cells) for row in block.rows)
            result['TEXT']='\n'.join(blocks)
            result['EXTRACTION_SCOPE']='DOCX_BODY_PARAGRAPHS_AND_TABLES_NO_IMAGES_HEADERS_FOOTNOTES'
            result['EXTRACT_STATUS']='EXTRACTED'
        elif kind=='txt':
            result['TEXT']=path.read_text(encoding='utf-8-sig');result['EXTRACT_STATUS']='EXTRACTED'
            result['EXTRACTION_SCOPE']='UTF8_TEXT'
    except Exception as exc:
        result['EXTRACT_STATUS']='EXTRACTION_FAILED';result['ERROR_TYPE']=type(exc).__name__
    text=result['TEXT'];result['LENGTH']=len(text)
    result['ALPHA_RATIO']=sum(c.isalpha() for c in text)/len(text) if text else 0.0
    result['SUCCESSFULLY_PARSED']=len(text)>=150 and result['ALPHA_RATIO']>=.6
    if result['EXTRACT_STATUS']=='EXTRACTED' and not result['SUCCESSFULLY_PARSED']:
        result['EXTRACT_STATUS']='LOW_QUALITY'
    return result
