"""Format-specific parsing into canonical spans with explicit read states.

PDF spans carry the physical page and a PDF-point bounding box. DOCX spans carry
paragraph/table-cell identity, not page numbers: Word files have no stable pages
until rendered. Pages with no text layer are recorded as needs_ocr, so a field
search over them can never be reported as absence.
"""
import io
import zipfile

from .store import put_span

MAX_ZIP_EXPANSION = 100 * 1024 * 1024


def detect(data):
    """Identify bytes by signature; attachment names and MIME types are not trusted."""
    if data.startswith(b'%PDF-'):
        return 'pdf'
    if data.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'):
        return 'ole'
    if data.lstrip().startswith(b'{\\rtf'):
        return 'rtf'
    if data.startswith(b'PK\x03\x04'):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                names = set(archive.namelist())
        except zipfile.BadZipFile:
            return 'invalid_zip'
        for member, kind in [('word/document.xml', 'docx'), ('xl/workbook.xml', 'xlsx'),
                             ('ppt/presentation.xml', 'pptx')]:
            if member in names:
                return kind
        return 'zip'
    try:
        text = data.decode('utf-8-sig')
        if text and sum(ord(c) < 32 and c not in '\r\n\t' for c in text) / len(text) < .01:
            return 'txt'
    except UnicodeDecodeError:
        pass
    return 'bin'


def parse_pdf(db, source, path):
    import pymupdf
    with pymupdf.open(path) as pdf:
        if pdf.needs_pass:
            return 'quarantined', 'encrypted_pdf', 0
        seq = 0
        for number, page in enumerate(pdf, 1):
            blocks = [b for b in page.get_text('blocks', sort=True) if len(b) > 6 and b[6] == 0 and b[4].strip()]
            chars = sum(len(b[4]) for b in blocks)
            # A page with images but no text layer is unread until OCR succeeds.
            state = 'native' if chars >= 20 else ('needs_ocr' if page.get_images() or not chars else 'native_sparse')
            db.execute('INSERT OR REPLACE INTO pages VALUES(?,?,?,?)', (source, number, state, chars))
            for b in blocks:
                seq += 1
                put_span(db, source, number, seq, 'text',
                         {'bbox': [round(v, 2) for v in b[:4]], 'units': 'pdf_points',
                          'page_size': [page.rect.width, page.rect.height]},
                         b[4].strip(), 'pdf_native')
        return 'parsed', 'PDF_TEXT_LAYER', len(pdf)


def parse_docx(db, source, path):
    with zipfile.ZipFile(path) as archive:
        if sum(i.file_size for i in archive.infolist()) > MAX_ZIP_EXPANSION:
            return 'quarantined', 'zip_expansion_limit', 0
    from docx import Document
    from docx.text.paragraph import Paragraph
    document = Document(path)
    seq = 0
    paragraph = 0
    table = 0
    for block in document.iter_inner_content():
        if isinstance(block, Paragraph):
            paragraph += 1
            if block.text.strip():
                seq += 1
                put_span(db, source, None, seq, 'text', {'paragraph': paragraph, 'style': block.style.name},
                         block.text.strip(), 'docx_native')
        else:
            table += 1
            for r, row in enumerate(block.rows, 1):
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    seq += 1
                    # One span per row keeps header/value association readable for the model.
                    put_span(db, source, None, seq, 'table_row', {'table': table, 'row': r},
                             ' | '.join(cells), 'docx_native')
    # Word files have no physical pages; record one logical unit so page accounting is explicit.
    db.execute('INSERT OR REPLACE INTO pages VALUES(?,?,?,?)', (source, 0, 'native_unpaginated', seq))
    return 'parsed', 'DOCX_BODY_PARAGRAPHS_AND_TABLES_NO_HEADERS_FOOTNOTES_IMAGES', None


def parse_txt(db, source, path):
    text = path.read_text(encoding='utf-8-sig')
    seq = 0
    for paragraph in [p.strip() for p in text.split('\n\n')]:
        if paragraph:
            seq += 1
            put_span(db, source, None, seq, 'text', {'paragraph': seq}, paragraph, 'utf8_text')
    db.execute('INSERT OR REPLACE INTO pages VALUES(?,?,?,?)', (source, 0, 'native_unpaginated', seq))
    return 'parsed', 'UTF8_TEXT', None


HANDLERS = {'pdf': parse_pdf, 'docx': parse_docx, 'txt': parse_txt}


def parse(db, source, path, kind):
    """Return (state, scope_or_reason, pages). Unsupported formats stay visible as quarantined."""
    handler = HANDLERS.get(kind)
    if handler is None:
        # Legacy .doc/.xls (OLE), RTF, spreadsheets and archives need an approved converter.
        return 'quarantined', 'unsupported_format:' + kind, None
    try:
        return handler(db, source, path)
    except Exception as exc:
        return 'quarantined', 'parse_failed:' + type(exc).__name__, None
