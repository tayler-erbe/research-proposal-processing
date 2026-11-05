# Offline demonstration

This example lets collaborators try evidence-linked document search without server, database or model access. It uses fictional proposal and award pages, the whitespace-cleaning helper and deterministic frequency keywords. Search returns a document ID, page and literal source quote. It does not reproduce the full semantic NLP pipeline, perform OCR, make compliance determinations or implement the broader planned document-intelligence system.

```sh
python demo.py --query "imaging reporting"
```

Python 3.10+ is required; the synthetic JSON path uses no third-party dependencies or network access. The example makes no database connections and writes no files. You can supply your own approved JSON with `--input`; the shape is a list of objects with unique `id` strings and nonempty `pages` lists of text strings.

For a local text PDF, create a virtual environment and install the optional dependency:

```sh
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.venv\Scripts\python.exe demo.py --pdf path/to/approved.pdf --query "reporting"
# Linux/macOS: use .venv/bin/python instead.
```

Blank pages are marked for OCR review. Encrypted PDFs are rejected. The demo prints page text to the terminal and provides no authentication; use only material approved for your environment. No live proposal, award, user-profile or internal configuration data is distributed here.

With `requirements-demo.txt` installed, run:

```sh
python tools/check_source.py
python -m unittest discover -s tests -v
```

Six local tests pass on Python 3.12, covering correct source-page evidence, no invented matches, invalid input, blank pages, synthetic PDF extraction, and running the CLI from outside the checkout. Source checks parse Python and notebook cells without executing production code. These checks do not validate live database access, schema compatibility, production retry safety, scheduling or model accuracy.

Production `main.py` and `notebooks/main.ipynb` are separate entry points. They clear working folders and perform live database writes. Database dry-run is not an offline mode. Review their effects and recovery plan against an isolated destination before any operational use.

The public example, tests and documentation are distinct from restricted deployment records and the unmodified server preservation archive. No deployment or recurring synchronization is configured by this update.
