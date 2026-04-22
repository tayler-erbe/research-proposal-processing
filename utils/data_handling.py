# Lightweight GUID normalizer.
#
# A more complete version lives in services/db_formatting_service.py
# that handles the "unhyphenated 32-char hex" case by reinserting the
# dashes. Use that one when writing to Oracle. This simpler version
# is kept for ad hoc reconciliation scripts where lowercase+strip is
# enough.

import pandas as pd


def normalize_guid(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()
