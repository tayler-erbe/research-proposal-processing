# Governance filter rebuilder.
#
# Rebuilds ANALYTICS_SCHEMA.T_RSRCH_PRPSL_PRCSSD_NLP_FND_ID — the
# allowed-proposal whitelist used by the production write's final
# DELETE clause. The name "metanode" is a holdover from the original
# KNIME workflow where this logic lived inside a reusable metanode;
# ported to Python, the name stuck.
#
# Must run BEFORE write_production() in db_writer_service.py so the
# governance DELETE filter has an up-to-date allowed list. Order
# matters: write_production does a TRUNCATE + INSERT + DELETE, and
# if this hasn't run recently, the DELETE operates on yesterday's
# allowed list and incorrectly filters today's valid proposals.
#
# ALLOWED = UIUC proposals (all, regardless of funding status)
#           ∪
#           UIC proposals that are funded (STATUS_CODE = 2)
#
# The business logic here reflects a deliberate scope decision:
# Urbana-Champaign contributes all its proposal activity to the
# analytics table because the college of research interest at this
# tier is broad; UIC's proposal footprint is large enough that we
# only include the funded subset to keep the downstream analytics
# focused on proposals that actually went somewhere.
#
# Join quirk to be aware of: KUALI.PROPOSAL.PROPOSAL_NUMBER is a
# zero-padded string (e.g. '00000003') while PROPOSAL_ADMIN_DETAILS.
# DEV_PROPOSAL_NUMBER is a plain integer (e.g. 3). TO_NUMBER() on
# the left side normalizes. Without it the join silently produces
# zero rows and the metanode build looks "successful" while writing
# an empty allowlist.

import pandas as pd
from configs.config_loader      import load_config
from ingestion.oracle_connection import oracle_conn


# ── STEP 1: UIUC proposals ───────────────────────────────────────────

def _get_uiuc_proposals(kuali_conn):
    """All UIUC proposals, regardless of funding status.

    UIUC identifier = owned_by_unit starts with '1' or '9U'. These are
    the department-prefix conventions Kuali uses internally for
    Urbana-Champaign organizational units."""
    print("  [METANODE] Fetching UIUC proposals from EPS_PROPOSAL...")

    cursor = kuali_conn.cursor()
    cursor.execute("""
        SELECT DISTINCT PROPOSAL_NUMBER
        FROM KUALI.EPS_PROPOSAL
        WHERE SUBSTR(owned_by_unit, 1, 1) = '1'
           OR SUBSTR(owned_by_unit, 1, 2) = '9U'
    """)

    rows = cursor.fetchall()
    cursor.close()

    proposals = set(str(r[0]) for r in rows if r[0] is not None)
    print(f"  [METANODE] UIUC proposals: {len(proposals):,}")
    return proposals


# ── STEP 2: UIC funded proposals ─────────────────────────────────────

def _get_uic_funded_proposals(kuali_conn):
    """UIC proposals filtered to funded only.

    UIC identifier   = LEAD_UNIT_NUMBER starts with '2'
    Funded           = STATUS_CODE = 2

    See module header for the TO_NUMBER() quirk between the
    zero-padded PROPOSAL_NUMBER string and the integer
    DEV_PROPOSAL_NUMBER that we're matching against.

    DEV_PROPOSAL_NUMBER is returned rather than PROPOSAL_NUMBER because
    the downstream analytics pipeline keys on the integer form — this
    join normalizes that format across the whole dataset."""
    print("  [METANODE] Fetching funded UIC proposals...")

    cursor = kuali_conn.cursor()

    cursor.execute("""
        SELECT DISTINCT pad.DEV_PROPOSAL_NUMBER
        FROM KUALI.PROPOSAL p
        INNER JOIN KUALI.PROPOSAL_ADMIN_DETAILS pad
            ON TO_NUMBER(p.PROPOSAL_NUMBER) = pad.DEV_PROPOSAL_NUMBER
        WHERE p.STATUS_CODE = 2
          AND SUBSTR(p.LEAD_UNIT_NUMBER, 1, 1) = '2'
          AND pad.INST_PROPOSAL_ID     IS NOT NULL
          AND pad.DEV_PROPOSAL_NUMBER  IS NOT NULL
    """)

    rows = cursor.fetchall()
    cursor.close()

    proposals = set(str(r[0]) for r in rows if r[0] is not None)
    print(f"  [METANODE] Funded UIC proposals: {len(proposals):,}")
    return proposals


# ── STEP 3: write combined set to analytics DB ───────────────────────

def _write_fnd_id_table(dsstag_conn, proposal_numbers):
    """TRUNCATE + INSERT the combined allowed proposal numbers into
    ANALYTICS_SCHEMA.T_RSRCH_PRPSL_PRCSSD_NLP_FND_ID.

    Full rebuild every run. There's no incremental logic here because
    the source-of-truth sets change continuously (proposals move into
    and out of funded status) and a MERGE would complicate the
    bookkeeping without meaningful speedup — the table is small."""
    print(f"  [METANODE] Writing {len(proposal_numbers):,} proposals to FND_ID table...")

    cursor = dsstag_conn.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM ANALYTICS_SCHEMA.T_RSRCH_PRPSL_PRCSSD_NLP_FND_ID"
    )
    before = cursor.fetchone()[0]
    print(f"  [METANODE] FND_ID rows before: {before:,}")

    cursor.execute(
        "TRUNCATE TABLE ANALYTICS_SCHEMA.T_RSRCH_PRPSL_PRCSSD_NLP_FND_ID"
    )

    # Batched insert — 500 is comfortably under Oracle's 1000-bind
    # ceiling and keeps each executemany call fast enough that
    # progress is visible if we ever log inside the loop.
    batch_size    = 500
    proposal_list = list(proposal_numbers)

    for i in range(0, len(proposal_list), batch_size):
        batch = proposal_list[i : i + batch_size]
        data  = [(p,) for p in batch]
        cursor.executemany(
            "INSERT INTO ANALYTICS_SCHEMA.T_RSRCH_PRPSL_PRCSSD_NLP_FND_ID "
            "(PROPOSAL_NUMBER) VALUES (:1)",
            data,
        )

    dsstag_conn.commit()

    cursor.execute(
        "SELECT COUNT(*) FROM ANALYTICS_SCHEMA.T_RSRCH_PRPSL_PRCSSD_NLP_FND_ID"
    )
    after = cursor.fetchone()[0]
    print(f"  [METANODE] FND_ID rows after:  {after:,}")

    cursor.close()


# ── Public entry point ───────────────────────────────────────────────

def run_metanode(dry_run=False):
    """Rebuild the governance filter table from Kuali source of truth.

    Reads from Kuali (source of truth for proposal governance),
    writes to the analytics database (where the governance filter
    executes at production-write time).

    Parameters
    ----------
    dry_run : bool
        If True, fetches and reports counts but doesn't write.

    Returns
    -------
    set
        The allowed proposal numbers that would be (or were) written.
    """
    print("\n===================================")
    print(" METANODE — GOVERNANCE FILTER BUILD")
    print("===================================")

    db     = load_config("database.yaml")
    creds  = db["oracle"]["credentials"]
    dsstag = db["oracle"]["dsstag"]
    kuali  = db["oracle"]["kuali"]

    # ── Pull both proposal sets from Kuali in one connection ───────
    print("\n[METANODE] Connecting to Kuali...")
    with oracle_conn(
        kuali["host"], kuali["port"], kuali["service"],
        creds["username"], creds["password"]
    ) as kuali_conn:
        print("[METANODE] Connected.")

        uiuc_proposals       = _get_uiuc_proposals(kuali_conn)
        uic_funded_proposals = _get_uic_funded_proposals(kuali_conn)

    allowed = uiuc_proposals | uic_funded_proposals
    print(f"\n  [METANODE] UIUC proposals:         {len(uiuc_proposals):,}")
    print(f"  [METANODE] UIC funded proposals:   {len(uic_funded_proposals):,}")
    print(f"  [METANODE] Combined (deduplicated): {len(allowed):,}")

    if dry_run:
        print("\n  [METANODE] DRY RUN — FND_ID table not updated.")
        print(f"  Would write {len(allowed):,} proposal numbers.")
        return allowed

    # ── Write combined set to analytics DB ─────────────────────────
    print("\n[METANODE] Connecting to analytics DB...")
    with oracle_conn(
        dsstag["host"], dsstag["port"], dsstag["service"],
        creds["username"], creds["password"]
    ) as dsstag_conn:
        print("[METANODE] Connected.")
        _write_fnd_id_table(dsstag_conn, allowed)

    print("\n===================================")
    print(" METANODE COMPLETE")
    print("===================================\n")

    return allowed
