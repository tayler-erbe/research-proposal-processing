# Thin context manager wrapper around oracledb.
#
# Thick-client init is wrapped in try/except because Jupyter kernels
# frequently re-run setup cells, and oracledb raises if you initialize
# it twice. The client lib path below is the default Instant Client
# install location on Linux; override it for your environment if
# Instant Client lives elsewhere, or switch to thin mode by removing
# the init_oracle_client call entirely (LOB and some data-type
# handling differs in thin mode — thick is more forgiving).

import oracledb
from contextlib import contextmanager

try:
    oracledb.init_oracle_client(
        lib_dir="/usr/lib/oracle/19.6/client64/lib"
    )
    print("Oracle Thick Client initialized")
    print("Client version:", oracledb.clientversion())
except Exception as e:
    # Expected on re-runs in the same kernel; not fatal.
    print("Oracle client initialization skipped:", e)


@contextmanager
def oracle_conn(host, port, service, user, password):
    """Yield an Oracle connection and guarantee it's closed on exit."""
    dsn = oracledb.makedsn(host=host, port=port, service_name=service)
    conn = None
    try:
        conn = oracledb.connect(user=user, password=password, dsn=dsn)
        yield conn
    finally:
        if conn:
            conn.close()
