# Splits an iterable into fixed-size batches.
#
# Oracle's default bind-variable limit is 1000 per statement, which is
# why every caller uses a batch_size <= 1000 when binding IDs into IN
# clauses. Keeping this generator-based rather than list-based means
# the caller can stream through very large ID sets without pre-
# materializing them all in memory.


def create_batches(ids, batch_size=500):
    for i in range(0, len(ids), batch_size):
        yield ids[i:i + batch_size]
