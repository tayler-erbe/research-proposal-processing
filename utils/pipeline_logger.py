# utils/pipeline_logger.py
#
# CANONICAL logger for all pipelines monitored by job_execution_monitor.
# Drop this file into utils/ for any pipeline.
#
# Replaces:
#   - logger_for_proposal_award_processing.py  (write_log function)
#   - logger_for_image_processing.py           (write_log function)
#   - pipeline_logger_for_pipeline_demo.py     (PipelineLogger class, v1)
#   - pipeline_logger_for_proposals.py         (PipelineLogger class, v2)
#
# All pipelines use the same class and the same parquet schema so the
# job_execution_monitor can read every log file uniformly.
#
# ── Quick-start ──────────────────────────────────────────────────────────────
#
#   from utils.pipeline_logger import PipelineLogger
#
#   logger = PipelineLogger(
#       workflow_name="proposal_processing",    # must be unique per pipeline
#       log_dir=Path(".../logs"),
#       pipeline_version="2026-04-17",          # optional but recommended
#   )
#
#   with logger.pipeline_run():
#       with logger.stage("ingestion", order=1):
#           ids = run_ingestion()
#           logger.set_stage_metadata(
#               input_count=20, output_count=len(ids),
#               input_file="kuali_blob_query",
#               output_file="storage/intermediate_tables/proposal_full_table.parquet",
#               output_file_written=True,
#           )
#
#       with logger.stage("nlp", order=2):
#           run_nlp()
#           logger.set_stage_metadata(input_count=10, output_count=10, rows_written=10)
#
#       # Intentional skip — still logs SUCCESS so monitor doesn't alert
#       with logger.stage("db_write", order=3):
#           logger.set_stage_metadata(skip_reason="no_new_proposals")
#
# ── Schema changelog ─────────────────────────────────────────────────────────
#
#   v1  Original PipelineLogger — doc_count, time_per_doc
#   v2  + input_count, output_count, rows_written, skip_reason, pipeline_version
#   v3  + input_file, output_file, input_file_exists, output_file_written   (from image logger)
#       + hostname, user, python_version, working_directory                  (from image logger)
#       + logged_at                                                           (from image logger)
#       + defensive write (logger never crashes the pipeline)
#       + back-compat column fill on read (old parquet files stay readable)
#
# ─────────────────────────────────────────────────────────────────────────────

from typing import Optional
import traceback as tb
import uuid
import socket
import getpass
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd


# ── Canonical schema ─────────────────────────────────────────────────────────
# Keep this in sync with log_reader.py REQUIRED_COLUMNS.

REQUIRED_COLUMNS = [
    # Identity
    "workflow_name",
    "pipeline_version",
    "run_id",
    "script_name",
    "script_order",
    # Status
    "status",
    "is_success",
    "failure_stage",
    # Timing
    "event_time",
    "start_time",
    "end_time",
    "duration_seconds",
    "logged_at",
    # Volume metrics
    "doc_count",            # back-compat alias (v1/v2 logs)
    "time_per_doc",         # back-compat alias (v1/v2 logs)
    "input_count",
    "output_count",
    "rows_written",
    # Skip tracking
    "skip_reason",
    # File tracking
    "input_file",
    "output_file",
    "input_file_exists",
    "output_file_written",
    # Error details
    "error_type",
    "error_message",
    "traceback",
    # Environment
    "hostname",
    "user",
    "python_version",
    "working_directory",
]

# Status constants — keep in sync with monitor's alert_rules.py
STATUS_SUCCESS         = "SUCCESS"
STATUS_FAILED          = "FAILED"
STATUS_UPSTREAM_FAILED = "UPSTREAM_FAILED"
STATUS_PIPELINE_START  = "PIPELINE_START"
STATUS_PIPELINE_END    = "PIPELINE_END"
SCRIPT_NAME_PIPELINE   = "PIPELINE"   # sentinel for detect_silent_pipelines


def _env() -> dict:
    """Capture environment snapshot once at logger construction time."""
    return {
        "hostname":          socket.gethostname(),
        "user":              getpass.getuser(),
        "python_version":    sys.version,
        "working_directory": str(Path.cwd()),
    }


class PipelineLogger:
    """
    Structured execution logger for all pipelines monitored by job_execution_monitor.

    One instance per pipeline run. Use as context managers:

        logger = PipelineLogger("my_pipeline", log_dir, pipeline_version="2026-04-17")
        with logger.pipeline_run():
            with logger.stage("step_one", order=1):
                do_work()
                logger.set_stage_metadata(input_count=100, output_count=95)

    Parameters
    ----------
    workflow_name : str
        Unique pipeline identifier. Must be consistent across runs — the monitor
        groups alerts and silence detection by this value.
    log_dir : Path
        Directory for pipeline_log.parquet. Created if it doesn't exist.
    run_id : str, optional
        Auto-generated UUID if not provided. Pass explicitly to correlate
        logs across systems.
    pipeline_version : str, optional
        Short version tag stamped on every row (e.g. "2026-04-17" or "1.3.0").
        Makes it easy to correlate behavioral changes to code deployments.
    """

    def __init__(
        self,
        workflow_name:    str,
        log_dir:          Path,
        run_id:           Optional[str] = None,
        pipeline_version: Optional[str] = None,
    ):
        self.workflow_name    = workflow_name
        self.log_dir          = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path         = self.log_dir / "pipeline_log.parquet"
        self.run_id           = run_id or str(uuid.uuid4())
        self.pipeline_version = pipeline_version
        self._env             = _env()

        # Tracks first failure so downstream stages get UPSTREAM_FAILED
        self._upstream_failed = False

        # Scratch dict populated by set_stage_metadata() inside a stage block.
        # Cleared at the start of every stage so values never bleed between stages.
        self._stage_meta: dict = {}

    # ── Metadata injection ───────────────────────────────────────────────────

    def set_stage_metadata(
        self,
        input_count:         Optional[int]   = None,
        output_count:        Optional[int]   = None,
        rows_written:        Optional[int]   = None,
        skip_reason:         Optional[str]   = None,
        input_file:          Optional[str]   = None,
        output_file:         Optional[str]   = None,
        input_file_exists:   Optional[bool]  = None,
        output_file_written: Optional[bool]  = None,
        doc_count:           Optional[int]   = None,
        time_per_doc:        Optional[float] = None,
    ) -> None:
        """
        Attach metadata to the currently-running stage.

        Call from *inside* a `with logger.stage(...)` block after the work
        completes. Values are written into the SUCCESS (or FAILED) row when
        the context manager exits.

        Parameters
        ----------
        input_count         : Items entering this stage (IDs detected, images found, bills fetched…)
        output_count        : Items successfully produced (rows extracted, images classified…)
        rows_written        : Rows committed to a DB table or written to output files
        skip_reason         : Why this stage was intentionally bypassed
                              ("no_new_proposals", "no_new_awards", "no_new_images"…).
                              Stage still logs SUCCESS so the monitor doesn't alert,
                              but the reason is visible in the parquet for audit.
        input_file          : Path or label for the primary input to this stage
        output_file         : Path or label for the primary output of this stage
        input_file_exists   : Whether the expected input file was present
        output_file_written : Whether the expected output file was written successfully
        doc_count           : Back-compat alias for items processed (v1/v2 pipelines)
        time_per_doc        : Back-compat — seconds per item processed (v1/v2 pipelines)
        """
        updates = {
            "input_count":         input_count,
            "output_count":        output_count,
            "rows_written":        rows_written,
            "skip_reason":         skip_reason,
            "input_file":          input_file,
            "output_file":         output_file,
            "input_file_exists":   input_file_exists,
            "output_file_written": output_file_written,
            "doc_count":           doc_count,
            "time_per_doc":        time_per_doc,
        }
        for k, v in updates.items():
            if v is not None:
                self._stage_meta[k] = v

    # ── Context managers ─────────────────────────────────────────────────────

    @contextmanager
    def pipeline_run(self):
        """
        Wraps the entire pipeline execution.
        Writes PIPELINE_START on enter, PIPELINE_END or FAILED on exit.
        detect_silent_pipelines in the monitor keys on this row.
        """
        pipeline_start = datetime.now(timezone.utc)
        self._write_row(
            script_name=SCRIPT_NAME_PIPELINE,
            script_order=0,
            status=STATUS_PIPELINE_START,
            is_success=False,
            start_time=pipeline_start,
            end_time=pipeline_start,
            duration_seconds=0.0,
        )

        pipeline_failed = False
        pipeline_error: Optional[Exception] = None

        try:
            yield self
        except Exception as exc:
            pipeline_failed = True
            pipeline_error = exc
        finally:
            end_time = datetime.now(timezone.utc)
            duration = (end_time - pipeline_start).total_seconds()

            if pipeline_failed and pipeline_error is not None:
                self._write_row(
                    script_name=SCRIPT_NAME_PIPELINE,
                    script_order=0,
                    status=STATUS_FAILED,
                    is_success=False,
                    start_time=pipeline_start,
                    end_time=end_time,
                    duration_seconds=duration,
                    error_type=type(pipeline_error).__name__,
                    error_message=str(pipeline_error),
                    traceback_str=tb.format_exc(),
                    failure_stage="pipeline",
                )
                raise pipeline_error
            else:
                self._write_row(
                    script_name=SCRIPT_NAME_PIPELINE,
                    script_order=0,
                    status=STATUS_PIPELINE_END,
                    is_success=True,
                    start_time=pipeline_start,
                    end_time=end_time,
                    duration_seconds=duration,
                )

    @contextmanager
    def stage(self, script_name: str, order: int):
        """
        Wraps a single named stage.

        Parameters
        ----------
        script_name : str   Human-readable stage name ("ingestion", "nlp", "db_write"…)
        order       : int   Execution order (1, 2, 3…)
        """
        # Clear metadata from previous stage
        self._stage_meta = {}

        if self._upstream_failed:
            now = datetime.now(timezone.utc)
            self._write_row(
                script_name=script_name,
                script_order=order,
                status=STATUS_UPSTREAM_FAILED,
                is_success=False,
                start_time=now,
                end_time=now,
                duration_seconds=0.0,
                failure_stage=script_name,
            )
            raise RuntimeError(
                f"Stage '{script_name}' skipped — upstream failure in this run."
            )

        start_time = datetime.now(timezone.utc)
        try:
            yield
            end_time = datetime.now(timezone.utc)
            self._write_row(
                script_name=script_name,
                script_order=order,
                status=STATUS_SUCCESS,
                is_success=True,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                **self._stage_meta,
            )
        except Exception as exc:
            end_time = datetime.now(timezone.utc)
            self._upstream_failed = True
            self._write_row(
                script_name=script_name,
                script_order=order,
                status=STATUS_FAILED,
                is_success=False,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                error_type=type(exc).__name__,
                error_message=str(exc),
                traceback_str=tb.format_exc(),
                failure_stage=script_name,
                **self._stage_meta,
            )
            raise

    # ── Internal write ───────────────────────────────────────────────────────

    def _write_row(
        self,
        script_name:         str,
        script_order:        int,
        status:              str,
        is_success:          bool,
        start_time:          datetime,
        end_time:            datetime,
        duration_seconds:    float,
        doc_count:           Optional[int]   = None,
        time_per_doc:        Optional[float] = None,
        input_count:         Optional[int]   = None,
        output_count:        Optional[int]   = None,
        rows_written:        Optional[int]   = None,
        skip_reason:         Optional[str]   = None,
        input_file:          Optional[str]   = None,
        output_file:         Optional[str]   = None,
        input_file_exists:   Optional[bool]  = None,
        output_file_written: Optional[bool]  = None,
        error_type:          Optional[str]   = None,
        error_message:       Optional[str]   = None,
        traceback_str:       Optional[str]   = None,
        failure_stage:       Optional[str]   = None,
    ) -> None:
        """
        Append one row to pipeline_log.parquet.
        Defensive: never raises — a logging failure must never crash the pipeline.
        """
        try:
            now = datetime.now(timezone.utc)
            row = {
                # Identity
                "workflow_name":     self.workflow_name,
                "pipeline_version":  self.pipeline_version,
                "run_id":            self.run_id,
                "script_name":       script_name,
                "script_order":      script_order,
                # Status
                "status":            status,
                "is_success":        is_success,
                "failure_stage":     failure_stage,
                # Timing
                "event_time":        now,
                "start_time":        start_time,
                "end_time":          end_time,
                "duration_seconds":  duration_seconds,
                "logged_at":         now,
                # Volume metrics
                "doc_count":         doc_count,
                "time_per_doc":      time_per_doc,
                "input_count":       input_count,
                "output_count":      output_count,
                "rows_written":      rows_written,
                # Skip tracking
                "skip_reason":       skip_reason,
                # File tracking
                "input_file":        input_file,
                "output_file":       output_file,
                "input_file_exists":    input_file_exists,
                "output_file_written":  output_file_written,
                # Error details
                "error_type":        error_type,
                "error_message":     error_message,
                "traceback":         traceback_str,
                # Environment (captured at construction)
                **self._env,
            }

            new_df = pd.DataFrame([row])

            if self.log_path.exists():
                try:
                    existing = pd.read_parquet(self.log_path)
                    # Back-compat: add columns missing from older log files
                    for col in REQUIRED_COLUMNS:
                        if col not in existing.columns:
                            existing[col] = None
                    combined = pd.concat([existing, new_df], ignore_index=True)
                except Exception:
                    # Corrupted parquet — preserve new record rather than losing it
                    combined = new_df
            else:
                combined = new_df

            combined.to_parquet(self.log_path, index=False)

        except Exception as logging_error:
            # Absolute last resort — print only, never raise
            print(f"[LOGGER ERROR] Failed to write log row for stage '{script_name}'")
            print(f"[LOGGER ERROR] {logging_error}")


# ── Tests ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import shutil
    TEST_LOG_DIR = Path("/tmp/test_pipeline_v3/logs")
    if TEST_LOG_DIR.exists():
        shutil.rmtree(TEST_LOG_DIR)

    def fake_work(label, seconds=0.05):
        import time; time.sleep(seconds); print(f"  [{label}] done")

    def fake_fail():
        raise ValueError("Simulated failure: model file not found")

    # ── Test 1: Full successful run — proposals pattern ───────────────────
    print("--- Test 1: proposal pipeline (full metadata) ---")
    logger = PipelineLogger("proposal_processing", TEST_LOG_DIR, pipeline_version="2026-04-17")
    with logger.pipeline_run():
        with logger.stage("ingestion", order=1):
            fake_work("ingestion", 0.1)
            logger.set_stage_metadata(
                input_count=12, output_count=10, doc_count=10,
                input_file="kuali_blob_query",
                output_file="storage/intermediate_tables/proposal_full_table.parquet",
                output_file_written=True,
            )
        with logger.stage("nlp", order=2):
            fake_work("nlp", 0.2)
            logger.set_stage_metadata(input_count=10, output_count=10)
        with logger.stage("db_write", order=3):
            fake_work("db_write", 0.05)
            logger.set_stage_metadata(rows_written=10)

    # ── Test 2: Intentional skip ──────────────────────────────────────────
    print("\n--- Test 2: intentional skip (no_new_proposals) ---")
    logger2 = PipelineLogger("proposal_processing", TEST_LOG_DIR, pipeline_version="2026-04-17")
    with logger2.pipeline_run():
        with logger2.stage("ingestion", order=1):
            fake_work("ingestion", 0.05)
            logger2.set_stage_metadata(input_count=0, output_count=0)
        with logger2.stage("nlp", order=2):
            logger2.set_stage_metadata(skip_reason="no_new_proposals")
        with logger2.stage("db_write", order=3):
            logger2.set_stage_metadata(skip_reason="no_new_proposals")

    # ── Test 3: Failure + UPSTREAM_FAILED cascade ─────────────────────────
    print("\n--- Test 3: NLP failure → db_write UPSTREAM_FAILED ---")
    logger3 = PipelineLogger("proposal_processing", TEST_LOG_DIR, pipeline_version="2026-04-17")
    try:
        with logger3.pipeline_run():
            with logger3.stage("ingestion", order=1):
                fake_work("ingestion", 0.05)
                logger3.set_stage_metadata(input_count=12, output_count=10)
            with logger3.stage("nlp", order=2):
                fake_fail()
            with logger3.stage("db_write", order=3):
                fake_work("db_write")
    except Exception:
        pass

    # ── Test 4: Image classification pattern ─────────────────────────────
    print("\n--- Test 4: image_classification pipeline ---")
    logger4 = PipelineLogger("image_classification", TEST_LOG_DIR, pipeline_version="2026-04-17")
    with logger4.pipeline_run():
        with logger4.stage("step_generate_descriptions", order=1):
            fake_work("descriptions", 0.1)
            logger4.set_stage_metadata(
                input_count=1000, output_count=987,
                input_file="storage/images/",
                output_file="storage/outputs/descriptions.parquet",
                output_file_written=True,
            )
        with logger4.stage("step_semantic_similarity", order=2):
            fake_work("similarity", 0.1)
            logger4.set_stage_metadata(input_count=987, output_count=987, rows_written=987)

    # ── Results ───────────────────────────────────────────────────────────
    print("\n--- Log contents ---")
    df = pd.read_parquet(TEST_LOG_DIR / "pipeline_log.parquet")
    display_cols = [
        "workflow_name", "script_name", "status", "is_success",
        "duration_seconds", "input_count", "output_count",
        "rows_written", "skip_reason", "output_file_written",
        "pipeline_version", "hostname", "error_message",
    ]
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(df[display_cols].to_string())

    print(f"\nFull schema ({len(df.columns)} columns):")
    print(sorted(df.columns.tolist()))
    print(f"\nLog written to: {TEST_LOG_DIR / 'pipeline_log.parquet'}")
