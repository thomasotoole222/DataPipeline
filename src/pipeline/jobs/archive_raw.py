"""The ``archive_raw`` job: write every settled month of raw pages to the archive, then check the archive.

Public API:
    ArchiveRawJob: The runner for ``archive_raw`` tasks.

"""

import logging
from collections.abc import Iterable, Mapping
from datetime import timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Self

from pipeline.archive.months import ArchiveCheckFailed, archive_month, check_manifest, plan_months
from pipeline.archive.store import ObjectStore
from pipeline.db.control import ClaimedTask, TaskReport
from pipeline.errors import Outcome, PermanentError
from pipeline.jobs.base import BatchJob, TaskLease
from pipeline.timeutil import utc_now

_log = logging.getLogger(__name__)


class ArchiveNotConfigured(PermanentError):
    """Signal that an ``archive_raw`` task ran on a worker with no archive store (``PIPELINE_ARCHIVE_*`` unset).

    Attributes:
        outcome (Outcome): ``Outcome.FAILED``, inherited from ``PermanentError``.

    """

    pass


class ArchiveRawJob:
    """Run ``archive_raw`` tasks: archive each settled month of each job, then check every recorded object.

    A task's source and partition key are ignored: one run covers every job and month. Months are written one at a
    time, oldest first (``plan_months``, ``archive_month``); each is recorded in its own transaction, so a month
    written stays written when a later one fails. A month whose check fails is skipped and the run goes on; once every
    ready month has been tried, ``check_manifest`` checks every recorded object. If any month failed its check or any
    object is missing or differs, the task fails with ``ArchiveCheckFailed`` naming each; otherwise it succeeds.
    Any other error (the store or Postgres unreachable) ends the run at once and is reported as the worker reports any
    runner error.

    The run logs, at ``WARNING``, the jobs with pages but no re-fetch horizon (never archived) and the pages whose
    task no longer exists, and at ``INFO`` the settled months that wait for an open task.

    The success report counts the pages archived as both ``rows_fetched`` and ``rows_written``.

    Attributes:
        name: ``"archive_raw"``.
        refetch_days: Each archived job's longest re-fetch in days of age, by job name, read-only.

    """

    name: ClassVar[str] = "archive_raw"

    def __init__(
        self, *, refetch_days: Mapping[str, int], store: ObjectStore | None, scratch: Path | None = None
    ) -> None:
        """Store the horizons, the store and the scratch directory.

        Preconditions:
            None.

        Args:
            refetch_days: Each archived job's longest re-fetch, in days of age, by job name.
            store: Where objects are written; ``None`` when this worker has none, so every task fails with
                ``ArchiveNotConfigured``.
            scratch: The directory local files are written in; the system's temporary directory when ``None``.

        Returns:
            None.

        Raises:
            ValueError: A horizon is not an ``int`` of 0 or more, or a job is named ``archive_raw``.

        """
        for job, days in refetch_days.items():
            if not isinstance(days, int) or isinstance(days, bool) or days < 0:
                raise ValueError(f"job {job}: refetch_days must be an int of 0 or more, not {days!r}")
        if self.name in refetch_days:
            raise ValueError(f"{self.name} archives other jobs' pages; it has none of its own")
        self.refetch_days: Mapping[str, int] = MappingProxyType(dict(refetch_days))
        self._store = store
        self._scratch = scratch

    @classmethod
    def for_jobs(cls, jobs: Iterable[BatchJob[Any]], store: ObjectStore | None, scratch: Path | None = None) -> Self:
        """Build the job with each batch job's own ``refetch_days`` as its horizon.

        Preconditions:
            None.

        Args:
            jobs: The registry's batch jobs.
            store: As ``__init__``.
            scratch: As ``__init__``.

        Returns:
            The job.

        Raises:
            ValueError: As ``__init__``.

        """
        return cls(refetch_days={job.name: job.refetch_days for job in jobs}, store=store, scratch=scratch)

    async def __call__(self, task: ClaimedTask, lease: TaskLease, /) -> None:
        """Archive every ready month, check the manifest, and report the outcome.

        Preconditions:
            ``lease`` is the lease this worker holds on ``task``.

        Args:
            task: The claimed task.
            lease: The lease on ``task``.

        Returns:
            None, once success has been reported.

        Raises:
            ArchiveNotConfigured: This worker has no store; nothing is read or written.
            ArchiveCheckFailed: A month failed its check or the manifest and the store disagree; the months that
                passed stay recorded.
            Exception: Whatever else ``plan_months``, ``archive_month``, ``check_manifest`` or ``lease.complete``
                raises, unchanged.

        """
        if self._store is None:
            raise ArchiveNotConfigured("no archive store: set the five PIPELINE_ARCHIVE_ settings")
        pool = lease.pool
        plan = await plan_months(pool, self.refetch_days, utc_now().date())
        for job, pages in sorted(plan.unknown.items()):
            _log.warning("archive: %d pages of job %s, which has no re-fetch horizon, are not archived", pages, job)
        if plan.orphans:
            _log.warning("archive: %d unarchived pages name no task and are not archived", plan.orphans)
        for waiting in plan.waiting:
            _log.info("archive: %s %s waits for its open tasks", waiting.job, f"{waiting.month:%Y-%m}")
        problems: list[str] = []
        written = 0
        for month_key in plan.ready:
            try:
                archived = await archive_month(pool, lease, self._store, month_key, self._scratch)
            except ArchiveCheckFailed as err:
                _log.error("archive: %s", err)
                problems.append(str(err))
                continue
            written += archived.rows
        problems += await check_manifest(pool, self._store)
        if problems:
            raise ArchiveCheckFailed("; ".join(problems))
        report = TaskReport(
            outcome=Outcome.SUCCEEDED,
            error_class=None,
            error_type=None,
            error_message=None,
            retry_after=None,
            rows_fetched=written,
            rows_written=written,
            rows_rejected=0,
            duration=max(utc_now() - task.claimed_at, timedelta(0)),
        )
        async with pool.connection() as conn:
            await lease.complete(conn, report)
