"""Unit tests for the raw archive's settling rule (``settled_on``) and ``Registry`` holding the archive job."""

from datetime import date

import pytest

from pipeline.archive.months import settled_on
from pipeline.jobs.archive_raw import ArchiveRawJob
from pipeline.registry import Registry


@pytest.mark.parametrize(
    ("month", "refetch_days", "first_day"),
    [
        (date(2026, 11, 1), 0, date(2026, 12, 3)),  # a snapshot job: the 3rd of the next month
        (date(2026, 11, 17), 2, date(2026, 12, 5)),  # orders: any day of the month names it
        (date(2026, 12, 1), 7, date(2027, 1, 10)),  # LIVE and video jobs, over the year end
        (date(2026, 1, 1), 35, date(2026, 3, 10)),  # shop_day: early in the month after next
        (date(2026, 2, 1), 60, date(2026, 5, 2)),  # samples: early in the third month
    ],
)
def test_a_month_settles_after_the_jobs_longest_refetch_and_the_margin(
    month: date, refetch_days: int, first_day: date
) -> None:
    """Check the first day a month may be written: next month's first day, plus the re-fetch, plus 2 days."""
    assert settled_on(month, refetch_days) == first_day


def test_the_registry_holds_the_archive_job_by_name() -> None:
    """Check a runner that is not a BatchJob goes in the registry under its name."""
    job = ArchiveRawJob(refetch_days={}, store=None)
    assert dict(Registry(jobs=(job,)).runners) == {"archive_raw": job}


def test_the_archive_job_refuses_a_horizon_for_itself() -> None:
    """Check archive_raw cannot be given a horizon: it has no pages of its own."""
    with pytest.raises(ValueError, match="has none of its own"):
        ArchiveRawJob(refetch_days={"archive_raw": 0}, store=None)
