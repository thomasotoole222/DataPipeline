"""Unit tests for ``pipeline.registry``: the published jobs and sources, and the runners that declare them."""

from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import pytest

from pipeline.db.control import PublishedJob, PublishedSource
from pipeline.jobs.archive_raw import ArchiveRawJob
from pipeline.jobs.base import BatchJob
from pipeline.registry import PIPELINE_SOURCE, Registry

SHOP = UUID("00000000-0000-0000-0000-00000000000a")
SHOP_A = PublishedSource("tiktok_shop", SHOP, "Shop A", in_all=True)


def test_registry_refuses_a_repeated_source() -> None:
    """Check a source listed twice under one connector is refused."""
    with pytest.raises(
        ValueError, match="sources listed more than once: tiktok_shop/00000000-0000-0000-0000-00000000000a"
    ):
        Registry(sources=(SHOP_A, SHOP_A))


def test_registry_refuses_a_package_source_that_is_the_pipeline_source() -> None:
    """Check a package source equal to the pipeline source is refused."""
    with pytest.raises(
        ValueError, match="sources listed more than once: pipeline/00000000-0000-0000-0000-000000000000"
    ):
        Registry(sources=(PIPELINE_SOURCE,))


def test_registry_publishes_each_job_with_its_connector_and_partition() -> None:
    """Check each job is published with the connector and partition it declares."""
    job = SimpleNamespace(name="orders", connector="tiktok_shop", partition="day")
    assert Registry(jobs=(cast(Any, job),)).published_jobs == (PublishedJob("orders", "tiktok_shop", "day"),)


def test_registry_publishes_the_pipeline_source_after_the_package_sources() -> None:
    """Check the pipeline source is published after the package's sources."""
    assert Registry(sources=(SHOP_A,)).published_sources == (SHOP_A, PIPELINE_SOURCE)
    assert PublishedSource("pipeline", UUID(int=0), "pipeline", in_all=True) == PIPELINE_SOURCE


def test_a_batch_job_names_its_spec_connector() -> None:
    """Check ``BatchJob.connector`` is the connector of the job's spec."""
    job = SimpleNamespace(spec=SimpleNamespace(connector="cruva"))
    assert cast(Any, BatchJob).connector.fget(job) == "cruva"


def test_archive_raw_runs_on_the_pipeline_connector_with_no_partition() -> None:
    """Check ``archive_raw`` is published on the pipeline connector with no partition."""
    assert (ArchiveRawJob.connector, ArchiveRawJob.partition) == ("pipeline", "none")
    registry = Registry(jobs=(ArchiveRawJob(refetch_days={}, store=None),))
    assert registry.published_jobs == (PublishedJob("archive_raw", "pipeline", "none"),)
