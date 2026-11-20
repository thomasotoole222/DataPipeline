"""The registry of the jobs a worker can run, which an application package builds and hands to ``run``.

Public API:
    PIPELINE_SOURCE: The one source of the framework's own jobs, always published.
    NamedRunner: Protocol for a job the registry holds: a task runner with a name, a connector and a partition.
    Registry: The jobs a worker runs, by name, and the sources each connector serves.
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol
from uuid import UUID

from pipeline.db.control import ClaimedTask, PublishedJob, PublishedSource

if TYPE_CHECKING:  # job code imports Polars, which must load only after run has pinned its threads
    from pipeline.jobs.base import Partition, TaskLease

#: The one source of the framework's own jobs, such as ``archive_raw``: the pipeline itself, always published.
PIPELINE_SOURCE: Final = PublishedSource("pipeline", UUID(int=0), "pipeline", in_all=True)


class NamedRunner(Protocol):
    """Define a job the registry holds: a runner for its tasks, with the name, connector and partition they carry.

    Every ``BatchJob`` satisfies it, through its ``name`` and ``partition`` class attributes and its ``connector``
    property, and so does a job that is not one, such as ``ArchiveRawJob``, through its three class attributes.
    """

    @property
    def name(self) -> str:
        """Return the job's name.

        Preconditions:
            None.

        Returns:
            The name tasks give in ``control.task.job``.

        Raises:
            Not specified by the protocol.

        """
        ...

    @property
    def connector(self) -> str:
        """Return the connector the job's tasks are created under.

        Preconditions:
            None.

        Returns:
            The connector its tasks name in ``control.task.connector``.

        Raises:
            Not specified by the protocol.

        """
        ...

    @property
    def partition(self) -> Partition:
        """Return how the job's tasks slice its work.

        Preconditions:
            None.

        Returns:
            ``"day"`` or ``"none"``.

        Raises:
            Not specified by the protocol.

        """
        ...

    async def __call__(self, task: ClaimedTask, lease: TaskLease, /) -> None:
        """Run one claimed task under its lease and report its success, as ``TaskRunner`` does.

        Preconditions:
            ``lease`` is the lease this worker holds on ``task``.

        Args:
            task: The claimed task.
            lease: The lease on ``task``.

        Returns:
            None, once the outcome has been reported.

        Raises:
            Exception: Whatever the job raises; the worker reports it as the task's outcome.

        """
        ...


@dataclass(frozen=True, slots=True)
class Registry:
    """Hold the jobs a worker runs, whose names are unique, and the sources each of the package's connectors serves.

    ``__post_init__`` enforces the unique names and the unique sources. An empty registry is allowed: it runs no jobs
    and publishes only ``PIPELINE_SOURCE``.

    Attributes:
        jobs: The jobs: batch jobs, each built with its connector's client, and any other runners such as
            ``ArchiveRawJob``; ``()`` by default.
        sources: For each of the package's connectors, every source that connector serves; ``()`` by default. The
            SANDBOX source is listed with ``in_all`` false, every other with ``in_all`` true; none is the pipeline
            source.

    """

    jobs: tuple[NamedRunner, ...] = ()  # batch jobs, and runners such as ArchiveRawJob
    sources: tuple[PublishedSource, ...] = ()  # every source each of the package's connectors serves

    def __post_init__(self) -> None:
        """Refuse two jobs with the same name, then a source published more than once.

        Every job has a ``name``: ``BatchJob.__init__`` raises ``AttributeError`` for a job class that does not set it,
        and a ``NamedRunner`` declares one. The sources checked are ``published_sources``, so a package source equal
        to ``PIPELINE_SOURCE`` is refused too.

        Preconditions:
            None.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: Two or more jobs share a name; the message lists the names.
            ValueError: A (connector, source id) appears more than once in ``published_sources``; the message lists
                them as ``connector/source_id``.

        """
        repeated = sorted(name for name, count in Counter(job.name for job in self.jobs).items() if count > 1)
        if repeated:
            raise ValueError(f"job names used more than once: {', '.join(repeated)}")
        keys = Counter((source.connector, source.source_id) for source in self.published_sources)
        twice = sorted(f"{connector}/{source_id}" for (connector, source_id), count in keys.items() if count > 1)
        if twice:
            raise ValueError(f"sources listed more than once: {', '.join(twice)}")

    @property
    def runners(self) -> Mapping[str, NamedRunner]:
        """Return each job by its name, read-only.

        A new mapping is built on each access. No job is lost to a name clash: ``__post_init__`` refuses repeated
        names.

        Preconditions:
            None.

        Returns:
            A read-only mapping from job name to job, one entry per job in ``jobs``.

        Raises:
            Nothing.

        """
        return MappingProxyType({job.name: job for job in self.jobs})

    @property
    def published_jobs(self) -> tuple[PublishedJob, ...]:
        """Return the jobs as ``control.publish_registry`` publishes them.

        Preconditions:
            None.

        Returns:
            One ``PublishedJob`` per job, in ``jobs`` order, with the job's name, connector and partition.

        Raises:
            ValueError: As ``PublishedJob``, for an empty name or connector or an unknown partition.

        """
        return tuple(PublishedJob(job.name, job.connector, job.partition) for job in self.jobs)

    @property
    def published_sources(self) -> tuple[PublishedSource, ...]:
        """Return the sources as ``control.publish_registry`` publishes them.

        Preconditions:
            None.

        Returns:
            The package's ``sources``, in order, then ``PIPELINE_SOURCE``.

        Raises:
            Nothing.

        """
        return (*self.sources, PIPELINE_SOURCE)
