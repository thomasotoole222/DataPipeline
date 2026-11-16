"""The registry of the jobs a worker can run, which an application package builds and hands to ``run``.

Public API:
    NamedRunner: Protocol for a job the registry holds: a task runner with a name.
    Registry: The jobs a worker runs, by name.
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # job code imports Polars, which must load only after run has pinned its threads
    from pipeline.db.control import ClaimedTask
    from pipeline.jobs.base import TaskLease


class NamedRunner(Protocol):
    """Define a job the registry holds: a runner for its tasks, named as tasks name it in ``control.task.job``.

    Every ``BatchJob`` satisfies it, and so does a job that is not one, such as ``ArchiveRawJob``.
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
    """Hold the jobs a worker runs; their names are unique.

    ``__post_init__`` enforces the unique names. An empty registry is allowed and runs no jobs.

    Attributes:
        jobs: The jobs: batch jobs, each built with its connector's client, and any other runners such as
            ``ArchiveRawJob``; ``()`` by default.

    """

    jobs: tuple[NamedRunner, ...] = ()

    def __post_init__(self) -> None:
        """Refuse two jobs with the same name.

        Every job has a ``name``: ``BatchJob.__init__`` raises ``AttributeError`` for a job class that does not set it,
        and a ``NamedRunner`` declares one.

        Preconditions:
            None.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: Two or more jobs share a name; the message lists the names.

        """
        repeated = sorted(name for name, count in Counter(job.name for job in self.jobs).items() if count > 1)
        if repeated:
            raise ValueError(f"job names used more than once: {', '.join(repeated)}")

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
