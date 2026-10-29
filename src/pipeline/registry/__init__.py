"""The registry of the jobs a worker can run, which an application package builds and hands to ``run``.

Public API:
    Registry: The jobs a worker runs, by name.
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # job code imports Polars, which must load only after run has pinned its threads
    from pipeline.jobs.base import BatchJob


@dataclass(frozen=True, slots=True)
class Registry:
    """Hold the jobs a worker runs; their names are unique.

    ``__post_init__`` enforces the unique names. An empty registry is allowed and runs no jobs.

    Attributes:
        jobs: The jobs, each built with its connector's client; ``()`` by default.

    """

    jobs: tuple[BatchJob[Any], ...] = ()

    def __post_init__(self) -> None:
        """Refuse two jobs with the same name.

        Every job has a ``name``: ``BatchJob.__init__`` raises ``AttributeError`` for a job class that does not set it.

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
    def runners(self) -> Mapping[str, BatchJob[Any]]:
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
