"""Startup checks that read the worker's CPU limit from cgroup v2 and pin the Polars thread pool to fit it.

Every failure is reported as ``StartupRefused``, so the worker refuses to start instead of running with a thread pool
that the container's CPU quota cannot serve.

Public API:
    cpu_quota: Return the tightest ``cpu.max`` limit on this process's cgroup and its ancestors, in CPUs.
    pin_polars_threads: Set the Polars thread pool size before Polars is imported, and check it against the quota.

Requirements:
    Linux with the cgroup v2 unified hierarchy mounted, by default at ``/sys/fs/cgroup``.

"""

import os
from pathlib import Path
from typing import Final

from cipher_pipeline.errors import StartupRefused

#: The file listing this process's cgroup memberships, one ``hierarchy-ID:controllers:path`` line each.
_PROC_CGROUP: Final = Path("/proc/self/cgroup")
#: The default mount point of the cgroup v2 unified hierarchy.
_CGROUP_ROOT: Final = Path("/sys/fs/cgroup")
#: The file name, at the root of a cgroup v2 hierarchy, whose presence marks the hierarchy as cgroup v2.
_CONTROLLERS: Final = "cgroup.controllers"
#: The file name, in each cgroup directory, holding the CPU bandwidth limit as ``"$QUOTA $PERIOD"``.
_CPU_MAX: Final = "cpu.max"
#: The prefix of the cgroup v2 line in ``/proc/self/cgroup``: hierarchy ID ``0`` and an empty controller list.
_V2_PREFIX: Final = "0::"


def _own_cgroup(proc_cgroup: Path) -> str:
    """Return this process's cgroup v2 path as listed in ``proc_cgroup``.

    The first line that starts with ``"0::"`` is used and any later ones are ignored. The text after the prefix is
    stripped of surrounding whitespace and is not otherwise normalised. Relies on the kernel's cgroup v2 contract that
    ``/proc/self/cgroup`` lists the unified hierarchy as ``0::$PATH``, with ``$PATH`` absolute and relative to the
    mount point of that hierarchy as seen by this process's cgroup namespace.

    Preconditions:
        None.

    Args:
        proc_cgroup: The file to read, normally ``/proc/self/cgroup``.

    Returns:
        The cgroup v2 path, beginning with ``"/"``. It is exactly ``"/"`` when the process is in the root cgroup.

    Raises:
        StartupRefused: If ``proc_cgroup`` cannot be read (any ``OSError``, chained as the cause), if no line starts
            with ``"0::"``, or if the first such line's path does not start with ``"/"``, including when it is empty.
        UnicodeDecodeError: If ``proc_cgroup`` is not valid text in the locale encoding. It is not caught.

    """
    try:
        text = proc_cgroup.read_text()
    except OSError as err:
        raise StartupRefused(f"cannot read {proc_cgroup}: {err}") from err
    for line in text.splitlines():
        if line.startswith(_V2_PREFIX):
            path = line.removeprefix(_V2_PREFIX).strip()
            if not path.startswith("/"):
                raise StartupRefused(f"cgroup v2 path in {proc_cgroup} is not absolute: {path!r}")
            return path
    raise StartupRefused(f"no cgroup v2 line ({_V2_PREFIX!r}) in {proc_cgroup}")


def _read_cpu_max(cpu_max: Path) -> float | None:
    """Return the CPU limit in one ``cpu.max`` file, as a number of CPUs.

    The contents are split on whitespace and must be exactly two fields, parsed by ``int``. The period is validated
    before the quota, so ``"max 0"`` is malformed rather than unlimited. Relies on the kernel's cgroup v2 contract that
    ``cpu.max`` holds ``"$MAX $PERIOD"``, where ``$MAX`` is ``max`` for no limit or the quota in microseconds, and
    that the file is absent where no limit can be set, such as in the root cgroup.

    Preconditions:
        None.

    Args:
        cpu_max: The ``cpu.max`` file to read.

    Returns:
        ``quota / period`` as a ``float``, for example ``1.5`` for ``"150000 100000"``. ``None`` if the file does not
        exist or its quota field is ``max``.

    Raises:
        StartupRefused: If the file cannot be read for any reason other than not existing (any other ``OSError``,
            chained as the cause), or if the contents are malformed: not exactly two fields, a period that is not an
            integer greater than zero, or a quota that is neither ``max`` nor an integer greater than zero. Each such
            ``ValueError`` is caught and chained as the cause.
        UnicodeDecodeError: If the file is not valid text in the locale encoding. It is not caught.
        Swallows ``FileNotFoundError``, returning ``None``.

    """
    try:
        text = cpu_max.read_text()
    except FileNotFoundError:
        return None
    except OSError as err:
        raise StartupRefused(f"cannot read {cpu_max}: {err}") from err
    fields = text.split()
    try:
        if len(fields) != 2:
            raise ValueError(f"expected 2 fields, got {len(fields)}")
        quota_text, period_text = fields
        period = int(period_text)
        if period <= 0:
            raise ValueError("period must be a positive integer")
        if quota_text == "max":
            return None
        quota = int(quota_text)
        if quota <= 0:
            raise ValueError("quota must be a positive")
    except ValueError as err:
        raise StartupRefused(f"malformed {cpu_max}: {text.strip()!r} ({err})") from err
    return quota / period


def cpu_quota(*, proc_cgroup: Path = _PROC_CGROUP, root: Path = _CGROUP_ROOT) -> float | None:
    """Return the tightest CPU limit on this process's cgroup and its ancestors, in CPUs.

    Reads ``cpu.max`` in the process's own cgroup directory under ``root``, then in each parent directory up to and
    including ``root``, and takes the smallest limit found. Directories without ``cpu.max`` and levels set to ``max``
    add no limit. The walk stops at ``root``, or earlier once ``root`` is no longer among the current directory's
    ``parents``; both tests compare paths lexically. Relies on the cgroup v2 contract that a cgroup's CPU bandwidth is
    also bounded by the ``cpu.max`` of every ancestor.

    Preconditions:
        ``root`` is the mount point of the cgroup v2 hierarchy that the path in ``proc_cgroup`` is relative to.

    Args:
        proc_cgroup: The file listing this process's cgroups, normally ``/proc/self/cgroup``.
        root: The cgroup v2 mount point, normally ``/sys/fs/cgroup``.

    Returns:
        The smallest ``quota / period`` across the levels read, as a ``float``. ``None`` if no level sets a limit,
        including when no ``cpu.max`` file exists at any level.

    Raises:
        StartupRefused: From ``_own_cgroup``, if ``proc_cgroup`` cannot be read or has no valid cgroup v2 line, and
            from ``_read_cpu_max``, if a ``cpu.max`` file exists but cannot be read or is malformed.
        UnicodeDecodeError: From either helper, if a file is not valid text. It is not caught.

    """
    level = root / _own_cgroup(proc_cgroup).lstrip("/")
    limits: list[float] = []
    while True:
        limit = _read_cpu_max(level / _CPU_MAX)
        if limit is not None:
            limits.append(limit)
        if level == root or root not in level.parents:
            break
        level = level.parent
    return min(limits, default=None)


def pin_polars_threads(threads: int, *, proc_cgroup: Path = _PROC_CGROUP, root: Path = _CGROUP_ROOT) -> None:
    """Set the Polars thread pool size, then check that Polars took it and that it fits the CPU quota.

    Checks that ``root / "cgroup.controllers"`` exists, as reported by ``Path.exists``. Only its presence is checked,
    not whether the ``cpu`` controller is listed in it. Then sets ``POLARS_MAX_THREADS`` in ``os.environ`` to
    ``str(threads)``, replacing any existing value, and imports ``polars``. Relies on the Polars contract that the
    global thread pool is sized from ``POLARS_MAX_THREADS`` once, when it is created, and that
    ``polars.thread_pool_size()`` reports its size. A pool created earlier with the same size as ``threads`` passes
    the check. The environment variable and the import persist when a later check fails. A quota equal to ``threads``
    is accepted.

    Preconditions:
        ``threads`` is a positive integer; the value is not validated here.
        Called before anything in the process imports ``polars``.

    Args:
        threads: The number of threads for the Polars pool.
        proc_cgroup: The file listing this process's cgroups, normally ``/proc/self/cgroup``.
        root: The cgroup v2 mount point, normally ``/sys/fs/cgroup``.

    Returns:
        None.

    Raises:
        StartupRefused: If ``root / "cgroup.controllers"`` does not exist, if the Polars thread pool size is not
            ``threads``, if ``threads`` is greater than the quota returned by ``cpu_quota``, or from ``cpu_quota``
            itself. The cgroup v2 check runs before ``os.environ`` is changed; the quota is read only after the pool
            size check passes.
        ImportError: If ``polars`` cannot be imported. It is not caught.
        UnicodeDecodeError: From ``cpu_quota``, if a cgroup file is not valid text. It is not caught.

    """
    controllers = root / _CONTROLLERS
    if not controllers.exists():
        raise StartupRefused(f"cgroup v2 is required: {controllers} does not exist")
    os.environ["POLARS_MAX_THREADS"] = str(threads)
    import polars as pl

    size = pl.thread_pool_size()
    if size != threads:
        raise StartupRefused(
            f"Polars thread pool is {size}, not polars_threads {threads}: Polars was imported before it was pinned"
        )
    quota = cpu_quota(proc_cgroup=proc_cgroup, root=root)
    if quota is not None and threads > quota:
        raise StartupRefused(f"polars_threads {threads} exceeds the CPU quota of {quota}")
