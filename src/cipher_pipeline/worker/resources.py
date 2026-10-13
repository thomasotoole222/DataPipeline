import os
from pathlib import Path
from typing import Final

from cipher_pipeline.errors import StartupRefused

_PROC_CGROUP: Final = Path("/proc/self/cgroup")
_CGROUP_ROOT: Final = Path("/sys/fs/cgroup")
_CONTROLLERS: Final = "cgroup.controllers"
_CPU_MAX: Final = "cpu.max"
_V2_PREFIX: Final = "0::"


def _own_cgroup(proc_cgroup: Path) -> str:
    try:
        text = proc_cgroup.read_text()
    except OSError as exc:
        raise StartupRefused(f"cannot read {proc_cgroup}: {exc}") from exc
    for line in text.splitlines():
        if line.startswith(_V2_PREFIX):
            path = line.removeprefix(_V2_PREFIX).strip()
            if not path.startswith("/"):
                raise StartupRefused(f"cgroup v2 path in {proc_cgroup} is not absolute: {path!r}")
            return path
    raise StartupRefused(f"no cgroup v2 line ({_V2_PREFIX!r}) in {proc_cgroup}")


def _read_cpu_max(cpu_max: Path) -> float | None:
    try:
        text = cpu_max.read_text()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise StartupRefused(f"cannot read {cpu_max}: {exc}") from exc
    fields = text.split()
    try:
        if len(fields) != 2:
            raise ValueError("expected 2 fields, got {len(fields)}")
        quota_text, period_text = fields
        period = int(period_text)
        if period <= 0:
            raise ValueError("period must be a positive integer")
        if quota_text == "max":
            return None
        quota = int(quota_text)
        if quota <= 0:
            raise ValueError("quota must be a positive")
    except ValueError as exc:
        raise StartupRefused(f"malformed {cpu_max}: {text.strip()!r} ({exc})") from exc
    return quota / period


def cpu_quota(*, proc_cgroup: Path = _PROC_CGROUP, root: Path = _CGROUP_ROOT) -> float | None:
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
    controllers = root / _CONTROLLERS
    if not controllers.exists():
        raise StartupRefused("cgroup v2 is required: {controllers} does not exist")
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
