"""Pipeline configuration read from ``PIPELINE_``-prefixed environment variables and a dotenv file.

Importing the module computes ``ENV_FILE`` and defines ``Settings`` but reads no settings. Settings are read and
validated when ``Settings`` is constructed.

Public API:
    ENV_FILE: Absolute path of the dotenv file that ``Settings`` reads.
    Settings: Frozen settings model holding the validated pipeline configuration.

Requirements:
    Python 3.14 or later, pydantic 2.x and pydantic-settings 2.x.

"""

#: Public names of the module: the names exported by ``from <module> import *``.
__all__ = ["ENV_FILE", "Settings"]

import os
import re
import socket
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, StringConstraints, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Pattern that the start of a warehouse URL must match: a scheme, then ``://``, then at least one non-whitespace
#: character.
_WAREHOUSE_URL_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://\S")


def _default_worker_id() -> str:
    """Return an identifier for the current worker.

    Uses the ``RAILWAY_REPLICA_ID`` environment variable when it is set to a non-blank value. Otherwise builds an
    identifier from the host name and the current process ID.

    Preconditions:
        None.

    Args:
        None.

    Returns:
        str: The value of ``RAILWAY_REPLICA_ID`` with surrounding whitespace stripped, if that is non-empty.
        Otherwise ``"<hostname>-<pid>"``, where ``<hostname>`` is ``socket.gethostname()``, or ``"unknown-host"``
        if that is empty, and ``<pid>`` is ``os.getpid()``. The value is computed afresh on every call.

    Raises:
        OSError: If ``socket.gethostname`` fails. This is let through.

    """
    replica_id = os.environ.get("RAILWAY_REPLICA_ID", "").strip()
    if replica_id:
        return replica_id
    hostname = socket.gethostname() or "unknown-host"
    return f"{hostname}-{os.getpid()}"


#: Absolute path of the ``.env`` file that ``Settings`` reads: the directory three levels above this file, after
#: symlinks are resolved, joined with ``.env``. Computed once, on import.
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    """Validated, immutable pipeline configuration read from ``PIPELINE_``-prefixed variables and ``ENV_FILE``.

    Values are read and validated once, at construction. Construction raises ``pydantic.ValidationError`` when a
    required value is missing or a value is invalid. Any key in ``ENV_FILE`` that is not a field fails validation, so
    the file may hold only ``PIPELINE_`` variables. Process environment variables that are not fields are ignored.
    Instances are frozen: assigning to an attribute raises ``ValidationError``. Validation errors do not include input
    values. ``env`` and ``log_level`` are stripped of surrounding whitespace and matched case-insensitively.

    Attributes:
        model_config (SettingsConfigDict): Class-level configuration: prefix ``PIPELINE_``, dotenv file ``ENV_FILE``
            read as UTF-8, extra keys forbidden, instances frozen, input values hidden in validation errors.
        warehouse_url (SecretStr): Connection URL for the warehouse database. Required. Must start with a scheme
            followed by ``://``. Masked in ``repr`` and ``str``.
        env (Literal["local", "staging", "production"]): Deployment environment. Required. Stored in lower case.
        worker_id (str): Identifier of this worker, stripped of surrounding whitespace and non-empty. Defaults to
            the result of ``_default_worker_id()``, computed once per instance.
        concurrency (int): Concurrency level, from 1 to 4 inclusive. Defaults to 1.
        log_level (Literal["DEBUG", "INFO", "WARNING"]): Log level name. Stored in upper case. Defaults to
            ``"INFO"``.

    """

    model_config = SettingsConfigDict(
        env_prefix="PIPELINE_",
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="forbid",  # Any key in ENV_FILE that isn't a field fails validation
        frozen=True,  # Instances are immutable after construction.
        hide_input_in_errors=True,  # Validation errors omit input values, so secrets are not echoed.
    )

    warehouse_url: SecretStr
    env: Literal["local", "staging", "production"]
    worker_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] = Field(
        default_factory=_default_worker_id
    )
    concurrency: int = Field(default=1, ge=1, le=4)
    log_level: Literal["DEBUG", "INFO", "WARNING"] = "INFO"

    @field_validator("warehouse_url")
    @classmethod
    def _check_warehouse_url(cls, value: SecretStr) -> SecretStr:
        """Reject a ``warehouse_url`` that does not start with a URL scheme followed by ``://``.

        Preconditions:
            None.

        Args:
            value: The ``warehouse_url`` value after ``SecretStr`` validation.

        Returns:
            ``value`` unchanged.

        Raises:
            ValueError: If the secret value does not match ``_WAREHOUSE_URL_PATTERN``, for example ``"localhost"``.
                Pydantic reports it as a ``ValidationError`` for ``warehouse_url``. The message does not contain the
                value.

        """
        if _WAREHOUSE_URL_PATTERN.match(value.get_secret_value()) is None:
            raise ValueError("warehouse_url must start with a URL scheme followed by '://'")
        return value

    @field_validator("env", mode="before")
    @classmethod
    def _normalise_env(cls, value: object) -> object:
        """Normalise a raw ``env`` value to lower case before ``Literal`` validation.

        Preconditions:
            None.

        Args:
            value: The raw input for ``env``, from any settings source.

        Returns:
            ``value.strip().lower()`` if ``value`` is a ``str``; otherwise ``value`` unchanged.

        Raises:
            None.

        """
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, value: object) -> object:
        """Normalise a raw ``log_level`` value to upper case before ``Literal`` validation.

        Preconditions:
            None.

        Args:
            value: The raw input for ``log_level``, from any settings source.

        Returns:
            ``value.strip().upper()`` if ``value`` is a ``str``; otherwise ``value`` unchanged.

        Raises:
            None.

        """
        return value.strip().upper() if isinstance(value, str) else value
