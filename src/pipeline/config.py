"""Pipeline configuration read from ``PIPELINE_``-prefixed environment variables and a dotenv file.

Importing the module computes ``ENV_FILE`` and defines ``Settings`` but reads no settings. Settings are read and
validated when ``Settings`` is constructed.

Public API:
    ENV_FILE: Absolute path of the dotenv file that ``Settings`` reads.
    Settings: Frozen settings model holding the validated pipeline configuration.

Requirements:
    Python 3.14 or later, pydantic 2.x, pydantic-settings 2.x and python-dotenv 1.x.

"""

#: Public names of the module: the names exported by ``from <module> import *``.
__all__ = ["ENV_FILE", "Settings"]

import hashlib
import inspect
import os
import re
import socket
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

import psycopg
from dotenv.main import resolve_variables
from dotenv.parser import Binding, parse_stream
from psycopg.conninfo import conninfo_to_dict
from pydantic import Field, SecretStr, StringConstraints, ValidationError, field_validator
from pydantic_core import InitErrorDetails
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict
from pydantic_settings.sources import DotenvType

#: pydantic-settings' underscore constructor arguments that ``Settings`` refuses: every one but ``_env_file``, since
#: each other one changes how keys or values are read (``_env_prefix``, ``_env_prefix_target``, ``_case_sensitive``,
#: ``_env_file_encoding``, ``_secrets_dir``, the ``_cli_`` arguments, ...) where ``_load`` does not follow it.
_REFUSED_ARGUMENTS = frozenset(
    name
    for name in inspect.signature(BaseSettings.__init__).parameters
    if name.startswith("_") and not name.startswith("__") and name != "_env_file"
)

#: Prefix of every setting's key, exactly as written: a field's key is this plus the field name in upper case, or
#: the field's validation alias, which must start with it.
_PREFIX = "PIPELINE_"

#: Pattern that the start of a warehouse URL must match: the scheme ``postgresql`` or ``postgres`` (the two libpq
#: accepts), then ``://``, then at least one non-whitespace character.
_WAREHOUSE_URL_PATTERN = re.compile(r"(?:postgresql|postgres)://\S")

#: libpq connection parameters that ``db.pool.open_pool`` sets on every connection, replacing any value the URL
#: gives: ``options``, ``application_name`` and the keys of ``db.pool.DEAD_CONNECTION_PARAMS``.
_POOL_PARAMETERS = frozenset(
    {
        "options",
        "application_name",
        "keepalives",
        "keepalives_idle",
        "keepalives_interval",
        "keepalives_count",
        "tcp_user_timeout",
    }
)

#: Longest ``worker_id`` in UTF-8 bytes: Postgres's 63-byte ``application_name`` limit less the 7 bytes of the
#: ``-listen`` suffix ``db.control`` appends.
_WORKER_ID_MAX_BYTES = 56

#: Number of hexadecimal digits of the host name's SHA-256 that ``_default_worker_id`` keeps when it cuts a long
#: host name.
_HOST_HASH_DIGITS = 8


def _default_worker_id() -> str:
    """Return an identifier for the current worker, at most ``_WORKER_ID_MAX_BYTES`` bytes in UTF-8.

    Builds the identifier from the host name and the current process ID. Set ``PIPELINE_WORKER_ID`` to use another
    value. A host name too long for the cap (a Kubernetes pod name may be 63 bytes) is cut and followed by a hash of
    the whole host name, so two long host names that share their start still give different identifiers.

    Preconditions:
        None.

    Args:
        None.

    Returns:
        str: ``"<hostname>-<pid>"``, where ``<hostname>`` is ``socket.gethostname()``, or ``"unknown-host"`` if that
        is empty, and ``<pid>`` is ``os.getpid()``, if that is at most ``_WORKER_ID_MAX_BYTES`` bytes in UTF-8.
        Otherwise ``"<head>-<hash>-<pid>"``, where ``<hash>`` is the first ``_HOST_HASH_DIGITS`` hexadecimal digits
        of the SHA-256 of the UTF-8 host name and ``<head>`` is the longest start of the host name, never splitting
        a character, that keeps the whole within ``_WORKER_ID_MAX_BYTES`` bytes. The value is computed afresh on
        every call.

    Raises:
        OSError: If ``socket.gethostname`` fails. This is let through.

    """
    hostname = socket.gethostname() or "unknown-host"
    worker_id = f"{hostname}-{os.getpid()}"
    if len(worker_id.encode("utf-8")) <= _WORKER_ID_MAX_BYTES:
        return worker_id
    digest = hashlib.sha256(hostname.encode("utf-8")).hexdigest()[:_HOST_HASH_DIGITS]
    tail = f"-{digest}-{os.getpid()}"
    head = hostname.encode("utf-8")[: _WORKER_ID_MAX_BYTES - len(tail)].decode("utf-8", errors="ignore")
    return head + tail


#: Absolute path of the ``.env`` file that ``Settings`` reads: ``Path(__file__).resolve().parents[2]`` (the directory
#: three levels above this file, after symlinks are resolved) joined with ``.env``. Computed once, on import. In an
#: editable install from ``src/pipeline/config.py`` this is the repository root's ``.env``; in a non-editable install
#: (wheel, ``uv sync --no-editable``; ``.../site-packages/pipeline/config.py``) it is ``.../lib/python3.14/.env``,
#: inside the environment, so no project ``.env`` is read there and, the file being missing, it is skipped without
#: error. An application package that installs this framework sets its own ``env_file``.
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


#: Line break as python-dotenv counts one.
_NEWLINE_PATTERN = re.compile(r"\r\n|\n|\r")


def _binding_line(binding: Binding) -> int:
    """Return the 1-based line number where a parsed dotenv binding's text starts, after any blank lines.

    python-dotenv's ``Original.line`` is the line where the binding's text starts, and that text includes the blank
    lines before it.

    Preconditions:
        ``binding`` came from ``dotenv.parser.parse_stream``.

    Args:
        binding: The parsed binding, a key or a line python-dotenv could not parse.

    Returns:
        ``binding.original.line`` plus the number of line breaks before the first non-whitespace character.

    Raises:
        None.

    """
    text = binding.original.string
    leading = text[: len(text) - len(text.lstrip())]
    return binding.original.line + len(_NEWLINE_PATTERN.findall(leading))


def _fold(key: str) -> str:
    """Return the form in which two keys that match case-insensitively are equal.

    Upper-casing first maps the dotless i (U+0131) to ``I``; case folding then maps ``I`` to ``i`` and the Kelvin sign
    (U+212A) to ``k``. So ``PIPELINE_ENV`` spelt with a dotless i and ``PIPELINE_WORKER_ID`` spelt with a Kelvin sign
    fold as ``PIPELINE_ENV`` and ``PIPELINE_WORKER_ID`` do.

    Preconditions:
        None.

    Args:
        key: The key.

    Returns:
        ``key.upper().casefold()``.

    Raises:
        None.

    """
    return key.upper().casefold()


def _setting_keys(settings_cls: type[BaseSettings]) -> dict[str, str]:
    """Return the exact key of each field of a settings class, subclass fields included, and its input name.

    A field's key is its validation alias if it has one, and ``_PREFIX`` plus the field name in upper case
    otherwise (``PIPELINE_WORKER_ID``).

    Preconditions:
        None.

    Args:
        settings_cls: The class being constructed.

    Returns:
        ``{key: name}``, where ``name`` is the name pydantic validates the field's value under: the validation alias,
        or the field name for a field without one.

    Raises:
        TypeError: If a field's validation alias is not a ``str`` (an ``AliasPath`` or ``AliasChoices``) or does not
            start with ``_PREFIX``; the message names the class and the field.

    """
    keys: dict[str, str] = {}
    for name, field in settings_cls.model_fields.items():
        alias = field.validation_alias
        if alias is None:
            keys[_PREFIX + name.upper()] = name
        elif isinstance(alias, str) and alias.startswith(_PREFIX):
            keys[alias] = alias
        else:
            raise TypeError(f"{settings_cls.__qualname__}.{name}: a validation alias must be a str starting {_PREFIX}")
    return keys


def _key_refusal(key: str, keys: Mapping[str, str]) -> str | None:
    """Return why a key is refused, or ``None`` if it is a setting's exact key or no ``PIPELINE_`` key at all.

    Keys are matched exactly. A key that folds (``_fold``) like a setting's key but is spelt otherwise
    (``pipeline_env``, or ``PIPELINE_ENV`` with a dotless i) is taken for a misspelling of it; any other key that
    folds like one starting with ``PIPELINE_`` is taken for a setting that does not exist.

    Preconditions:
        None.

    Args:
        key: The key, from the process environment or a dotenv file.
        keys: The settings' exact keys, from ``_setting_keys``.

    Returns:
        ``"<key> must be spelt <setting's key>"``, ``"<key> is not a setting"``, or ``None``. It holds no value.

    Raises:
        None.

    """
    if key in keys:
        return None
    spellings = {_fold(setting): setting for setting in keys}
    folded = _fold(key)
    if folded in spellings:
        return f"{key} must be spelt {spellings[folded]}"
    if folded.startswith(_fold(_PREFIX)):
        return f"{key} is not a setting"
    return None


def _read_env_file(path: Path, keys: Mapping[str, str]) -> tuple[dict[str, str], list[int], list[str]]:
    """Parse one dotenv file once, as UTF-8, into its settings, its offending line numbers and its refused keys.

    The grammar is python-dotenv's. An offending line holds a key with an empty value or no ``=`` at all, a key that
    does not fold (``_fold``) like one starting with ``PIPELINE_``, or text python-dotenv cannot parse; comments and
    blank lines are not offending. A key that folds like a ``PIPELINE_`` key is refused as ``_key_refusal`` says
    unless it is a setting's exact key, and a setting's key set on more than one line is refused. Values are
    interpolated as python-dotenv's ``dotenv_values`` does (``${NAME}`` from an earlier accepted line of this file if
    one set it, otherwise from the process environment, otherwise empty), so an interpolated value may be empty.

    Preconditions:
        ``path`` is a regular file (``_load``, the only caller, checks this with ``stat`` before calling).

    Args:
        path: The dotenv file.
        keys: The settings' exact keys, from ``_setting_keys``.

    Returns:
        ``(values, lines, refusals)``: ``values`` maps the input name of each setting the file sets (see
        ``_setting_keys``) to its value; ``lines`` holds the offending lines' 1-based numbers, in file order;
        ``refusals`` holds ``"key " + reason`` for each key ``_key_refusal`` refuses and ``"key <key> is set on more
        than one line"`` once for each repeated setting's key, in the order of the lines found. No value is in
        ``lines`` or ``refusals``.

    Raises:
        OSError: If the file cannot be read.
        UnicodeDecodeError: If the file is not valid UTF-8.

    """
    lines: list[int] = []
    refusals: list[str] = []
    accepted: dict[str, str | None] = {}  # setting's key -> its raw value, first line only
    repeated: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for binding in parse_stream(stream):
            key = binding.key
            if binding.error or (key is not None and (not binding.value or not _fold(key).startswith(_fold(_PREFIX)))):
                lines.append(_binding_line(binding))
            elif key is None:  # a comment or blank line
                continue
            elif (reason := _key_refusal(key, keys)) is not None:
                refusals.append(f"key {reason}")
            elif key not in accepted:
                accepted[key] = binding.value
            elif key not in repeated:
                repeated.add(key)
                refusals.append(f"key {key} is set on more than one line")
    resolved = resolve_variables(accepted.items(), override=True)
    return {keys[key]: value for key, value in resolved.items() if value is not None}, lines, refusals


def _load(settings_cls: type[BaseSettings], env_file: DotenvType | None) -> dict[str, str]:
    """Read the settings from the dotenv files and the process environment once, refusing any offending key or line.

    Each dotenv path is expanded with ``expanduser``, outside the ``stat`` error handling, so a path whose ``~`` names a
    home directory that cannot be found raises ``RuntimeError`` instead of being refused. A regular file is parsed by
    ``_read_env_file``; a path that exists but is not a regular file (a FIFO, a device, a directory) is refused without
    being opened, naming the path only, as its content cannot be checked; a missing path (``stat`` raises
    ``FileNotFoundError`` or ``NotADirectoryError``, a broken symlink included) is skipped silently; a path whose
    existence cannot be determined (``stat`` raises any other ``OSError``, such as ``PermissionError`` for a file in a
    directory without search permission, or ``ValueError`` for a path holding a NUL) is refused, naming the path only. A
    later file's value replaces an earlier file's, and a process environment value replaces both. Of the process
    environment's keys, a setting's exact key (see ``_setting_keys``) is read, a key ``_key_refusal`` refuses is
    refused, and any other key is ignored.

    Preconditions:
        None.

    Args:
        settings_cls: The class being constructed.
        env_file: The dotenv file or files to read; ``None`` reads none.

    Returns:
        The settings found, mapped from each one's input name (see ``_setting_keys``) to its value, a ``str``.

    Raises:
        ValidationError: If any key, path or line is refused: one error for the model, whose message joins every problem
            with ``"; "`` in this order: the process environment's (``environment variable PIPELINE_CONCURENCY is not a
            setting``, ``environment variable pipeline_env must be spelt PIPELINE_ENV``), sorted; the paths that are not
            regular files (``env file /dev/fd/63 is not a regular file``) or whose existence cannot be determined (``env
            file /root/.env cannot be checked: its status cannot be read``), in the order given; the offending lines of
            every file in one problem (``env file lines 3, 7 (2 in all): each line needs a PIPELINE_ key with a
            value``); each file's refused keys (``env file key PIPELINE_CONCURRENCY is set on more than one line``).
            ``env file`` is followed by the file's path when ``env_file`` is a sequence. The message names keys, paths
            and line numbers only, never a value or line text.
        TypeError: From ``_setting_keys``.
        OSError: If a dotenv path is a regular file but opening or reading it fails (no read permission). This is
            let through.
        UnicodeDecodeError: If a dotenv file is not valid UTF-8. This is let through.
        RuntimeError: If ``Path.expanduser`` cannot determine a home directory for a dotenv path (``~ghostuser/.env``
            for an unknown user, or ``~/.env`` with no ``HOME`` and no password-database entry). This is let through,
            before any other path or key is checked.

    """
    keys = _setting_keys(settings_cls)
    if isinstance(env_file, (str, os.PathLike)):
        single, files = True, [env_file]
    else:
        single, files = False, [] if env_file is None else list(env_file)
    values: dict[str, str] = {}
    refused_paths: list[str] = []
    bad_lines: list[str] = []
    file_refusals: list[str] = []
    for each in files:
        path = Path(each).expanduser()
        try:
            mode = path.stat().st_mode
        except FileNotFoundError, NotADirectoryError:  # missing: skipped
            continue
        except OSError, ValueError:  # existence unknown (no search permission, a NUL in the path): refused
            refused_paths.append(f"env file {path} cannot be checked: its status cannot be read")
            continue
        if stat.S_ISREG(mode):
            label = "env file " if single else f"env file {path} "
            file_values, lines, refusals = _read_env_file(path, keys)
            values |= file_values
            if lines:
                bad_lines.append(f"{label}lines {', '.join(map(str, lines))} ({len(lines)} in all)")
            file_refusals += [label + refusal for refusal in refusals]
        else:  # a FIFO, device or directory: never opened, so a FIFO is not consumed
            refused_paths.append(f"env file {path} is not a regular file")
    env_refusals: list[str] = []
    for key, value in os.environ.items():
        if key in keys:
            values[keys[key]] = value
        elif (reason := _key_refusal(key, keys)) is not None:
            env_refusals.append(f"environment variable {reason}")
    problems = sorted(env_refusals) + refused_paths
    if bad_lines:
        problems.append(f"{'; '.join(bad_lines)}: each line needs a PIPELINE_ key with a value")
    problems += file_refusals
    if problems:
        error = InitErrorDetails(type="value_error", loc=(), input=None, ctx={"error": "; ".join(problems)})
        raise ValidationError.from_exception_data(settings_cls.__name__, [error], hide_input=True)
    return values


class Settings(BaseSettings):
    """Validated, immutable pipeline configuration read from ``PIPELINE_`` variables and ``ENV_FILE``.

    Values are read once, at construction, by ``_load`` (pydantic-settings' environment, dotenv and secrets-directory
    sources are not used, see ``settings_customise_sources``) and then validated. Keys are matched exactly: a field's
    key is ``PIPELINE_`` plus its name in upper case (``PIPELINE_WORKER_ID``), or its validation alias, which must start
    with ``PIPELINE_``. Constructor arguments take precedence over the process environment, the process environment over
    the dotenv file, and a missing dotenv path is skipped without error. Construction raises
    ``pydantic.ValidationError`` when a key, line or dotenv path is refused (see ``_load``), a required value is missing
    or a value is invalid; it lets through ``OSError`` when a dotenv regular file cannot be opened or read or
    ``socket.gethostname`` fails (for the ``worker_id`` default), ``UnicodeDecodeError`` when a dotenv file is not valid
    UTF-8, and ``RuntimeError`` when a dotenv path's ``~`` names a home directory that cannot be found. Of
    pydantic-settings' underscore constructor arguments only ``_env_file`` is accepted; any other one (``_env_prefix``,
    ``_env_prefix_target``, ``_case_sensitive``, ``_env_file_encoding``, ``_secrets_dir``, ...) raises ``TypeError``
    (see ``__init__``).

    The dotenv file or files read are the ``_env_file`` constructor argument if one is passed (``None`` reads no
    file), otherwise ``model_config["env_file"]``, ``ENV_FILE`` here. Every line of a dotenv file must be a
    ``PIPELINE_`` key with a non-empty value, or a comment or blank line, or it is refused by line number, never by
    its text. In the process environment and the dotenv file alike, a key spelt like a setting's key in another case
    (``pipeline_env``, or one with a dotless i or a Kelvin sign) and an unknown ``PIPELINE_`` key are refused,
    naming the key and never its value; in the dotenv file a setting's key set on more than one line is refused,
    naming the key. A dotenv path that exists but is not a regular file (a FIFO, a device, a directory), or whose
    existence cannot be determined (a file in a directory without search permission), is refused, naming the path
    only. Values reach validation as strings, so a field must accept a string.

    Instances are frozen: assigning to an attribute raises ``ValidationError``. Validation errors do not include
    input values. ``env`` and ``log_level`` are stripped of surrounding whitespace and matched case-insensitively.

    Attributes:
        model_config (SettingsConfigDict): Class-level configuration: prefix ``PIPELINE_``, dotenv file ``ENV_FILE``
            read as UTF-8, extra keys forbidden, instances frozen, input values hidden in validation errors. Defaults
            are validated (``validate_default=True``, inherited from ``BaseSettings``).
        warehouse_url (SecretStr): Connection URL for the warehouse database. Required. Must start with
            ``postgresql://`` or ``postgres://`` and at least one non-whitespace character
            (``_WAREHOUSE_URL_PATTERN``), be a URL libpq parses, and set none of the connection parameters in
            ``_POOL_PARAMETERS``. Masked in ``repr`` and ``str``.
        env (Literal["local", "staging", "production"]): Deployment environment. Required. Stored in lower case.
        worker_id (str): Identifier of this worker, stripped of surrounding whitespace, non-empty and at most
            ``_WORKER_ID_MAX_BYTES`` (56) bytes in UTF-8, so ``"<worker_id>-listen"`` fits Postgres's 63-byte
            ``application_name``. Defaults to the result of ``_default_worker_id()``, computed once per instance and
            validated like a supplied value.
        concurrency (int): Concurrency level, from 1 to 4 inclusive. Defaults to 1.
        log_level (Literal["DEBUG", "INFO", "WARNING"]): Log level name. Stored in upper case. Defaults to
            ``"INFO"``.
        polars_threads (int): Size of Polars' thread pool, 1 or more, with no upper bound here. Defaults to 2. Not
            checked against the container's CPU limit by this class; the caller does that at startup.

    Connector secrets (app keys, API keys) are not settings: each connector client takes its own as constructor
    arguments, so ``ENV_FILE`` must not hold them.

    ``_load`` hard-codes the prefix ``PIPELINE_``, exact key matching and UTF-8, so a subclass whose ``model_config``
    (or class keywords) changes ``env_prefix``, ``case_sensitive`` or ``env_file_encoding`` is refused when the class
    is defined (see ``__init_subclass__``).

    """

    model_config = SettingsConfigDict(
        env_prefix="PIPELINE_",
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="forbid",  # A constructor argument that isn't a field fails validation
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
    polars_threads: int = Field(default=2, ge=1)  # Polars' thread-pool size; fixed apart from the container's CPU limit

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Refuse a subclass whose ``model_config`` changes ``env_prefix``, ``case_sensitive`` or ``env_file_encoding``.

        ``_load`` hard-codes the prefix ``PIPELINE_``, exact key matching and UTF-8, so a subclass must keep all
        three as ``model_config`` states them. A subclass may not set ``secrets_dir`` either: ``_load`` is the only
        reader of settings and does not read a secrets directory. Every other configuration key is accepted, including
        ones ``_load`` does not follow (``env_ignore_empty``, ``env_parse_none_str``, ``env_nested_delimiter``,
        ``env_prefix_target``), which then have no effect; ``__init__`` refuses the matching underscore arguments.

        Preconditions:
            Called by Python while defining ``cls``, after pydantic has merged its ``model_config``.

        Args:
            **kwargs: Class keywords, passed on to ``BaseSettings.__init_subclass__``.

        Returns:
            None.

        Raises:
            TypeError: If ``cls.model_config`` has an ``env_prefix`` other than ``"PIPELINE_"``, a ``case_sensitive``
                other than False or an ``env_file_encoding`` other than ``"utf-8"``, or a ``secrets_dir`` other than
                ``None``; the message names the class.
            Any exception ``BaseSettings.__init_subclass__`` raises.

        """
        super().__init_subclass__(**kwargs)
        config = cls.model_config
        if (
            config.get("env_prefix") != "PIPELINE_"
            or config.get("case_sensitive", False) is not False
            or config.get("env_file_encoding") != "utf-8"
        ):
            keep = "env_prefix='PIPELINE_', case_sensitive=False and env_file_encoding='utf-8'"
            raise TypeError(f"{cls.__qualname__} must keep {keep}")
        if config.get("secrets_dir") is not None:
            raise TypeError(f"{cls.__qualname__} may not set secrets_dir: _load does not read a secrets directory")

    def __init__(self, **values: Any) -> None:
        """Refuse every pydantic-settings underscore argument but ``_env_file``, then read and validate the settings.

        The settings ``_load`` reads from the dotenv files and the process environment are validated with ``values``,
        a value in ``values`` replacing the one read for the same name. pydantic-settings' own dotenv source is given
        no file. An argument in ``_REFUSED_ARGUMENTS`` is refused even when it passes the value already in force.

        Preconditions:
            None.

        Args:
            **values: Field values, and optionally ``_env_file``: the dotenv file or files to read instead of
                ``model_config["env_file"]``, or ``None`` for none.

        Returns:
            None.

        Raises:
            TypeError: If ``values`` holds a name in ``_REFUSED_ARGUMENTS``; the message names each such argument and
                no value. Also from ``_load``.
            ValidationError: From ``_load``, or if a value is missing or invalid, or ``values`` holds a name that is
                not a field.
            Any other exception ``_load`` or ``BaseSettings.__init__`` raises; see the class docstring.

        """
        refused = sorted(_REFUSED_ARGUMENTS.intersection(values))
        if refused:
            raise TypeError(f"Settings takes no {', '.join(refused)}: of the underscore arguments only _env_file")
        env_file = values.pop("_env_file") if "_env_file" in values else self.model_config.get("env_file")
        super().__init__(_env_file=None, **(_load(type(self), env_file) | values))

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Return ``init_settings`` only, switching the environment, dotenv and secrets-directory sources off.

        ``__init__`` passes the settings ``_load`` read as constructor arguments, so ``init_settings`` holds them and
        ``_load`` stays the only reader of the environment and files.

        Preconditions:
            Called by pydantic-settings while constructing ``settings_cls``.

        Args:
            settings_cls: The class being constructed.
            init_settings: The constructor's keyword arguments.
            env_settings: The process environment; not used.
            dotenv_settings: The dotenv source; not used.
            file_secret_settings: The secrets-directory source; not used.

        Returns:
            ``(init_settings,)``.

        Raises:
            None.

        """
        return (init_settings,)

    @field_validator("warehouse_url")
    @classmethod
    def _check_warehouse_url(cls, value: SecretStr) -> SecretStr:
        """Reject a ``warehouse_url`` with another scheme, or one that sets a connection parameter the pool sets.

        Only the two schemes libpq accepts are allowed, so a URL for anything else (``ftp://x``) fails here, at
        startup, instead of at the first connection. The URL is then parsed by libpq's own parser
        (``psycopg.conninfo.conninfo_to_dict``), so its parameters are the ones libpq will read, whatever the URL's
        other parts hold (an unencoded ``?`` in the password included). ``db.pool.open_pool`` sets every parameter in
        ``_POOL_PARAMETERS`` itself, replacing the URL's value, so a URL that sets one
        (``?options=-c%20search_path%3Dwarehouse``) would lose it unnoticed.

        Preconditions:
            None.

        Args:
            value: The ``warehouse_url`` value after ``SecretStr`` validation.

        Returns:
            ``value`` unchanged.

        Raises:
            ValueError: If the secret value does not match ``_WAREHOUSE_URL_PATTERN``, for example ``"localhost"`` or
                ``"ftp://x"``. Pydantic reports it as a ``ValidationError`` for ``warehouse_url``. The message does
                not contain the value.
            ValueError: If the URL sets a parameter in ``_POOL_PARAMETERS``, even to an empty value, e.g.
                ``warehouse_url may not set keepalives_idle, options: the connection pool sets them``; the message
                names the parameters, sorted, and holds no part of the URL. Pydantic reports it as a
                ``ValidationError`` for ``warehouse_url``.
            ValueError: If libpq cannot parse the URL (an unknown query parameter, a query parameter without ``=``,
                an unclosed IPv6 bracket) or it cannot be encoded as UTF-8: ``warehouse_url is not a connection URL
                libpq accepts``. The parser's own error is dropped, as its message may hold the URL. Pydantic
                reports it as a ``ValidationError`` for ``warehouse_url``.

        """
        if _WAREHOUSE_URL_PATTERN.match(value.get_secret_value()) is None:
            raise ValueError("warehouse_url must start with 'postgresql://' or 'postgres://'")
        try:
            names = conninfo_to_dict(value.get_secret_value()).keys()
        except psycopg.Error, UnicodeEncodeError:  # its message may hold the URL, password included
            raise ValueError("warehouse_url is not a connection URL libpq accepts") from None
        if pool_set := sorted(names & _POOL_PARAMETERS):
            raise ValueError(f"warehouse_url may not set {', '.join(pool_set)}: the connection pool sets them")
        return value

    @field_validator("worker_id")
    @classmethod
    def _check_worker_id(cls, value: str) -> str:
        """Reject a ``worker_id`` longer than ``_WORKER_ID_MAX_BYTES`` bytes in UTF-8.

        ``db.control`` sets ``application_name`` to ``"<worker_id>-listen"``, and Postgres cuts an
        ``application_name`` at 63 bytes, so a longer ``worker_id`` would lose the suffix.

        Preconditions:
            None.

        Args:
            value: The ``worker_id`` value after stripping and the length-one minimum.

        Returns:
            ``value`` unchanged.

        Raises:
            ValueError: If ``value.encode("utf-8")`` is longer than ``_WORKER_ID_MAX_BYTES`` bytes. Pydantic reports
                it as a ``ValidationError`` for ``worker_id``. The message does not contain the value.

        """
        if len(value.encode("utf-8")) > _WORKER_ID_MAX_BYTES:
            raise ValueError(f"worker_id must be at most {_WORKER_ID_MAX_BYTES} bytes in UTF-8")
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
