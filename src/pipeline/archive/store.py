"""Where the raw archive's objects are kept: an S3-compatible bucket, or a local directory in tests.

Public API:
    StoredObject: An object's size and the SHA-256 stored with it.
    ObjectStore: Protocol for the store the archive writes to and reads back from.
    S3Store: An ``ObjectStore`` on an S3-compatible bucket (a Railway Storage Bucket).
    DirectoryStore: An ``ObjectStore`` on a local directory, for tests and local runs.
    store_from_settings: The ``S3Store`` a worker's settings describe, or ``None`` when they describe none.

"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

from pipeline.config import Settings

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

#: The object metadata key the hex SHA-256 of an object's bytes is stored under (``x-amz-meta-sha256``).
SHA256_KEY: Final = "sha256"


@dataclass(frozen=True, slots=True)
class StoredObject:
    """Describe an object as the store holds it.

    Attributes:
        size: The object's length in bytes.
        sha256: The hex SHA-256 stored with the object when it was written, or ``None`` if it has none.

    """

    size: int
    sha256: str | None


class ObjectStore(Protocol):
    """Define the store the archive writes objects to, reads them back from and checks.

    Every method blocks; the archive calls them in a worker thread. A key is a ``/``-separated path such as
    ``raw/month=2026-05/job=orders.parquet``.
    """

    def put(self, key: str, path: Path, sha256: str) -> None:
        """Store the file at ``path`` under ``key``, with ``sha256`` stored alongside it, replacing any object there.

        Preconditions:
            ``sha256`` is the hex SHA-256 of the file's bytes.

        Args:
            key: The object's key.
            path: The local file to upload.
            sha256: The hex SHA-256 of the file, stored with the object.

        Returns:
            None, once the whole object is stored.

        Raises:
            Exception: Whatever the store raises; nothing is stored under ``key`` unless it returns.

        """
        ...

    def head(self, key: str) -> StoredObject | None:
        """Return the object under ``key``'s size and stored SHA-256, or ``None`` if there is no such object.

        Preconditions:
            None.

        Args:
            key: The object's key.

        Returns:
            The object's ``StoredObject``, or ``None`` when ``key`` holds no object.

        Raises:
            Exception: Whatever the store raises for anything other than a missing object.

        """
        ...

    def get(self, key: str, path: Path) -> None:
        """Download the object under ``key`` to the file at ``path``, replacing it.

        Preconditions:
            The object exists.

        Args:
            key: The object's key.
            path: The local file to write.

        Returns:
            None, once the file holds the whole object.

        Raises:
            Exception: Whatever the store raises, a missing object included.

        """
        ...


class S3Store:
    """Keep objects in one S3-compatible bucket, through boto3.

    ``put`` uploads with boto3's managed transfer, which splits a large file into a multipart upload and aborts the
    upload if a part fails, so a failed ``put`` leaves no object. The SHA-256 is stored as user metadata
    (``x-amz-meta-sha256``), so ``head`` reads it back without downloading the object.

    Attributes:
        bucket: The bucket's name.

    """

    def __init__(self, client: S3Client, bucket: str) -> None:
        """Store the client and the bucket.

        Preconditions:
            None.

        Args:
            client: A boto3 S3 client for the bucket's endpoint and credentials.
            bucket: The bucket's name.

        Returns:
            None.

        Raises:
            Nothing.

        """
        self._client = client
        self.bucket = bucket

    def put(self, key: str, path: Path, sha256: str) -> None:
        """Upload ``path`` to ``key`` with its SHA-256 as metadata, as ``ObjectStore.put``.

        Preconditions:
            As ``ObjectStore.put``.

        Args:
            key: The object's key.
            path: The local file to upload.
            sha256: The hex SHA-256 of the file.

        Returns:
            None.

        Raises:
            boto3.exceptions.S3UploadFailedError: The upload failed.
            botocore.exceptions.BotoCoreError: The bucket could not be reached.

        """
        self._client.upload_file(
            str(path),
            self.bucket,
            key,
            ExtraArgs={"Metadata": {SHA256_KEY: sha256}, "ContentType": "application/vnd.apache.parquet"},
        )

    def head(self, key: str) -> StoredObject | None:
        """Return the object's size and stored SHA-256, or ``None`` when it does not exist, as ``ObjectStore.head``.

        Preconditions:
            None.

        Args:
            key: The object's key.

        Returns:
            The object's ``StoredObject``, or ``None`` on a 404.

        Raises:
            botocore.exceptions.ClientError: Any error response but a 404.
            botocore.exceptions.BotoCoreError: The bucket could not be reached.

        """
        from botocore.exceptions import ClientError

        try:
            head = self._client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as err:
            if err.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
                return None
            raise
        return StoredObject(size=head["ContentLength"], sha256=head.get("Metadata", {}).get(SHA256_KEY))

    def get(self, key: str, path: Path) -> None:
        """Download ``key`` to ``path``, as ``ObjectStore.get``.

        Preconditions:
            As ``ObjectStore.get``.

        Args:
            key: The object's key.
            path: The local file to write.

        Returns:
            None.

        Raises:
            botocore.exceptions.ClientError: The object does not exist, or another error response.
            botocore.exceptions.BotoCoreError: The bucket could not be reached.

        """
        self._client.download_file(self.bucket, key, str(path))


class DirectoryStore:
    """Keep objects as files under one local directory; the SHA-256 goes in a ``<file>.sha256`` file beside each.

    For tests and local runs only: nothing here is atomic.

    Attributes:
        root: The directory objects are kept under.

    """

    def __init__(self, root: Path) -> None:
        """Store the root directory.

        Preconditions:
            None; the directory is created on the first ``put``.

        Args:
            root: The directory objects are kept under.

        Returns:
            None.

        Raises:
            Nothing.

        """
        self.root = root

    def put(self, key: str, path: Path, sha256: str) -> None:
        """Copy ``path`` to ``root / key`` and write its SHA-256 beside it, as ``ObjectStore.put``.

        Preconditions:
            As ``ObjectStore.put``.

        Args:
            key: The object's key.
            path: The local file to copy.
            sha256: The hex SHA-256 of the file.

        Returns:
            None.

        Raises:
            OSError: The copy or the write failed.

        """
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        target.with_name(f"{target.name}.{SHA256_KEY}").write_text(sha256)

    def head(self, key: str) -> StoredObject | None:
        """Return the file's size and stored SHA-256, or ``None`` when it does not exist, as ``ObjectStore.head``.

        Preconditions:
            None.

        Args:
            key: The object's key.

        Returns:
            The object's ``StoredObject``, or ``None``.

        Raises:
            OSError: The file exists but could not be read.

        """
        target = self.root / key
        if not target.is_file():
            return None
        digest = target.with_name(f"{target.name}.{SHA256_KEY}")
        return StoredObject(size=target.stat().st_size, sha256=digest.read_text() if digest.is_file() else None)

    def get(self, key: str, path: Path) -> None:
        """Copy ``root / key`` to ``path``, as ``ObjectStore.get``.

        Preconditions:
            As ``ObjectStore.get``.

        Args:
            key: The object's key.
            path: The local file to write.

        Returns:
            None.

        Raises:
            FileNotFoundError: There is no such object.

        """
        shutil.copyfile(self.root / key, path)


def store_from_settings(settings: Settings) -> S3Store | None:
    """Build the ``S3Store`` the five ``archive_`` settings describe, or return ``None`` when they are unset.

    ``Settings`` refuses a partial set, so either all five are set or none is. The client retries a failed call up to
    5 times in boto3's ``standard`` mode, signs with SigV4 and addresses the bucket virtual-hosted style
    (``<bucket>.<endpoint host>``), as Railway's buckets require.

    Preconditions:
        None.

    Args:
        settings: The worker's settings.

    Returns:
        The bucket's ``S3Store``, or ``None`` when ``archive_bucket`` is unset.

    Raises:
        ValueError: From boto3, for an endpoint URL it cannot use.

    """
    if (
        settings.archive_bucket is None
        or settings.archive_endpoint is None
        or settings.archive_access_key_id is None
        or settings.archive_secret_access_key is None
    ):
        return None
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        endpoint_url=settings.archive_endpoint,
        region_name=settings.archive_region,
        aws_access_key_id=settings.archive_access_key_id.get_secret_value(),
        aws_secret_access_key=settings.archive_secret_access_key.get_secret_value(),
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 5, "mode": "standard"},
            s3={"addressing_style": "virtual"},  # Railway buckets: the bucket is the endpoint's subdomain
        ),
    )
    return S3Store(client, settings.archive_bucket)
