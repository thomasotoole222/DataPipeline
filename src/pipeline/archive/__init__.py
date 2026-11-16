"""The raw archive: one Parquet object per job per month in a bucket, written once that month has settled.

Public API:
    months: Plan and write each job's months, and check the manifest against the store.
    store: The store the objects are kept in: an S3-compatible bucket, or a local directory in tests.

"""
