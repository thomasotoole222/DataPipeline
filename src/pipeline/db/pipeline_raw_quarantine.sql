CREATE SCHEMA raw;
COMMENT ON SCHEMA raw IS
    'Source pages as fetched: each page''s envelope in raw.page, its hashed part once in raw.payload. Archived by pipeline.archive, then purged by control.purge_raw.';

CREATE TABLE raw.payload (
    hash        bytea       PRIMARY KEY,              -- sha256 of body's canonical JSON (raw.save_page)
    body        jsonb       NOT NULL                  -- the hashed part: the connector's hashed keys and their values
);
COMMENT ON TABLE raw.payload IS
    'Hashed parts of source responses, each stored once. Grain: one row per distinct hashed part. Writer: pipeline Extract; control.purge_raw. Layer: raw.';

CREATE TABLE raw.page (
    task_id      bigint      NOT NULL,                -- control.task id; no foreign key, so raw does not depend on control
    page_no      integer     NOT NULL CHECK (page_no >= 0),
    connector    text        NOT NULL,                -- the connector that fetched it, e.g. 'tiktok_shop'
    endpoint     text        NOT NULL,                -- the endpoint within the connector, e.g. 'orders_search'
    cursor_in    text,                                -- cursor sent; NULL on the first page
    cursor_next  text,                                -- cursor for the next page; NULL on the last page
    fetched_at   timestamptz NOT NULL,
    envelope     jsonb       NOT NULL,                -- the response without its hashed keys, e.g. code and request_id
    payload_hash bytea       NOT NULL REFERENCES raw.payload (hash),  -- the response's hashed part
    ids          text[]                               -- the ids a fanned-out call was for; NULL: not per id
                 CHECK (cardinality(ids) > 0 AND array_ndims(ids) = 1 AND array_position(ids, NULL) IS NULL),
    error_class  text        CHECK (error_class <> ''),  -- a failed fanned-out call's error class; its page is a marker
    CHECK (error_class IS NULL OR ids IS NOT NULL),
    archived_month date      CHECK (extract(day FROM archived_month) = 1),  -- set once the page is in the archive
    PRIMARY KEY (task_id, page_no)                    -- numbered across the task: list pages, then per-id pages
);
CREATE INDEX page_payload ON raw.page (payload_hash);  -- control.purge_raw finds the payloads no page references
CREATE INDEX page_fetched ON raw.page (fetched_at);    -- control.purge_raw finds the pages past the retention
COMMENT ON TABLE raw.page IS
    'Source responses from every endpoint, without their hashed part. Grain: one row per page of a task. Writer: pipeline Extract; archived_month by pipeline.archive; control.purge_raw. Layer: raw.';

-- The month a task's pages are archived under, as its first day: the month of the task's day key for a day partition
-- ('YYYY-MM-DD'), otherwise the UTC month the task was created in, which is the day a snapshot job fetched.
CREATE FUNCTION raw.data_month(partition_key text, created_at timestamptz) RETURNS date
LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS $$
    SELECT CASE WHEN partition_key ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
                THEN make_date(left(partition_key, 4)::int, substr(partition_key, 6, 2)::int, 1)
                ELSE date_trunc('month', created_at AT TIME ZONE 'UTC')::date END
$$;

-- One archived Parquet object per job and month, written once: UNIQUE (job, month) refuses a second.
CREATE TABLE raw.archive_manifest (
    job        text        NOT NULL,
    month      date        NOT NULL CHECK (extract(day FROM month) = 1),  -- the month's first day
    object_key text        NOT NULL UNIQUE,                               -- e.g. 'raw/month=2026-05/job=orders.parquet'
    rows       bigint      NOT NULL CHECK (rows > 0),                     -- pages in the object, one row each
    bytes      bigint      NOT NULL CHECK (bytes > 0),
    sha256     bytea       NOT NULL CHECK (octet_length(sha256) = 32),    -- of the object's bytes
    written_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (job, month)
);
COMMENT ON TABLE raw.archive_manifest IS
    'The raw archive''s objects. Grain: one row per job and month. Writer: pipeline.archive. Layer: raw.';

CREATE SCHEMA quarantine;
COMMENT ON SCHEMA quarantine IS 'Records held back by validation until they are replayed, released or discarded.';

CREATE TABLE quarantine.row (
    id            bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    task_id       bigint      NOT NULL,
    connector     text        NOT NULL,
    job           text        NOT NULL,
    source_id     uuid        NOT NULL,
    partition_key text        NOT NULL,
    raw_page_no   integer     NOT NULL CHECK (raw_page_no >= 0),  -- with task_id, the raw.page row it came from
    raw_index     integer     NOT NULL CHECK (raw_index >= 0),
    group_key     text,                               -- records held together, e.g. an order and its line items
    payload       jsonb       NOT NULL,               -- the record as landed
    failures      jsonb       NOT NULL CHECK (jsonb_typeof(failures) = 'array' AND jsonb_array_length(failures) > 0),
    status        text        NOT NULL DEFAULT 'open'
                  CHECK (status IN ('open', 'replayed', 'released', 'discarded')),
    resolved_by   text,
    resolved_at   timestamptz,
    note          text,
    held_at       timestamptz NOT NULL DEFAULT now(),
    CHECK ((status = 'open') = (resolved_at IS NULL))
);
CREATE INDEX row_open ON quarantine.row (connector, job, source_id) WHERE status = 'open';
CREATE INDEX row_failures ON quarantine.row USING gin (failures jsonb_path_ops);
COMMENT ON TABLE quarantine.row IS
    'Records that failed an error rule. Grain: one row per held record. Writer: pipeline Validate; status by control API. Layer: quarantine.';
