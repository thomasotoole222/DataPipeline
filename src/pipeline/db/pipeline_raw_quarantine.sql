CREATE SCHEMA raw;
COMMENT ON SCHEMA raw IS
    'Source pages as fetched: each page''s envelope in raw.page, its hashed part once in raw.payload. Purged by control.purge_raw.';

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
    PRIMARY KEY (task_id, page_no)                    -- a task fetches one endpoint, so the task's pages are its key
);
CREATE INDEX page_payload ON raw.page (payload_hash);  -- control.purge_raw finds the payloads no page references
CREATE INDEX page_fetched ON raw.page (fetched_at);    -- control.purge_raw finds the pages past the retention
COMMENT ON TABLE raw.page IS
    'Source responses from every endpoint, without their hashed part. Grain: one page per task. Writer: pipeline Extract; control.purge_raw. Layer: raw.';

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
