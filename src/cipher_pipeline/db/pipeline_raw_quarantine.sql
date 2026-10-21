CREATE SCHEMA raw;
COMMENT ON SCHEMA raw IS 'Source pages exactly as fetched, in one table. Hot for 5 days, then archived to Parquet.';

CREATE TABLE raw.page (
    task_id     bigint      NOT NULL,                 -- control.task id; no foreign key, so raw does not depend on control
    page_no     integer     NOT NULL CHECK (page_no >= 0),
    endpoint    text        NOT NULL,                 -- the spec that fetched it, e.g. 'tiktok_shop.orders_search'
    cursor_in   text,                                 -- cursor sent; NULL on the first page
    cursor_next text,                                 -- cursor for the next page; NULL on the last page
    fetched_at  timestamptz NOT NULL,
    payload     jsonb       NOT NULL,                 -- the whole response, envelope included
    PRIMARY KEY (task_id, page_no)                    -- a task fetches one endpoint, so the task's pages are its key
);
COMMENT ON TABLE raw.page IS
    'Source responses from every endpoint. Grain: one page per task. Writer: pipeline Extract. Layer: raw.';

CREATE SCHEMA quarantine;
COMMENT ON SCHEMA quarantine IS 'Records held back by validation until they are replayed, released or discarded.';

CREATE TABLE quarantine.row (
    id            bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    task_id       bigint      NOT NULL,
    job           text        NOT NULL,
    shop_id       uuid        NOT NULL,
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
CREATE INDEX row_open ON quarantine.row (job, shop_id) WHERE status = 'open';
CREATE INDEX row_failures ON quarantine.row USING gin (failures jsonb_path_ops);
COMMENT ON TABLE quarantine.row IS
    'Records that failed an error rule. Grain: one row per held record. Writer: pipeline Validate; status by control API. Layer: quarantine.';
