CREATE SCHEMA raw;
COMMENT ON SCHEMA raw IS 'Source pages exactly as fetched, one table per endpoint. Hot for 5 days, then archived to Parquet.';

CREATE TABLE raw.tiktok_orders_search (
    task_id bigint  NOT NULL,
    page_no integer NOT NULL CHECK (page_no >= 0),
    cursor_in text,
    cursor_next text,
    fetched_at timestamptz NOT NULL,
    payload jsonb NOT NULL,
    PRIMARY KEY (task_id, page_no)
);
COMMENT ON TABLE raw.tiktok_orders_search IS
    'TikTok Shop orders search responses. Grain: one page per task. Writer: pipeline Extract. Layer: raw.';

CREATE SCHEMA quarantine;
COMMENT ON SCHEMA quarantine IS 'Records held back by validation until they are replayed, released or discarded.';

CREATE TABLE quarantine.row(
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    task_id bigint NOT NULL,
    job text NOT NULL,
    shop_id uuid NOT NULL,
    partition_key text NOT NULL,
    raw_table text NOT NULL,
    raw_page_no integer NOT NULL CHECK (raw_page_no >= 0),
    raw_index integer NOT NULL CHECK (raw_index >= 0),
    group_key text,
    payload jsonb NOT NULL,
    failures jsonb NOT NULL CHECK (jsonb_typeof(failures) = 'array' AND jsonb_array_length(failures) > 0),
    status text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'replayed', 'released', 'discarded')),
    resolved_by text,
    resolved_at timestamptz,
    note text,
    held_at timestamptz NOT NULL DEFAULT now(), CHECK ((status ='open') = (resolved_at IS NULL))
);
CREATE INDEX row_open ON quarantine.row (job, shop_id) WHERE status = 'open';
CREATE INDEX row_failures ON quarantine.row USING gin (failures jsonb_path_ops);
COMMENT ON TABLE quarantine.row IS
'Records that failed an error rule. Grain: one row per held record. Writer: pipeline Validate; status by control API. Layer: quarantine.';