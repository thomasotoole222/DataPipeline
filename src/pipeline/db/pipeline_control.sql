-- Create the warehouse's control schema: the task queue workers claim from and the log of leases they gave up.
--
-- Public API:
--     control.task: One row per unit of work, with its state, its current lease and its last report.
--     control.lease_loss: One row per lease a worker gave up.
--     control.claim: Claim the next due pending or retry_wait task for a worker.
--     control.heartbeat: Set a running task's lease to expire 5 minutes after the row is updated.
--     control.fence: Lock a running task's row for the caller's transaction and extend its lease to at least 5
--         minutes after that transaction's start.
--     control.report_outcome: Record an attempt's outcome, release the lease and return the task's new state.
--     control.record_lease_loss: Log a lease a worker gave up.
--     control.expire_leases: Release every expired lease whose row is not locked, log each one and count them.
--     control.purge_raw: Delete the raw pages past a retention whose task is not open, then the payloads left unused.
--     control.notify_task: Trigger function that notifies channel control_task with the task's id.
--     task_notify: Trigger that runs control.notify_task when a task is inserted as, or moved to, pending.
--
-- Create the control schema that holds every object below.
--
CREATE SCHEMA control;
COMMENT ON SCHEMA control IS 'The task queue the workers claim from, and the record of every lease lost.';

-- Hold one unit of work: one job for one source of one connector and one partition, inside one run.
--
-- state moves pending or retry_wait -> running (claim, once not_before has passed); running -> pending, retry_wait,
-- blocked, succeeded, failed or dead_lettered (report_outcome); running -> pending or dead_lettered (expire_leases).
-- Nothing on this page moves a task out of blocked, or into cancelled. claim sets lease_token, lease_owner and
-- lease_expires_at; heartbeat and fence reset lease_expires_at; report_outcome and expire_leases clear all three, and
-- only report_outcome copies the released token to last_lease_token. heartbeat_at is set by heartbeat and cleared by
-- claim. started_at is set by claim; finished_at is set by report_outcome and expire_leases and cleared by claim. The
-- report columns (error_class to duration_ms) are written by report_outcome and kept by claim; expire_leases sets
-- error_class to 'LeaseExpired' and clears error_type, error_message, the row counts and duration_ms. Constraints:
-- state is one of the eight listed values, not_before and created_at are finite, and state is 'running' exactly when
-- lease_token and lease_expires_at are both set, so every running task has a lease that expire_leases can see.
-- lease_owner is not constrained. max_attempts is 1 to 100 and attempt is 0 to max_attempts; claim takes only a task
-- with attempts left, so no claim or backoff can overflow, and a row with no attempts left waits for a person instead
-- of blocking the queue. priority is unchecked.
--
CREATE TABLE control.task (
    id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id           bigint      NOT NULL,
    connector        text        NOT NULL,
    job              text        NOT NULL,
    source_id        uuid        NOT NULL,
    partition_key    text        NOT NULL,
    state            text        NOT NULL DEFAULT 'pending'
        CHECK (state IN ('pending', 'running', 'retry_wait', 'blocked',
                         'succeeded', 'failed', 'dead_lettered', 'cancelled')),
    priority         int         NOT NULL DEFAULT 0,          -- lower runs first
    attempt          int         NOT NULL DEFAULT 0,          -- claims counted so far
    max_attempts     int         NOT NULL DEFAULT 5 CHECK (max_attempts BETWEEN 1 AND 100),
    not_before       timestamptz NOT NULL DEFAULT now()       -- not claimable before this
        CHECK (isfinite(not_before)),
    -- Lease: set by claim, extended by heartbeat and fence, cleared by report_outcome and expire_leases.
    lease_token      uuid,
    lease_owner      text,
    lease_expires_at timestamptz,
    heartbeat_at     timestamptz,
    -- The last report. last_lease_token is the token report_outcome released; expire_leases rewrites error_class to
    -- duration_ms but not last_lease_token.
    last_lease_token uuid,
    error_class      text,                                    -- the category, e.g. 'Unexpected' or 'TransientError'
    error_type       text,                                    -- the exception's own type, e.g. 'RecursionError'
    error_message    text,
    rows_fetched     bigint,
    rows_written     bigint,
    rows_rejected    bigint,
    duration_ms      bigint,
    trace_context    text,                                    -- W3C traceparent written by the Orchestrator
    created_at       timestamptz NOT NULL DEFAULT now() CHECK (isfinite(created_at)),
    started_at       timestamptz,                             -- the last claim
    finished_at      timestamptz,                             -- the last report
    CHECK ((state = 'running') = (lease_token IS NOT NULL AND lease_expires_at IS NOT NULL)),
    CHECK (attempt BETWEEN 0 AND max_attempts)
);

COMMENT ON TABLE control.task IS
    'Units of work. Grain: one row per task. Writer: task creation, and the control functions for state. Layer: control.';

-- Serve claim's pick: pending and retry_wait rows in (priority, created_at) order. not_before is filtered row by row,
-- not indexed.
--
CREATE INDEX task_claimable ON control.task (priority, created_at) WHERE state IN ('pending', 'retry_wait');

-- Allow at most one task in pending, running, retry_wait or blocked per connector, job, source_id and partition_key.
-- A second such row raises unique_violation; succeeded, failed, dead_lettered and cancelled rows are not limited.
--
CREATE UNIQUE INDEX task_live ON control.task (connector, job, source_id, partition_key)
    WHERE state IN ('pending', 'running', 'retry_wait', 'blocked');

-- Log one row each time a worker gave up a lease it could no longer prove it held, written by record_lease_loss, and
-- one each time expire_leases released a lease that ran out, kind expired.
--
-- task_id must name an existing task and kind is heartbeat, silence, commit or expired; worker and unwind_ms may be
-- NULL, and at defaults to the inserting transaction's start. There is no key: the same loss recorded twice is two
-- rows.
--
CREATE TABLE control.lease_loss (
    task_id     bigint      NOT NULL REFERENCES control.task (id),
    lease_token uuid        NOT NULL,
    kind        text        NOT NULL CHECK (kind IN ('heartbeat', 'silence', 'commit', 'expired')),
    worker      text,                                         -- the worker that held the lease
    unwind_ms   bigint,                                       -- how long the task took to unwind; NULL if unknown
    at          timestamptz NOT NULL DEFAULT now()
);
COMMENT ON TABLE control.lease_loss IS
    'Leases lost. Grain: one row per lease given up or expired. Writer: control.record_lease_loss, control.expire_leases. Layer: control.';

-- Claim the next due pending or retry_wait task for a worker, or return no row.
--
-- Picks one row with state 'pending' or 'retry_wait', not_before <= now() and attempts left, lowest priority first,
-- then oldest created_at; rows tied on both (inserted in one transaction share created_at) come in no defined order.
-- Only a row with attempt < max_attempts is taken. FOR UPDATE SKIP LOCKED lets concurrent claimers take different rows
-- without waiting; under READ COMMITTED a row another claimer claimed and committed first fails the recheck of state,
-- and the next row is taken. The row becomes running: attempt goes up by 1, lease_token is a new gen_random_uuid(),
-- lease_owner is worker, started_at is now(), lease_expires_at is now() + 5 minutes, and heartbeat_at and finished_at
-- are cleared. now() is the calling transaction's start.
--
-- Preconditions:
--     None. worker is not checked; NULL is stored as a NULL lease_owner.
--
-- Args:
--     worker: ID of the claiming worker, stored as lease_owner.
--
-- Returns:
--     No row, or one row of the claimed task's columns with started_at as claimed_at. lease_expires_at - claimed_at
--     is exactly 5 minutes, so the positive lease ClaimedTask requires always holds.
--
-- Raises:
--     Other database errors propagate.
--
CREATE FUNCTION control.claim(worker text)
RETURNS TABLE (
    id bigint, run_id bigint, connector text, job text, source_id uuid, partition_key text, attempt int,
    lease_token uuid, created_at timestamptz, not_before timestamptz, claimed_at timestamptz,
    lease_expires_at timestamptz, trace_context text
)
LANGUAGE sql AS $$
    UPDATE control.task AS t
       SET state            = 'running',
           attempt          = t.attempt + 1,
           lease_token      = gen_random_uuid(),
           lease_owner      = worker,
           lease_expires_at = now() + interval '5 minutes',
           heartbeat_at     = NULL,
           started_at       = now(),
           finished_at      = NULL
     WHERE t.id = (
               SELECT q.id
                 FROM control.task AS q
                WHERE q.state IN ('pending', 'retry_wait') AND q.not_before <= now() AND q.attempt < q.max_attempts
                ORDER BY q.priority, q.created_at
                  FOR UPDATE SKIP LOCKED
                LIMIT 1
           )
    RETURNING t.id, t.run_id, t.connector, t.job, t.source_id, t.partition_key, t.attempt,
              t.lease_token, t.created_at, t.not_before, t.started_at, t.lease_expires_at, t.trace_context
$$;

-- Extend a task's lease to clock_timestamp() + 5 minutes and set heartbeat_at = clock_timestamp() if it is running
-- under lease_token.
--
-- The row is locked first: if another transaction holds it (a fenced Load), the call waits until that ends, then
-- updates the row if it is still running under lease_token. clock_timestamp() is read in the update, after any wait, so
-- a true result means the lease runs for 5 minutes from about when the call returned; control.fence never shortens it.
-- lease_expires_at is not compared with now(): a running lease that has expired but that expire_leases has not yet
-- released is extended, and the call returns true.
--
-- Preconditions:
--     None.
--
-- Args:
--     task_id: ID of the task.
--     lease_token: Token the caller's lease was issued with.
--
-- Returns:
--     True if the row was running under lease_token and was extended. False otherwise: no such task, another state or
--     token, or a NULL argument. Never NULL.
--
-- Raises:
--     Nothing of its own; database errors propagate.
--
CREATE FUNCTION control.heartbeat(task_id bigint, lease_token uuid)
RETURNS boolean
LANGUAGE plpgsql AS $$
BEGIN
    -- Wait for any transaction holding the row (a fenced Load) first, so the clock is read after the wait.
    PERFORM 1 FROM control.task AS t WHERE t.id = heartbeat.task_id FOR UPDATE;
    UPDATE control.task AS t
       SET lease_expires_at = clock_timestamp() + interval '5 minutes',
           heartbeat_at     = clock_timestamp()
     WHERE t.id = heartbeat.task_id
       AND t.state = 'running'
       AND t.lease_token = heartbeat.lease_token;
    RETURN FOUND;
END
$$;

-- Check that a task is running under lease_token, lock its row for the caller's transaction and extend the lease.
--
-- One UPDATE sets lease_expires_at = greatest(lease_expires_at, now() + 5 minutes) where the task is running under
-- lease_token, so it never moves the expiry earlier; heartbeat_at is not set. The row lock lasts until the caller's
-- transaction ends, so expire_leases (SKIP LOCKED) passes over the task and heartbeat and report_outcome from other
-- transactions wait. now() is the transaction's start, so the extension counts from then: Lease.fenced runs the fence
-- as its transaction's first statement, and a transaction that stays open longer than 5 minutes commits with the lease
-- already expired. The extension commits or rolls back with the transaction. As in heartbeat, lease_expires_at is not
-- compared with now(), so an expired lease not yet released is still held. When no row matches, a second read (a new
-- snapshot, after any wait for the row lock) decides the error: CH002 if last_lease_token is lease_token, else CH001.
-- Every fenced transaction (Load's, and each of Extract's) runs it first, so a stale worker fails before writing or
-- locking any data row.
--
-- Preconditions:
--     To guard later writes, the caller runs them in the same transaction; on an autocommit connection the lock and
--     the check end with the statement.
--
-- Args:
--     task_id: ID of the task.
--     lease_token: Token the caller's lease was issued with.
--
-- Returns:
--     Nothing (void).
--
-- Raises:
--     CH002: The task's last_lease_token is lease_token, so report_outcome already released this lease.
--     CH001: Otherwise when the task is missing, not running, or running under another token (or lease_token is
--         NULL).
--
CREATE FUNCTION control.fence(task_id bigint, lease_token uuid)
RETURNS void
LANGUAGE plpgsql AS $$
BEGIN
    UPDATE control.task AS t
       SET lease_expires_at = greatest(t.lease_expires_at, now() + interval '5 minutes')
     WHERE t.id = fence.task_id
       AND t.state = 'running'
       AND t.lease_token = fence.lease_token;
    IF NOT FOUND THEN
        IF EXISTS (SELECT FROM control.task AS t
                    WHERE t.id = fence.task_id AND t.last_lease_token = fence.lease_token) THEN
            RAISE EXCEPTION 'task % already reported under lease %', fence.task_id, fence.lease_token
                USING ERRCODE = 'CH002';
        END IF;
        RAISE EXCEPTION 'task % is not running under lease %', fence.task_id, fence.lease_token
            USING ERRCODE = 'CH001';
    END IF;
END
$$;

-- Record an attempt's outcome under a lease, release the lease and return the task's new state.
--
-- Locks the task row FOR UPDATE (inside Load's transaction control.fence already holds it). Raises CH002 when
-- last_lease_token is lease_token, else CH001 when the task is missing, not running, or running under another token,
-- so a worker that lost its lease cannot overwrite the new holder's work; called last inside Load's transaction, the
-- raise rolls back that transaction's writes too. As in fence, lease_expires_at is not compared with now(): a lease
-- that expired but that expire_leases has not yet released is still reported under. Otherwise the next state is,
-- checked in this order:
--     error_class = 'Interrupted': 'pending', attempt reduced by 1, not_before kept, whatever outcome is.
--     outcome 'succeeded', 'failed' or 'blocked': that state.
--     outcome 'retryable': 'dead_lettered' when attempt >= max_attempts, else 'retry_wait' with not_before = now() +
--         least(greatest(least(30 s * 2 ^ least(attempt - 1, 7), 1 hour), coalesce(retry_after, 0)), 6 hours); the
--         exponent is capped at 7 (64 minutes, past the 1-hour cap) so the product never overflows interval.
--     outcome 'unexpected': 'retry_wait' with not_before = now() + 30 s when attempt <= 1 and attempt < max_attempts,
--         else 'failed'.
-- The row then stores the new state, attempt, the report's columns and finished_at = now(), clears lease_token,
-- lease_owner and lease_expires_at, and keeps the released token in last_lease_token.
--
-- Preconditions:
--     outcome is an Outcome value and error_class is NULL exactly when outcome is 'succeeded', as TaskReport
--     guarantees, so the Interrupted check never overrides a success. Nothing else is checked: the row counts and
--     duration_ms are stored as given.
--
-- Args:
--     task_id: ID of the task.
--     lease_token: Token the caller's lease was issued with.
--     outcome: 'succeeded', 'retryable', 'blocked', 'failed' or 'unexpected'.
--     error_class: Error class of an unsuccessful run, or NULL; 'Interrupted' marks a shutdown.
--     error_type: Type name of the exception that ended the run, or NULL; stored as given.
--     error_message: Error text, or NULL.
--     retry_after: Smallest wait the source asked for, or NULL; read only for outcome 'retryable'.
--     rows_fetched: Rows fetched, stored as given.
--     rows_written: Rows written, stored as given.
--     rows_rejected: Rows rejected, stored as given.
--     duration_ms: Run time in milliseconds, stored as given.
--
-- Returns:
--     The task's new state: pending, succeeded, failed, blocked, retry_wait or dead_lettered.
--
-- Raises:
--     CH002: An outcome was already recorded under lease_token.
--     CH001: The task is missing, not running, or running under another token (or task_id or lease_token is NULL).
--     invalid_parameter_value (22023): outcome is not one of the five values and error_class is not 'Interrupted'.
--
CREATE FUNCTION control.report_outcome(
    task_id bigint, lease_token uuid, outcome text, error_class text, error_type text, error_message text,
    retry_after interval, rows_fetched bigint, rows_written bigint, rows_rejected bigint, duration_ms bigint
)
RETURNS text
LANGUAGE plpgsql AS $$
DECLARE
    cur        control.task%ROWTYPE;
    next_state text;
    wait       interval := interval '0';
BEGIN
    SELECT * INTO cur FROM control.task AS t WHERE t.id = report_outcome.task_id FOR UPDATE;
    IF FOUND AND cur.last_lease_token = report_outcome.lease_token THEN
        RAISE EXCEPTION 'task % already reported under lease %', report_outcome.task_id, report_outcome.lease_token
            USING ERRCODE = 'CH002';
    END IF;
    IF NOT FOUND OR cur.state <> 'running' OR cur.lease_token IS DISTINCT FROM report_outcome.lease_token THEN
        RAISE EXCEPTION 'task % is not running under lease %', report_outcome.task_id, report_outcome.lease_token
            USING ERRCODE = 'CH001';
    END IF;

    -- Map the outcome to the next state.
    IF report_outcome.error_class = 'Interrupted' THEN
        -- The worker stopped before finishing (shutdown): run again at once; the attempt is not counted.
        next_state := 'pending';
        cur.attempt := cur.attempt - 1;
    ELSIF outcome IN ('succeeded', 'failed', 'blocked') THEN
        next_state := outcome;
    ELSIF outcome = 'retryable' THEN
        IF cur.attempt >= cur.max_attempts THEN
            next_state := 'dead_lettered';
        ELSE
            next_state := 'retry_wait';
            -- 30 s doubling per attempt, capped at 1 hour, but never sooner than the source asked, up to 6 hours.
            wait := least(greatest(least(interval '30 seconds' * 2 ^ least(cur.attempt - 1, 7), interval '1 hour'),
                                   coalesce(retry_after, interval '0')),
                          interval '6 hours');
        END IF;
    ELSIF outcome = 'unexpected' THEN
        -- One more try on the first attempt, after the first backoff step, if max_attempts allows it; then a person
        -- looks at it.
        next_state := CASE WHEN cur.attempt <= 1 AND cur.attempt < cur.max_attempts THEN 'retry_wait' ELSE 'failed' END;
        wait := interval '30 seconds';
    ELSE
        RAISE EXCEPTION 'unknown outcome %', outcome USING ERRCODE = 'invalid_parameter_value';
    END IF;

    UPDATE control.task AS t
       SET state            = next_state,
           attempt          = cur.attempt,
           not_before       = CASE WHEN next_state = 'retry_wait' THEN now() + wait ELSE t.not_before END,
           lease_token      = NULL,
           last_lease_token = report_outcome.lease_token,
           lease_owner      = NULL,
           lease_expires_at = NULL,
           error_class      = report_outcome.error_class,
           error_type       = report_outcome.error_type,
           error_message    = report_outcome.error_message,
           rows_fetched     = report_outcome.rows_fetched,
           rows_written     = report_outcome.rows_written,
           rows_rejected    = report_outcome.rows_rejected,
           duration_ms      = report_outcome.duration_ms,
           finished_at      = now()
     WHERE t.id = report_outcome.task_id;
    RETURN next_state;
END
$$;

-- Insert one control.lease_loss row with the given values and at = now(). The task row is not touched.
--
-- Preconditions:
--     None. lease_token is not checked against the task: this is a log, not a fenced write.
--
-- Args:
--     task_id: ID of the task whose lease was given up.
--     lease_token: Token of the lease given up.
--     kind: 'heartbeat', 'silence', 'commit' or 'expired'.
--     worker: ID of the worker that gave it up, or NULL.
--     unwind_ms: Time the task took to unwind in milliseconds, or NULL if unknown; stored as given.
--
-- Returns:
--     Nothing (void).
--
-- Raises:
--     foreign_key_violation (23503): task_id names no task.
--     check_violation (23514): kind is not one of the four values.
--     not_null_violation (23502): task_id, lease_token or kind is NULL.
--
CREATE FUNCTION control.record_lease_loss(task_id bigint, lease_token uuid, kind text, worker text, unwind_ms bigint)
RETURNS void
LANGUAGE sql AS $$
    INSERT INTO control.lease_loss (task_id, lease_token, kind, worker, unwind_ms)
    VALUES (task_id, lease_token, kind, worker, unwind_ms)
$$;

-- Release every running task whose lease expired and whose row is not locked, log each one, and return how many.
--
-- Matches rows with state 'running' and lease_expires_at < now(). SKIP LOCKED passes over a row another transaction
-- holds, such as a fenced Load, so a lease cannot be taken over while that transaction is open; under READ COMMITTED a
-- row renewed and committed meanwhile fails the recheck. Each released row becomes 'dead_lettered' when attempt >=
-- max_attempts, else 'pending'; either way not_before = now(), lease_token, lease_owner and lease_expires_at are
-- cleared, error_class is 'LeaseExpired', error_type, error_message, rows_fetched, rows_written, rows_rejected and
-- duration_ms are NULL, and finished_at is now(), so the report columns describe the attempt whose lease expired, not
-- an earlier one. attempt and last_lease_token keep their earlier values, so the released token is not in
-- last_lease_token and its holder gets CH001 from control.fence and control.report_outcome. Each released lease is
-- logged in control.lease_loss with kind 'expired', its token, its lease_owner as worker and a NULL unwind_ms, since a
-- worker that died cannot log its own loss. A row moved to 'pending' fires task_notify.
--
-- Preconditions:
--     None.
--
-- Args:
--     None.
--
-- Returns:
--     The number of tasks released; 0 when none.
--
-- Raises:
--     Nothing of its own; database errors propagate.
--
CREATE FUNCTION control.expire_leases()
RETURNS bigint
LANGUAGE sql AS $$
    WITH expired AS (
        SELECT q.id, q.lease_token, q.lease_owner
          FROM control.task AS q
         WHERE q.state = 'running' AND q.lease_expires_at < now()
           FOR UPDATE SKIP LOCKED
    ), released AS (
        UPDATE control.task AS t
           SET state            = CASE WHEN t.attempt >= t.max_attempts THEN 'dead_lettered' ELSE 'pending' END,
               not_before       = now(),
               lease_token      = NULL,
               lease_owner      = NULL,
               lease_expires_at = NULL,
               error_class      = 'LeaseExpired',
               error_type       = NULL,
               error_message    = NULL,
               rows_fetched     = NULL,
               rows_written     = NULL,
               rows_rejected    = NULL,
               duration_ms      = NULL,
               finished_at      = now()
          FROM expired AS e
         WHERE t.id = e.id
        RETURNING e.id, e.lease_token, e.lease_owner
    ), logged AS (
        INSERT INTO control.lease_loss (task_id, lease_token, kind, worker, unwind_ms)
        SELECT r.id, r.lease_token, 'expired', r.lease_owner, NULL FROM released AS r
        RETURNING 1
    )
    SELECT count(*) FROM logged
$$;

-- Delete the raw pages fetched more than retention ago whose task is not open, then every payload no page references.
--
-- A page goes when fetched_at < now() - retention and no control.task row with its task_id is pending, running or
-- blocked; a page whose task_id names no task goes too. Then every raw.payload row that no raw.page row references
-- goes, whatever its age. The two deletes are separate statements, so the second sees the first's deletions. now()
-- is the calling transaction's start. A page saved while the purge runs can find the payload it shares deleted
-- between raw.save_page's two statements; that save then fails with foreign_key_violation. Called by the
-- Orchestrator with the PIPELINE_RAW_RETENTION setting.
--
-- Preconditions:
--     raw.page and raw.payload exist, as pipeline_raw_quarantine.sql creates them. retention is not checked: a NULL
--     retention deletes no page, and a negative one deletes pages fetched up to now().
--
-- Args:
--     retention: How long a page is kept after it was fetched.
--
-- Returns:
--     One row: pages, the raw.page rows deleted, and payloads, the raw.payload rows deleted; 0 when none.
--
-- Raises:
--     Nothing of its own; database errors propagate.
--
CREATE FUNCTION control.purge_raw(retention interval)
RETURNS TABLE (pages bigint, payloads bigint)
LANGUAGE plpgsql AS $$
BEGIN
    DELETE FROM raw.page AS p
     WHERE p.fetched_at < now() - retention
       AND NOT EXISTS (SELECT FROM control.task AS t
                        WHERE t.id = p.task_id AND t.state IN ('pending', 'running', 'blocked'));
    GET DIAGNOSTICS pages = ROW_COUNT;
    DELETE FROM raw.payload AS y
     WHERE NOT EXISTS (SELECT FROM raw.page AS p WHERE p.payload_hash = y.hash);
    GET DIAGNOSTICS payloads = ROW_COUNT;
    RETURN NEXT;
END
$$;

-- Notify channel 'control_task' (TASK_CHANNEL) with NEW.id as text, delivered when the firing transaction commits.
--
-- Preconditions:
--     Called as a row trigger, so NEW is set.
--
-- Args:
--     None (trigger function).
--
-- Returns:
--     NULL, which an AFTER trigger ignores.
--
-- Raises:
--     Nothing of its own.
--
CREATE FUNCTION control.notify_task() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    PERFORM pg_notify('control_task', NEW.id::text);
    RETURN NULL;
END
$$;

-- Run control.notify_task after each row inserted as 'pending' and each update of state that leaves it 'pending'
-- (report_outcome for an Interrupted attempt, expire_leases).
--
-- not_before is not checked, so a task that is not yet due also wakes workers, whose claim then returns no row. A
-- retry_wait task whose wait has passed sends no notification: workers find it when their wait_for_task wait elapses.
--
CREATE TRIGGER task_notify
AFTER INSERT OR UPDATE OF state ON control.task
FOR EACH ROW WHEN (NEW.state = 'pending')
EXECUTE FUNCTION control.notify_task();
