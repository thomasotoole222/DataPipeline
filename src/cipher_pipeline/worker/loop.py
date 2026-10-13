import asyncio
import contextlib
import logging
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final, Protocol

import psycopg
from opentelemetry import propagate, trace
from opentelemetry.context import Context

from cipher_pipeline import telemetry
from cipher_pipeline.config import Context
from cipher_pipeline.db import control
from cipher_pipeline.db.control import ClaimedTask, LeaseLoss, TaskReport
from cipher_pipeline.errors import (
    Interrupted,
    JobNotRunnable,
    LeaseConflict,
    QualityFailed,
    UnknownJob,
    classify
)
from cipher_pipeline.telemetry import Instruments
from cipher_pipeline.timeutil import utc_now
from cipher_pipeline.worker.lease import HEARTBEAT_INTERVAL, SILENCE_LIMIT, Lease

