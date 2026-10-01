"""Shared pytest configuration.

Investigations write to operational memory. Tests must never write to the real
``data/operational_memory.db``, so the database path is redirected to a session
temporary file for the whole run. Individual tests can still pass an explicit
``memory_db`` path when they need a completely fresh database.
"""

import os
import tempfile

_SESSION_DB = os.path.join(tempfile.mkdtemp(prefix="opmem-tests-"), "test_memory.db")

os.environ["OPERATIONAL_MEMORY_DB"] = _SESSION_DB